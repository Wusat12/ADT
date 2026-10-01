import torch
import torchaudio
import torch.nn.functional as F
import partitura

from pathlib import Path
from typing import Dict, Set, Optional


MS_PER_FRAME = 10

# Thesis frontend configuration
N_MELS = 84
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 441  # 100 frames/second at 44.1 kHz
F_MIN = 20
F_MAX = 20000


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = 44100,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
    f_min: int = F_MIN,
    f_max: int = F_MAX,
) -> torch.Tensor:
    """Compute a genuine Mel spectrogram."""

    # Convert stereo to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        power=1.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="htk",
    )

    return transform(waveform)


def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = 44100,
) -> torch.Tensor:
    """Compute the thesis Log-Mel representation."""

    mel = compute_mel_spectrogram(waveform, sr=sr)

    return torch.log10(mel + 1.0)


def compute_pcen(
    mel: torch.Tensor,
    s: float = 0.025,
    alpha: float = 0.98,
    delta: float = 2.0,
    r: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Per-Channel Energy Normalization (PCEN).

    Input shape:
        (mel_bins, time)

    Output shape:
        (mel_bins, time)
    """

    if mel.ndim != 2:
        raise ValueError(
            f"Expected Mel spectrogram with shape (mel_bins, time), "
            f"got {tuple(mel.shape)}"
        )

    mel = torch.clamp(mel, min=0.0)

    # Exponential moving average of each Mel channel.
    M = torch.empty_like(mel)
    M[:, 0] = mel[:, 0]

    for t in range(1, mel.shape[1]):
        M[:, t] = s * mel[:, t] + (1.0 - s) * M[:, t - 1]

    pcen = (
        mel / (eps + M.pow(alpha)) + delta
    ).pow(r) - delta**r

    return pcen


def compute_pcen_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = 44100,
) -> torch.Tensor:
    """Compute the thesis PCEN-Mel representation."""

    mel = compute_mel_spectrogram(waveform, sr=sr)

    return compute_pcen(mel)


def compute_positive_delta(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the positive first-order temporal difference.

    The first frame has zero delta.
    Negative changes are discarded.
    """

    delta = torch.zeros_like(features)

    delta[:, 1:] = features[:, 1:] - features[:, :-1]

    return torch.clamp(delta, min=0.0)


def add_delta_features(features: torch.Tensor) -> torch.Tensor:
    """
    Concatenate static features with positive temporal differences.

    84 Mel bands -> 168 features.
    """

    delta = compute_positive_delta(features)

    return torch.cat([features, delta], dim=0)


def compute_frontend(
    waveform: torch.Tensor,
    sr: int = 44100,
    representation: str = "logmel",
) -> torch.Tensor:
    """
    Compute the complete thesis frontend.

    Pipeline:

        waveform
            -> 84-band Mel
            -> Log-Mel or PCEN-Mel
            -> positive first-order delta
            -> concatenate
            -> 168 features
    """

    if representation == "logmel":
        features = compute_log_mel_spectrogram(waveform, sr=sr)

    elif representation == "pcen":
        features = compute_pcen_mel_spectrogram(waveform, sr=sr)

    else:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            "Expected 'logmel' or 'pcen'."
        )

    return add_delta_features(features)


def readAudio(
    path: Path,
    accompaniment: Optional[Path] = None,
    representation: str = "logmel",
) -> torch.Tensor:
    """Read a WAV file and compute the thesis acoustic representation."""

    # Read the data into mono
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)

    # If accompaniment is set, add it to the waveform
    if accompaniment:
        accompaniment_waveform, _ = torchaudio.load(accompaniment)
        waveform += accompaniment_waveform.mean(dim=0)

    # Pad the waveform with zeroes to be divisible into 400-frame intervals
    samples = torch.tensor(waveform.shape[0])
    timeframes = 1 + torch.floor(samples / (sr // 100))
    padding = (torch.ceil(timeframes / 400) * 400 - 1) * (sr // 100) - samples
    waveform = F.pad(
        waveform,
        (0, int(padding)),
        mode="constant",
        value=0,
    )

    # Compute Log-Mel or PCEN-Mel + positive delta features
    features = compute_frontend(
        waveform,
        sr=sr,
        representation=representation,
    )

    # Return shape:
    # (timesteps, features, channel)
    return features.T.unsqueeze(-1)


def readAnnotations(
    path: Path,
    mapping: Dict[str, int],
    num_frames: int,
    num_labels: int,
) -> torch.Tensor:
    """Read an annotation file into a torch tensor."""

    # Store label and frame indices in lists
    frame_indices = []
    label_indices = []

    # Open the file
    with open(path, "r") as f:
        for line in f.readlines():
            # Parse line
            elements = line.strip().split(" ")
            if len(elements) == 1:
                elements = elements[0].split("\t")

            time, event = elements[0], elements[-1]

            # If the event ends in a "-" or a digit, remove it
            if event[-1] in ["-"] or event[-1].isnumeric():
                event = event[:-1]

            # Turn into frames and labels
            frame = int(round(float(time) / MS_PER_FRAME * 1000))
            label = mapping[event]

            if label is None:
                # Skip if label is None
                continue

            frame_indices.append(frame)
            label_indices.append(label)

    # Turn into torch tensor
    tensor = torch.sparse_coo_tensor(
        torch.tensor([frame_indices, label_indices]),
        torch.ones(len(frame_indices)),
        (num_frames, num_labels),
    ).to_dense()

    # If two events happen really close, combine them
    tensor = torch.min(torch.tensor(1.0), tensor)

    # Perform target widening
    adjacents = (
        F.max_pool1d(
            tensor.T,
            kernel_size=3,
            stride=1,
            padding=1,
        ).T
        - tensor
    )

    tensor += adjacents * 0.5

    return tensor


def readMidi(
    path: Path,
    mapping: Dict[str, int],
    num_frames: int,
    num_labels: int,
    vocabulary: Optional[Set[int]] = None,
) -> torch.Tensor:
    """Read a MIDI-annotation file into a torch tensor."""

    # Load the MIDI into list of notes
    notes = partitura.load_performance_midi(path)[0].note_array()

    # Store label and frame indices in lists
    frame_indices = []
    label_indices = []

    # Only print invalid encounters once
    seen_invalid_pitches = set()

    # Iterate every note
    for note in notes:
        time, pitch = note[0], note[4]

        # Skip if pitch is invalid
        if pitch not in mapping:
            if pitch not in seen_invalid_pitches:
                print(f"Encountered invalid pitch {pitch}")
                seen_invalid_pitches.add(pitch)
            continue

        if vocabulary is not None:
            vocabulary.add(pitch)

        # Turn into frames and labels
        frame = int(round(float(time) / MS_PER_FRAME * 1000))

        # Extract label from pitch
        label = mapping[pitch]

        if label is None:
            continue

        frame_indices.append(frame)
        label_indices.append(label)

    # Turn into torch tensor
    tensor = torch.sparse_coo_tensor(
        torch.tensor([frame_indices, label_indices]),
        torch.ones(len(frame_indices)),
        (num_frames, num_labels),
    ).to_dense()

    # If two events happen really close, combine them
    tensor = torch.min(torch.tensor(1.0), tensor)

    # Perform target widening
    adjacents = (
        F.max_pool1d(
            tensor.T,
            kernel_size=3,
            stride=1,
            padding=1,
        ).T
        - tensor
    )

    tensor += adjacents * 0.5

    return tensor