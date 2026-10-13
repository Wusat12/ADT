import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchvision import transforms, ops
from pathlib import Path
from typing import Tuple, List
import sys
sys.path.append("io/")


def load_dataset(path: Path):
    """ Load a stored TensorDataset. These are full pickles, so weights_only must be False on PyTorch >= 2.6. """
    return torch.load(path, weights_only=False)


def compute_normalization(train_paths: List[Path], device: str = "cpu", per_feature: bool = True) -> Tuple[torch.Tensor]:
    """ Compute the normalization terms, (mean, std), of the training dataset.

    Entries have shape (time, features, 1). With per_feature=True one mean/std is computed for every
    feature, otherwise a single scalar mean/std. Statistics are accumulated batch by batch, so the
    dataset never has to fit in memory. (device is kept for compatibility and unused.) """
    train_dataset = ConcatDataset(map(load_dataset, train_paths))
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=128, num_workers=0)

    dims = (0, 1, 3) if per_feature else (0, 1, 2, 3)
    total, total_sq, count = 0.0, 0.0, 0
    for feats, _ in train_loader:
        x = feats.to(torch.float64)
        total = total + x.sum(dim=dims)
        total_sq = total_sq + (x ** 2).sum(dim=dims)
        count += x.numel() // (x.shape[2] if per_feature else 1)

    mean = total / count
    std = (total_sq / count - mean ** 2).clamp(min=0.0).sqrt().clamp(min=1e-8)
    return mean.float(), std.float()


class FeatureNormalize(nn.Module):
    """ Normalize inputs of shape (batch, time, features, 1) with per-feature (or scalar) mean and std. """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


def create_transform(mean: torch.Tensor, std: torch.Tensor, channels_last: bool) -> transforms.Compose:
    """ Create a preprocessing transforms pipeline. """
    composition = [FeatureNormalize(mean=mean, std=std)]

    # Permute (batch, time, features, 1) -> (batch, 1, time, features)
    if channels_last:
        composition.append(ops.Permute((0, 3, 1, 2)))

    return transforms.Compose(composition)


def compute_infrequency_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights using training labels only.

    Supports:
        [T, 3]       - one recording or chunk
        [B, T, 3]    - batched chunks

    Keeps the existing inverse-entropy weighting formula.
    """
    positive_counts = None
    total_frames = 0

    for _, labels in dataset:
        labels = torch.as_tensor(labels, dtype=torch.float32)

        if labels.ndim not in (2, 3):
            raise ValueError(
                f"Unexpected label shape: {tuple(labels.shape)}"
            )

        labels = labels.reshape(-1, labels.shape[-1])

        if positive_counts is None:
            positive_counts = torch.zeros(
                labels.shape[-1], dtype=torch.float64
            )

        positive_counts += (labels == 1.0).sum(dim=0)
        total_frames += labels.shape[0]

    if total_frames == 0:
        raise ValueError("Cannot compute class weights: empty dataset.")

    probabilities = positive_counts / total_frames
    probabilities = probabilities.clamp(min=1e-10, max=1 - 1e-10)

    entropy = (
        -probabilities * torch.log(probabilities)
        -(1 - probabilities) * torch.log(1 - probabilities)
    )

    return (1.0 / entropy).float()


def invert_log_filter_spectrogram(log_filter_spectrogram: torch.Tensor, sr: int = 44100, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 441, f_min: int = 20, f_max: int = 20000, power: int = 1) -> torch.Tensor:
    """ Given a log logarithmically filterbank spectrogram, invert it, and return its waveform. """

    # Invert log magnitude
    log_filter_spectrogram = torch.pow(torch.tensor(10), log_filter_spectrogram) - 1
    
    # Remove filters from spectrogram
    filterbank = compute_log_filterbank(sr=sr, n_fft=n_fft, f_min=f_min, f_max=f_max)
    inverse_filterbank = torch.linalg.pinv(filterbank)
    spectrogram = inverse_filterbank @ log_filter_spectrogram

    # Apply Griffin-Lim transform
    griffin_lim_transform = torchaudio.transforms.GriffinLim(power=power, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    waveform = griffin_lim_transform(spectrogram)

    return waveform


def invert_mel_spectrogram(mel_spectrogram: torch.Tensor, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 441, n_mels: int = 84, f_min: int = 20, f_max: int = 20000, norm: str = "slaney", mel_scale: str = "htk", n_iter: int = 32, power: int = 1) -> torch.Tensor:
    """ Given a log mel spectrogram, invert it, and return its waveform. """

    # Sampling rate corresponds to 44.1 khz
    sr = 44100
    n_stft = int((n_fft//2) + 1)

    # Invert log magnitude
    mel_spectrogram = torch.pow(torch.tensor(10), mel_spectrogram) - 1
    
    # Remove mel-filters from spectrogram
    inverse_transform = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=n_stft, n_mels=n_mels, f_min=f_min, f_max=f_max, norm=norm, mel_scale=mel_scale)
    spectrogram = inverse_transform(mel_spectrogram)

    # Apply Griffin-Lim transform
    grifflim_transform = torchaudio.transforms.GriffinLim(power=power, n_iter=n_iter, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    waveform = grifflim_transform(spectrogram)

    return waveform


class MelRecordingDataset(torch.utils.data.Dataset):
    """
    Recording-level linear Mel dataset for trainable PCEN.

    One item = one complete recording.

    Expected directory structure:
        mel_recordings/train/ENST/*.pt
        mel_recordings/train/IDMT/*.pt
        mel_recordings/validation/ENST/*.pt
        mel_recordings/validation/IDMT/*.pt
    """

    def __init__(self, root, split):
        from pathlib import Path

        self.root = Path(root)
        self.split = split

        self.paths = sorted(
            (self.root / "mel_recordings" / split).glob("*/*.pt")
        )

        if not self.paths:
            raise FileNotFoundError(
                f"No linear Mel recordings found for {split} "
                f"in {self.root}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        saved = torch.load(
            self.paths[index],
            map_location="cpu",
            weights_only=True,
        )

        mel = saved["mel"]
        labels = saved["labels"]

        if mel.ndim != 3 or mel.shape[1:] != (84, 1):
            raise ValueError(
                f"Invalid Mel shape: {tuple(mel.shape)}"
            )

        if labels.shape != (mel.shape[0], 3):
            raise ValueError(
                f"Invalid label shape: {tuple(labels.shape)}"
            )

        if mel.shape[0] == 0 or mel.shape[0] % 400 != 0:
            raise ValueError(
                "Recording length must be a positive "
                "multiple of 400 frames."
            )

        return mel, labels

