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


def compute_infrequency_weights(dataloader: DataLoader) -> torch.Tensor:
    """ Compute the infrequency weight of an instrument, given as the 'inverse estimated entropy of their event activity distribution'. """
    num_classes = None
    probabilities = None

    num_timesteps = 0.0
    for _, labels in dataloader:
        # If probability tensor is not defined, count number of classes and do so
        if probabilities == None:
            num_classes = torch.tensor(labels.shape[-1])
            probabilities = torch.zeros(num_classes)
        
        num_timesteps += labels.shape[0] * labels.shape[1]
        probabilities += torch.sum(labels == 1.0, dim=(0, 1))
    
    # Divide to finish computing probabilities
    probabilities /= num_timesteps

    # To prevent Nan's, set probabilities of 0 to a very low number
    probabilities = torch.where(probabilities == 0, torch.tensor(1e-10), probabilities)

    # And compute final weights
    weights = 1.0 / (-probabilities * torch.log(probabilities) - (1.0 - probabilities) * torch.log(1.0 - probabilities))
    return weights


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
