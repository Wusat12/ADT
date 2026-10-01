""" Expose PyTorch models """

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import tune

from .convolutional import FrameSynchronousCNNEncoder
from .recurrent import RNNDecoder
from .attention import AttentionDecoder, PatchEmbedding

from typing import Tuple
from warnings import warn


class RNN(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_size: int = 288, use_gru: bool = True):
        super().__init__()
        self.recurrent = RNNDecoder(input_size=84, num_layers=num_layers, hidden_size=hidden_size, use_gru=use_gru)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = torch.flatten(x.permute(0, 2, 1, 3), start_dim=2)
        return self.recurrent(latent)
    
    name = "RNN"
    hyperparameters = {
        "num_layers": tune.choice([2, 3, 4, 5, 6]),
        "hidden_size": tune.choice([72, 144, 288, 576]),
        "use_gru": tune.choice([True, False])
    }

class CNN(nn.Module):
    def __init__(self, num_convs: int = 2, num_layers: int = 2, hidden_size: int = 288):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder(num_convolutions=num_convs)

        latent_size = 84 // (torch.pow(torch.tensor(3), num_convs)) * (32 * num_convs)
        self.dense = nn.Sequential(
            nn.Linear(latent_size, hidden_size), 
            nn.ReLU(), 
            *[layer for _ in range(num_layers - 1) for layer in [nn.Linear(hidden_size, hidden_size), nn.ReLU()]]
        )
        self.fc = nn.Linear(hidden_size, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.fc(self.dense(latent))
    
    name = "CNN"
    hyperparameters = {
        "num_convs": tune.choice([1, 2, 3]), 
        "num_layers": tune.choice([1, 2, 3, 4]),
        "hidden_size": tune.choice([72, 144, 288, 576])
    }

class ConvolutionalRNN(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_size: int = 288, use_gru: bool = True):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = RNNDecoder(num_layers=num_layers, hidden_size=hidden_size, use_gru=use_gru)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.decoder(latent)
    
    name = "Convolutional RNN"
    hyperparameters = {
        "num_layers": tune.choice([2, 3, 4, 5]),
        "hidden_size": tune.choice([72, 144, 288, 576]),
        "use_gru": tune.choice([True, False])
    }
    
    
class ConvolutionalTransformer(nn.Module):
    def __init__(self, num_heads: int = 6, num_layers: int = 5, embed_dim: int = 576):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.projection = nn.Linear(576, embed_dim)
        self.decoder = AttentionDecoder(num_heads=num_heads, num_layers=num_layers, embed_dim=embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        latent = self.projection(latent)
        return self.decoder(latent)

    name = "Convolutional Transformer"
    hyperparameters = {
        "num_heads": tune.choice([2, 4, 6, 8]),
        "num_layers": tune.choice([2, 4, 6, 8, 10]),
        "embed_dim": tune.choice([72, 144, 288, 576])
    }
    
class VisionTransformer(nn.Module):
    def __init__(self, patch_size: Tuple[int, int] = (1, 21), num_heads: int = 6, num_layers: int = 5, embed_dim: int = 576):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        self.decoder = AttentionDecoder(num_heads=num_heads, num_layers=num_layers, embed_dim=embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.patch_embedding(x)
        return self.decoder(latent)
    
    name = "Vision Transformer"
    hyperparameters = {
        "patch_size": tune.choice([(1, 7), (1, 14), (1, 21)]),
        "num_heads": tune.choice([2, 4, 6, 8]),
        "num_layers": tune.choice([2, 4, 6, 8, 10]),
        "embed_dim": tune.choice([72, 144, 288, 576])
    }


# --- Old model names ---
class ADTOF_FrameRNN(ConvolutionalRNN):
    def __init__(self, *args, **kwargs):
        warn("Class 'ADTOF_FrameRNN' is deprecated. Use 'ConvolutionalRNN' instead.", DeprecationWarning)
        super().__init__(*args, **kwargs)

class ADTOF_FrameAttention(ConvolutionalTransformer):
    def __init__(self, *args, **kwargs):
        warn("Class 'ADTOF_FrameAttention' is deprecated. Use 'ConvolutionalTransformer' instead.", DeprecationWarning)
        super().__init__(*args, **kwargs)
