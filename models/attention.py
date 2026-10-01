import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: Tuple[int, int] = (1, 21), embed_dim: int = 576):
        super().__init__()
        num_patches = 84 // patch_size[1]
        if 84 % patch_size[1] or embed_dim % num_patches:
            raise ValueError(f"Patch_size {patch_size}, dimension 1 has to be factor of {84}, and their division result a factor of embed_dim {embed_dim}")

        self.projection = nn.Conv2d(1, embed_dim // num_patches, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, embed_dim // num_patches, 1, num_patches))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projection(x) + self.position_embedding
        out = torch.flatten(out.permute(0, 2, 1, 3), start_dim=2)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class AttentionLayer(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int = 576):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p = 0.1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.norm1(x)
        out1, _ = self.attention(out1, out1, out1)
        out1 = self.dropout(out1) + x

        out2 = self.norm2(out1)
        out2 = F.gelu(self.fc1(out2))
        out2 = self.dropout(self.fc2(out2))
        out2 = out2 + out1
        return out2

class AttentionDecoder(nn.Module):
    def __init__(self, num_heads: int = 6, num_layers: int = 5, embed_dim: int = 576):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model=embed_dim)
        self.layers = nn.Sequential(*[AttentionLayer(num_heads=num_heads, embed_dim=embed_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.positional_encoding(x)
        out = self.layers(out.transpose(0, 1))

        return self.fc(out.transpose(0, 1))