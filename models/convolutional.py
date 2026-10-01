import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameSynchronousCNNEncoder(nn.Module):
    def __init__(self, num_convolutions: int = 2):
        super().__init__()
        self.convolutions = nn.Sequential(
            *[ConvolutionalBlock(i + 1) for i in range(num_convolutions)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolutions(x)

class ConvolutionalBlock(nn.Module):
    def __init__(self, i: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(max(1, 32 * (i - 1)), 32 * i, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32 * i)
        self.conv2 = nn.Conv2d(32 * i, 32 * i, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32 * i)

        self.pool = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = self.batchnorm1(out)
        out = F.relu(self.conv2(out))
        out = self.batchnorm2(out)

        out = self.pool(out)
        out = self.dropout(out)
        
        return out