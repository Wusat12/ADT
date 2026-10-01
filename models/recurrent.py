import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, input_size: int = 576, num_layers: int = 3, hidden_size: int = 288, use_gru: bool = True):
        super().__init__()
        if use_gru:
            self.rnns = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, dropout=0.1)
        else:
            self.rnns = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(2 * hidden_size, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnns(x.transpose(0, 1))
        return self.fc(out.transpose(0, 1))