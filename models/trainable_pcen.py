
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainablePCEN(nn.Module):
    """
    Trainable Per-Channel Energy Normalization.

    Input:
        (batch, mel_bins, time)

    Output:
        (batch, mel_bins, time)

    The four PCEN parameters are shared across Mel channels.
    """

    def __init__(
        self,
        s=0.025,
        alpha=0.98,
        delta=2.0,
        r=0.5,
        eps=1e-6,
    ):
        super().__init__()

        # Unconstrained parameters transformed into valid ranges.
        self.raw_s = nn.Parameter(
            torch.logit(torch.tensor(float(s)))
        )

        self.raw_alpha = nn.Parameter(
            torch.logit(torch.tensor(float(alpha)))
        )

        self.raw_r = nn.Parameter(
            torch.logit(torch.tensor(float(r)))
        )

        # Softplus guarantees a positive delta.
        self.raw_delta = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(float(delta))))
        )

        self.eps = eps

    def get_parameters(self):
        """Return constrained PCEN parameter values."""
        return {
            "s": torch.sigmoid(self.raw_s),
            "alpha": torch.sigmoid(self.raw_alpha),
            "delta": F.softplus(self.raw_delta),
            "r": torch.sigmoid(self.raw_r),
        }

    def forward(self, mel):
        if mel.ndim != 3:
            raise ValueError(
                "Expected Mel shape (batch, mel_bins, time), "
                f"got {tuple(mel.shape)}"
            )

        if mel.shape[-1] == 0:
            raise ValueError("Mel input has zero time frames.")

        mel = torch.clamp(mel, min=0.0)

        params = self.get_parameters()

        s = params["s"]
        alpha = params["alpha"]
        delta = params["delta"]
        r = params["r"]

        # Differentiable temporal smoothing.
        smoothed_frames = [mel[:, :, 0]]

        for t in range(1, mel.shape[-1]):
            current = (
                s * mel[:, :, t]
                + (1.0 - s) * smoothed_frames[-1]
            )

            smoothed_frames.append(current)

        M = torch.stack(smoothed_frames, dim=-1)

        # Standard PCEN equation.
        pcen = (
            mel / (self.eps + M).pow(alpha) + delta
        ).pow(r) - delta.pow(r)

        return pcen
