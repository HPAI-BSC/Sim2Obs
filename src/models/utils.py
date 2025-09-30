import torch
import torch.nn as nn


class MaskedMAE(nn.Module):
    """
    Differentiable mean absolute error between x and y,
    ignoring positions where the difference is exactly zero.

    By measuring ICL diff only on non-zero regions we can compare between datasets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x - y)            # keeps gradients
        mask = (diff != 0.0).float()         # 1 where nonzero, 0 where zero
        masked_sum = (diff * mask).sum()   # only nonzero positions contribute
        count = mask.sum()
        if count == 0:                     # avoid division by zero
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return masked_sum / count