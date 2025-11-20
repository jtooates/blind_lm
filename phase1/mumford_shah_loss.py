"""
Mumford-Shah loss for piecewise constant regions.

Encourages the latent to be smooth within regions but allows
sharp boundaries between regions.
"""

import torch
import torch.nn as nn


class MumfordShahLoss(nn.Module):
    """
    Mumford-Shah loss for piecewise constant regions.

    Encourages the latent to be smooth within regions but allows
    sharp boundaries between regions.

    The key insight: L2 smoothness term heavily penalizes gradients
    (wants constant regions), but L1 boundary term allows sparse
    boundaries to exist (linear cost doesn't explode for edges).

    Args:
        alpha: Weight for L2 smoothness term (within-region smoothness)
        beta: Weight for L1 boundary term (allows sparse boundaries)
        epsilon: Small constant for numerical stability
    """
    def __init__(self, alpha=1.0, beta=0.1, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha  # Smoothness within regions (L2)
        self.beta = beta    # Boundary sparsity (L1)
        self.epsilon = epsilon  # Numerical stability

    def forward(self, latent):
        """
        Args:
            latent: [B, H, W, C] RGB/grayscale latent tensor

        Returns:
            Scalar loss value
        """
        # Compute gradients in horizontal and vertical directions
        diff_h = latent[:, 1:, :, :] - latent[:, :-1, :, :]  # [B, H-1, W, C]
        diff_w = latent[:, :, 1:, :] - latent[:, :, :-1, :]  # [B, H, W-1, C]

        # Smoothness term: L2 penalty on gradients
        # Heavily penalizes gradients - wants regions to be constant
        smoothness = (diff_h.pow(2).mean() + diff_w.pow(2).mean())

        # Boundary term: L1 penalty on gradients
        # Allows sparse boundaries to exist
        # Add epsilon for numerical stability of sqrt
        boundary = (
            (diff_h.pow(2).sum(dim=-1, keepdim=True) + self.epsilon).sqrt().mean() +
            (diff_w.pow(2).sum(dim=-1, keepdim=True) + self.epsilon).sqrt().mean()
        )

        return self.alpha * smoothness + self.beta * boundary
