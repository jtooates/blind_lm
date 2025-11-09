"""
Object-forming losses for creating shape-like patterns in grayscale latents.
Based on successful experiments with Gaussian noise optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SparsityLoss(nn.Module):
    """
    Encourage sparse activations - most pixels should be background (near 0).
    This creates a dark background with bright objects.
    """

    def __init__(self, target_sparsity: float = 0.7):
        super().__init__()
        self.target_sparsity = target_sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] or [B, H, W, 1] grayscale latent
        Returns:
            sparsity loss
        """
        if x.dim() == 4:
            x = x.squeeze(-1)  # Remove channel dim if present

        # Simple L1 sparsity
        sparsity = torch.mean(torch.abs(x))

        return sparsity


class ObjectSizeLoss(nn.Module):
    """
    Control the ratio of foreground (objects) to background.
    Prevents everything becoming all dark or all bright.
    """

    def __init__(self, target_object_ratio: float = 0.25, threshold: float = 0.3):
        super().__init__()
        self.target_ratio = target_object_ratio
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] or [B, H, W, 1] grayscale latent
        Returns:
            object size loss
        """
        if x.dim() == 4:
            x = x.squeeze(-1)

        # Consider pixels above threshold as "object"
        object_mask = (x > self.threshold).float()
        object_ratio = object_mask.mean()

        # Penalize deviation from target ratio
        loss = (object_ratio - self.target_ratio) ** 2

        return loss


class BinaryLoss(nn.Module):
    """
    Encourage pixels to be either background (0) or foreground (1).
    Reduces gray values in between.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] or [B, H, W, 1] grayscale latent
        Returns:
            binary loss
        """
        if x.dim() == 4:
            x = x.squeeze(-1)

        # Distance from nearest binary value (0 or 1)
        # Assuming x is normalized to [0, 1] range
        distance_from_binary = torch.min(
            x ** 2,  # Distance from 0
            (x - 1.0) ** 2  # Distance from 1
        )

        return torch.mean(distance_from_binary)


class SpatialContrastiveLoss(nn.Module):
    """
    Encourage different inputs to produce different spatial patterns.
    This prevents all latents from collapsing to the same pattern.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] or [B, H, W, 1] batch of grayscale latents
        Returns:
            contrastive loss
        """
        if x.dim() == 4:
            x = x.squeeze(-1)

        B = x.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=x.device)

        # Flatten spatial dimensions
        flat = x.reshape(B, -1)

        # L2 normalize
        flat_norm = F.normalize(flat, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(flat_norm, flat_norm.T)

        # We want different samples to be different
        # So penalize high similarity (except diagonal)
        mask = 1.0 - torch.eye(B, device=x.device)

        # Mean absolute similarity (excluding self-similarity)
        loss = torch.sum(torch.abs(sim_matrix) * mask) / (B * (B - 1))

        return loss


class LocalCoherenceLoss(nn.Module):
    """
    Encourage local spatial coherence - nearby pixels should be similar within objects.
    This is similar to Total Variation but specifically tuned for object formation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] or [B, H, W, 1] grayscale latent
        Returns:
            local coherence loss
        """
        if x.dim() == 4:
            x = x.squeeze(-1)

        # Compute gradients
        dx = x[:, 1:, :] - x[:, :-1, :]  # Vertical gradient
        dy = x[:, :, 1:] - x[:, :, :-1]  # Horizontal gradient

        # L1 total variation
        tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

        return tv_loss


class ObjectFormingLoss(nn.Module):
    """
    Combined loss for creating object-like patterns in grayscale latents.
    Based on successful experiments with Gaussian noise optimization.
    """

    def __init__(
        self,
        lambda_sparsity: float = 0.5,
        lambda_object_size: float = 1.0,
        lambda_binary: float = 0.3,
        lambda_contrastive: float = 2.0,
        lambda_tv: float = 0.1,
        lambda_recon: float = 1.0,  # Keep reconstruction
        target_object_ratio: float = 0.25,
        sparsity_target: float = 0.7
    ):
        super().__init__()

        # Loss weights
        self.lambda_sparsity = lambda_sparsity
        self.lambda_object_size = lambda_object_size
        self.lambda_binary = lambda_binary
        self.lambda_contrastive = lambda_contrastive
        self.lambda_tv = lambda_tv
        self.lambda_recon = lambda_recon

        # Individual losses
        self.sparsity_loss = SparsityLoss(sparsity_target)
        self.object_size_loss = ObjectSizeLoss(target_object_ratio)
        self.binary_loss = BinaryLoss()
        self.contrastive_loss = SpatialContrastiveLoss()
        self.tv_loss = LocalCoherenceLoss()

        # Keep reconstruction loss from original
        from losses import ReconstructionLoss
        self.recon_loss = ReconstructionLoss()

    def forward(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> dict:
        """
        Args:
            x: [B, H, W, C] visual latent (C=1 for grayscale)
            logits: [B, L, V] predicted logits (optional, for reconstruction)
            target_ids: [B, L] target token IDs (optional, for reconstruction)
            return_components: whether to return individual loss components
        Returns:
            dict with 'loss' and optionally component losses
        """
        # Handle shape - expecting [B, H, W, 1] or [B, H, W]
        if x.dim() == 4 and x.shape[-1] == 1:
            x_gray = x.squeeze(-1)  # [B, H, W]
        elif x.dim() == 3:
            x_gray = x
        else:
            # If multiple channels, average them
            x_gray = x.mean(dim=-1)

        # Normalize to [0, 1] range for some losses
        x_norm = torch.sigmoid(x_gray)

        # Compute individual losses
        sparsity = self.sparsity_loss(x_norm)
        object_size = self.object_size_loss(x_norm)
        binary = self.binary_loss(x_norm)
        contrastive = self.contrastive_loss(x_gray)  # Use unnormalized
        tv = self.tv_loss(x_gray)  # Use unnormalized

        # Combined loss
        total_loss = (
            self.lambda_sparsity * sparsity +
            self.lambda_object_size * object_size +
            self.lambda_binary * binary +
            self.lambda_contrastive * contrastive +
            self.lambda_tv * tv
        )

        # Add reconstruction loss if provided
        recon_loss_value = torch.tensor(0.0, device=x.device)
        if logits is not None and target_ids is not None and self.lambda_recon > 0:
            recon_loss_value = self.recon_loss(logits, target_ids)
            total_loss = total_loss + self.lambda_recon * recon_loss_value

        result = {
            'loss': total_loss,
            'metrics': {}  # Could add metrics here
        }

        if return_components:
            result['components'] = {
                'sparsity': sparsity.item(),
                'object_size': object_size.item(),
                'binary': binary.item(),
                'contrastive': contrastive.item(),
                'tv': tv.item(),
                'reconstruction': recon_loss_value.item() if self.lambda_recon > 0 else 0.0
            }

        return result


if __name__ == "__main__":
    # Test the losses
    print("Testing object-forming losses...")

    # Create random input (grayscale)
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, H, W, 1) * 0.5 + 0.5  # Roughly [0, 1]

    # Test individual losses
    print("\n1. Sparsity Loss:")
    loss = SparsityLoss()(x)
    print(f"   Loss: {loss:.4f}")

    print("\n2. Object Size Loss:")
    loss = ObjectSizeLoss()(x)
    print(f"   Loss: {loss:.4f}")

    print("\n3. Binary Loss:")
    loss = BinaryLoss()(x)
    print(f"   Loss: {loss:.4f}")

    print("\n4. Contrastive Loss:")
    loss = SpatialContrastiveLoss()(x)
    print(f"   Loss: {loss:.4f}")

    print("\n5. TV Loss:")
    loss = LocalCoherenceLoss()(x)
    print(f"   Loss: {loss:.4f}")

    # Test combined loss
    print("\n6. Combined Object-Forming Loss:")
    criterion = ObjectFormingLoss()
    result = criterion(x, return_components=True)
    print(f"   Total loss: {result['loss']:.4f}")
    print("   Components:")
    for name, value in result['components'].items():
        print(f"      {name}: {value:.4f}")

    print("\n✓ All losses working!")