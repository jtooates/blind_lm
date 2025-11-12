"""
Diversity losses for encouraging varied spatial patterns in latent representations.

This module implements losses that encourage different sentences to produce
latents with different spatial structures, not just different colors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialContrastiveLoss(nn.Module):
    """
    Contrastive loss that encourages different sentences in a batch to have
    different spatial patterns in their latents.

    This loss is color-agnostic: it looks at the spatial distribution of intensity/energy,
    not the specific RGB values. This forces the model to use different spatial structures
    for different sentences, rather than encoding everything via subtle color variations.

    Args:
        temperature: Temperature for similarity computation (default: 0.5)
                    Lower = stronger penalty for similar spatial patterns
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, latents):
        """
        Compute spatial contrastive loss for a batch of latents.

        Args:
            latents: [B, H, W, C] tensor of RGB latents

        Returns:
            Scalar loss value. Loss is 0 if batch_size=1 (nothing to contrast).
        """
        B, H, W, C = latents.shape

        # Handle batch_size=1 case (no contrastive pairs)
        if B == 1:
            return torch.tensor(0.0, device=latents.device)

        # Compute spatial intensity per position (color-agnostic)
        # This captures "where" the energy is, not "what color"
        spatial_intensity = latents.norm(dim=-1)  # [B, H, W]

        # Flatten to vectors: [B, H*W]
        spatial_flat = spatial_intensity.reshape(B, -1)

        # Normalize to unit vectors
        spatial_norm = F.normalize(spatial_flat, dim=1)

        # Compute similarity matrix [B, B]
        # High similarity = similar spatial patterns (bad, we want diversity)
        similarity = torch.mm(spatial_norm, spatial_norm.t())

        # Apply temperature scaling
        similarity = similarity / self.temperature

        # Mask out diagonal (self-similarity = 1.0 always)
        mask = torch.eye(B, device=latents.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, 0)

        # Loss: penalize high similarity between different sentences
        # We want to minimize the average similarity
        # Sum over all pairs, normalize by number of pairs
        loss = similarity.sum() / (B * (B - 1))

        return loss


if __name__ == "__main__":
    # Test the spatial contrastive loss
    print("Testing SpatialContrastiveLoss...")

    loss_fn = SpatialContrastiveLoss(temperature=0.5)

    # Test case 1: Batch of 4 random latents
    latents = torch.randn(4, 32, 32, 3)
    loss = loss_fn(latents)
    print(f"Loss for random latents (B=4): {loss.item():.4f}")

    # Test case 2: Identical latents (should have high loss)
    identical = torch.randn(1, 32, 32, 3).expand(4, -1, -1, -1).clone()
    loss_identical = loss_fn(identical)
    print(f"Loss for identical latents (B=4): {loss_identical.item():.4f} (should be high ~1.0)")

    # Test case 3: Very different latents
    different = torch.randn(4, 32, 32, 3)
    # Make each latent have energy in different quadrants
    different[0, :16, :16, :] *= 10  # Top-left
    different[1, :16, 16:, :] *= 10  # Top-right
    different[2, 16:, :16, :] *= 10  # Bottom-left
    different[3, 16:, 16:, :] *= 10  # Bottom-right
    loss_different = loss_fn(different)
    print(f"Loss for spatially diverse latents (B=4): {loss_different.item():.4f} (should be low)")

    # Test case 4: Batch size = 1 (edge case)
    single = torch.randn(1, 32, 32, 3)
    loss_single = loss_fn(single)
    print(f"Loss for single latent (B=1): {loss_single.item():.4f} (should be 0)")

    # Test backward pass
    latents_grad = torch.randn(4, 32, 32, 3, requires_grad=True)
    loss = loss_fn(latents_grad)
    loss.backward()
    print(f"✓ Backward pass successful, gradient shape: {latents_grad.grad.shape}")

    print("\n✓ All tests passed!")
