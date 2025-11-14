"""
Batch InfoNCE Loss with Cross-Image Negatives

This module implements a cross-image diversity loss that encourages:
- Patches at corresponding spatial locations in different images should differ

This is complementary to regular InfoNCE which handles within-image coherence.
The separation of concerns allows independent control over:
- InfoNCE: local smoothness (nearby patches similar, distant patches different)
- Batch InfoNCE: spatial diversity (same locations across images look different)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchInfoNCELoss(nn.Module):
    """
    Batch InfoNCE loss for cross-image spatial diversity.

    For each anchor patch at position (y, x) in an image:
    - Sample K other images
    - Find patches at similar positions (y±radius, x±radius) in those images
    - Minimize similarity to those cross-image patches (encourage diversity)

    This is complementary to regular InfoNCE which handles within-image coherence.

    Args:
        patch_size: Size of patches (e.g., 3 for 3x3x3 RGB patches)
        num_samples: Number of anchor patches to sample per image
        temperature: Temperature for cross-image similarity scaling
        cross_image_radius: Spatial tolerance for cross-image negatives (pixels)
        num_cross_images: Number of other images to sample for cross-image negatives
    """

    def __init__(
        self,
        patch_size: int = 3,
        num_samples: int = 100,
        temperature: float = 0.5,
        cross_image_radius: float = 2.0,
        num_cross_images: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.temperature = temperature
        self.cross_image_radius = cross_image_radius
        self.num_cross_images = num_cross_images

    def forward(self, latents):
        """
        Compute cross-image diversity loss.

        Args:
            latents: [B, H, W, C] tensor of RGB latents

        Returns:
            Scalar loss value
        """
        B, H, W, C = latents.shape

        if B == 1:
            # Can't compute cross-image loss with single image
            return torch.tensor(0.0, device=latents.device)

        # Reshape to [B, C, H, W] for conv operations
        latents_bchw = latents.permute(0, 3, 1, 2)

        # Extract all patches using unfold
        pad = self.patch_size // 2
        latents_padded = F.pad(latents_bchw, (pad, pad, pad, pad), mode='replicate')

        # patches: [B, C*patch_size^2, H*W]
        patches = F.unfold(latents_padded, kernel_size=self.patch_size, stride=1)
        patches = patches.permute(0, 2, 1)  # [B, H*W, C*patch_size^2]

        # Normalize all patches
        patches_norm = F.normalize(patches, dim=2)  # [B, H*W, C*patch_size^2]

        # Precompute all patch coordinates
        num_positions = H * W
        y_coords = torch.arange(H, device=latents.device).unsqueeze(1).expand(H, W).reshape(-1)
        x_coords = torch.arange(W, device=latents.device).unsqueeze(0).expand(H, W).reshape(-1)
        all_coords = torch.stack([y_coords, x_coords], dim=1).float()  # [H*W, 2]

        # Sample anchor positions (same for all images to enable vectorization)
        n_samples = min(self.num_samples, num_positions)
        anchor_indices = torch.randperm(num_positions, device=latents.device)[:n_samples]

        # Get anchors for all images: [B, n_samples, C*patch_size^2]
        anchor_patches_norm = patches_norm[:, anchor_indices, :]
        anchor_coords = all_coords[anchor_indices]  # [n_samples, 2]

        # Compute pairwise distances for cross-image mask: [n_samples, H*W]
        distances = torch.norm(
            anchor_coords.unsqueeze(1) - all_coords.unsqueeze(0),
            dim=2
        )

        # Mask for cross-image negatives (patches at similar locations)
        cross_mask = distances <= self.cross_image_radius  # [n_samples, H*W]

        # Sample K other images for cross-image negatives
        K = min(self.num_cross_images, B - 1)

        total_loss = 0.0
        num_valid_anchors = 0

        # Process each image's anchors
        for b in range(B):
            # Sample K other images (different for each b to get variety)
            other_indices = [idx for idx in range(B) if idx != b]
            if len(other_indices) > K:
                other_indices = torch.tensor(other_indices, device=latents.device)
                sampled_other = other_indices[torch.randperm(len(other_indices))[:K]].tolist()
            else:
                sampled_other = other_indices

            # Get cross-image patches: [K, H*W, C*patch_size^2]
            cross_patches = patches_norm[sampled_other]

            # Compute cross-image similarities: [n_samples, K, H*W]
            cross_patches_T = cross_patches.transpose(1, 2)  # [K, C*patch_size^2, H*W]
            sims_across = torch.einsum('nc,kcp->nkp', anchor_patches_norm[b], cross_patches_T)

            # Process each anchor
            for i in range(n_samples):
                cross_m = cross_mask[i]

                # Get cross-image similarities at matching locations: [K, H*W]
                sims_cross = sims_across[i]  # [K, H*W]

                # Apply spatial mask and flatten: [K*num_cross_positions]
                cross_sims_masked = sims_cross[:, cross_m].reshape(-1) / self.temperature

                if cross_sims_masked.numel() == 0:
                    continue  # No cross-image patches found

                # Cross-image diversity loss: minimize similarity to cross-image patches
                # Use mean squared similarity as the loss (want similarities near 0)
                loss = cross_sims_masked.pow(2).mean()

                total_loss += loss
                num_valid_anchors += 1

        # Average over all valid anchors
        if num_valid_anchors == 0:
            return torch.tensor(0.0, device=latents.device)

        return total_loss / num_valid_anchors


if __name__ == "__main__":
    # Test the batch InfoNCE loss
    print("Testing BatchInfoNCELoss (cross-image diversity only)...")

    loss_fn = BatchInfoNCELoss(
        patch_size=3,
        num_samples=50,
        temperature=0.5,
        cross_image_radius=2.0,
        num_cross_images=3
    )

    # Test case 1: Random batch
    latents = torch.randn(4, 32, 32, 3)
    loss = loss_fn(latents)
    print(f"Loss for random latents (B=4): {loss.item():.4f}")

    # Test case 2: Identical spatial patterns (should have HIGH loss)
    # Same patterns at same locations = high similarity = high loss
    base_pattern = torch.randn(1, 32, 32, 3)
    identical = base_pattern.expand(4, -1, -1, -1).clone()
    # Add slight color variations
    identical[:, :, :, 0] += torch.randn(4, 32, 32) * 0.1
    loss_identical = loss_fn(identical)
    print(f"Loss for identical patterns (B=4): {loss_identical.item():.4f} (should be HIGH)")

    # Test case 3: Different spatial patterns (should have LOW loss)
    # Different patterns at same locations = low similarity = low loss
    different = torch.randn(4, 32, 32, 3)
    # Put energy in different quadrants
    different[0, :16, :16, :] *= 10
    different[1, :16, 16:, :] *= 10
    different[2, 16:, :16, :] *= 10
    different[3, 16:, 16:, :] *= 10
    loss_different = loss_fn(different)
    print(f"Loss for diverse patterns (B=4): {loss_different.item():.4f} (should be LOW)")

    # Verify expected relationship
    assert loss_identical > loss_different, "Identical patterns should have higher loss than diverse!"
    print("✓ Loss relationship verified: identical > diverse")

    # Test case 4: Single image (edge case)
    single = torch.randn(1, 32, 32, 3)
    loss_single = loss_fn(single)
    print(f"Loss for single image (B=1): {loss_single.item():.4f} (should be 0)")

    # Test backward pass
    latents_grad = torch.randn(4, 32, 32, 3, requires_grad=True)
    loss = loss_fn(latents_grad)
    loss.backward()
    print(f"✓ Backward pass successful, gradient shape: {latents_grad.grad.shape}")

    print("\n✓ All tests passed!")
    print("\nNote: This loss ONLY handles cross-image diversity.")
    print("Combine with regular InfoNCE for within-image coherence.")
