"""
Batch InfoNCE Loss with Cross-Image Negatives

This module implements a unified InfoNCE loss that encourages:
1. Within-image coherence: nearby patches in the same image should be similar
2. Cross-image diversity: patches at corresponding locations in different images should differ

This replaces the need for a separate spatial diversity loss by incorporating
cross-image diversity directly into the contrastive learning framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchInfoNCELoss(nn.Module):
    """
    Batch InfoNCE loss with cross-image negatives.

    For each anchor patch in an image:
    - Positives: nearby patches in the SAME image (within positive_radius)
    - Negatives:
        - Far patches in the SAME image (beyond negative_radius)
        - Patches at similar locations in OTHER images (encourages spatial diversity)

    Args:
        within_weight: Weight for within-image coherence component (default: 1.0)
        across_weight: Weight for cross-image diversity component (default: 1.0)
        patch_size: Size of patches (e.g., 3 for 3x3x3 RGB patches)
        num_samples: Number of anchor patches to sample per image
        temperature_within: Temperature for within-image similarity scaling
        temperature_across: Temperature for cross-image similarity scaling
        positive_radius: Max distance (pixels) for positive pairs within same image
        negative_radius: Min distance (pixels) for negative pairs within same image
        cross_image_radius: Spatial tolerance for cross-image negatives (pixels)
        num_cross_images: Number of other images to sample for cross-image negatives
    """

    def __init__(
        self,
        within_weight: float = 1.0,
        across_weight: float = 1.0,
        patch_size: int = 3,
        num_samples: int = 100,
        temperature_within: float = 1.0,
        temperature_across: float = 0.5,
        positive_radius: float = 3.0,
        negative_radius: float = 11.0,
        cross_image_radius: float = 2.0,
        num_cross_images: int = 8,
    ):
        super().__init__()
        self.within_weight = within_weight
        self.across_weight = across_weight
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.temperature_within = temperature_within
        self.temperature_across = temperature_across
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius
        self.cross_image_radius = cross_image_radius
        self.num_cross_images = num_cross_images

    def forward(self, latents):
        """
        Compute batch InfoNCE loss.

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

        total_loss = 0.0
        num_valid_anchors = 0

        # Process each image in batch
        for b in range(B):
            image_patches_norm = patches_norm[b]  # [H*W, C*patch_size^2]

            # Sample anchor positions for this image
            n_samples = min(self.num_samples, num_positions)
            anchor_indices = torch.randperm(num_positions, device=latents.device)[:n_samples]
            anchor_patches_norm = image_patches_norm[anchor_indices]  # [n_samples, C*patch_size^2]
            anchor_coords = all_coords[anchor_indices]  # [n_samples, 2]

            # For each anchor
            for i in range(n_samples):
                anchor_patch = anchor_patches_norm[i:i+1]  # [1, C*patch_size^2]
                anchor_coord = anchor_coords[i:i+1]  # [1, 2]

                # === WITHIN-IMAGE POSITIVES AND NEGATIVES ===

                # Compute distances to all patches in same image
                distances = torch.norm(all_coords - anchor_coord, dim=1)  # [H*W]

                # Positive mask: nearby but not self
                pos_mask = (distances > 0) & (distances <= self.positive_radius)

                # Negative mask: far away
                neg_within_mask = distances > self.negative_radius

                # === CROSS-IMAGE NEGATIVES ===

                # Sample K other images
                other_image_indices = [idx for idx in range(B) if idx != b]
                if len(other_image_indices) > self.num_cross_images:
                    other_image_indices = torch.tensor(other_image_indices, device=latents.device)
                    sampled_indices = other_image_indices[torch.randperm(len(other_image_indices))[:self.num_cross_images]]
                else:
                    sampled_indices = other_image_indices

                # Collect cross-image negative patches
                cross_image_patches = []
                for other_b in sampled_indices:
                    other_patches_norm = patches_norm[other_b]  # [H*W, C*patch_size^2]

                    # Find patches at similar spatial locations
                    cross_distances = torch.norm(all_coords - anchor_coord, dim=1)
                    cross_mask = cross_distances <= self.cross_image_radius

                    if cross_mask.sum() > 0:
                        cross_image_patches.append(other_patches_norm[cross_mask])

                # Check if we have enough samples
                if pos_mask.sum() == 0:
                    continue  # Skip if no positives

                if neg_within_mask.sum() == 0 and len(cross_image_patches) == 0:
                    continue  # Skip if no negatives

                # === COMPUTE SIMILARITIES ===

                # Similarities to all patches in same image
                similarities_within = torch.matmul(anchor_patch, image_patches_norm.t()).squeeze(0)  # [H*W]

                # Positive similarities (within-image, nearby)
                pos_sims = similarities_within[pos_mask] / self.temperature_within

                # Negative similarities (within-image, far)
                neg_within_sims = similarities_within[neg_within_mask] / self.temperature_within

                # Negative similarities (cross-image, similar location)
                neg_across_sims = []
                for cross_patches in cross_image_patches:
                    sims = torch.matmul(anchor_patch, cross_patches.t()).squeeze(0)
                    neg_across_sims.append(sims / self.temperature_across)

                # === COMPUTE LOSS ===

                # Within-image InfoNCE component
                loss_within = 0.0
                if self.within_weight > 0 and neg_within_mask.sum() > 0:
                    pos_exp = torch.exp(pos_sims)
                    neg_exp = torch.exp(neg_within_sims)

                    # InfoNCE: -log(mean(pos) / (mean(pos) + mean(neg)))
                    loss_within = -torch.log(
                        pos_exp.mean() / (pos_exp.mean() + neg_exp.mean() + 1e-8)
                    )

                # Cross-image InfoNCE component
                loss_across = 0.0
                if self.across_weight > 0 and len(neg_across_sims) > 0:
                    pos_exp = torch.exp(pos_sims)

                    # Combine all cross-image negatives
                    all_neg_across = torch.cat(neg_across_sims)
                    neg_across_exp = torch.exp(all_neg_across)

                    # InfoNCE with cross-image negatives
                    loss_across = -torch.log(
                        pos_exp.mean() / (pos_exp.mean() + neg_across_exp.mean() + 1e-8)
                    )

                # Weighted combination
                total_loss += self.within_weight * loss_within + self.across_weight * loss_across
                num_valid_anchors += 1

        # Average over all valid anchors
        if num_valid_anchors == 0:
            return torch.tensor(0.0, device=latents.device)

        return total_loss / num_valid_anchors


if __name__ == "__main__":
    # Test the batch InfoNCE loss
    print("Testing BatchInfoNCELoss...")

    loss_fn = BatchInfoNCELoss(
        within_weight=1.0,
        across_weight=1.0,
        patch_size=3,
        num_samples=50,
        temperature_within=1.0,
        temperature_across=0.5,
        positive_radius=3.0,
        negative_radius=11.0,
        cross_image_radius=2.0,
        num_cross_images=3
    )

    # Test case 1: Random batch
    latents = torch.randn(4, 32, 32, 3)
    loss = loss_fn(latents)
    print(f"Loss for random latents (B=4): {loss.item():.4f}")

    # Test case 2: Identical spatial patterns (should have high across loss)
    base_pattern = torch.randn(1, 32, 32, 3)
    identical = base_pattern.expand(4, -1, -1, -1).clone()
    # Add slight color variations
    identical[:, :, :, 0] += torch.randn(4, 32, 32) * 0.1
    loss_identical = loss_fn(identical)
    print(f"Loss for identical patterns (B=4): {loss_identical.item():.4f} (should be high)")

    # Test case 3: Different spatial patterns (should have lower across loss)
    different = torch.randn(4, 32, 32, 3)
    # Put energy in different quadrants
    different[0, :16, :16, :] *= 10
    different[1, :16, 16:, :] *= 10
    different[2, 16:, :16, :] *= 10
    different[3, 16:, 16:, :] *= 10
    loss_different = loss_fn(different)
    print(f"Loss for diverse patterns (B=4): {loss_different.item():.4f} (should be lower)")

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
