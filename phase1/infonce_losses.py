"""
InfoNCE-based losses for RGB latent training.

This module implements:
- InfoNCE patch coherence loss (spatial RGB patch similarity)
- Magnitude loss (prevents collapse to zero)
- Spatial diversity loss (encourages different spatial patterns)
- Combined wrapper for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diversity_losses import SpatialContrastiveLoss
from batch_infonce_loss import BatchInfoNCELoss


class MagnitudeLoss(nn.Module):
    """
    Prevents latent collapse to near-zero values.

    Encourages the latent to maintain a minimum magnitude,
    ensuring meaningful signal strength.

    Args:
        min_magnitude: Minimum target for mean absolute value (default: 0.3)
    """
    def __init__(self, min_magnitude=0.3):
        super().__init__()
        self.min_magnitude = min_magnitude

    def forward(self, latent):
        """
        Args:
            latent: [B, H, W, C] tensor

        Returns:
            Scalar loss value
        """
        magnitude = torch.mean(torch.abs(latent))
        loss = torch.relu(self.min_magnitude - magnitude)
        return loss


class InfoNCEPatchLoss(nn.Module):
    """
    InfoNCE contrastive loss for spatial patch coherence.

    Encourages nearby patches to be similar (positive pairs)
    and distant patches to be different (negative pairs).

    For RGB latents, patches are extracted across all 3 channels
    and compared as complete RGB patterns.

    Args:
        patch_size: Size of square patches (default: 3)
        num_samples: Number of anchor patches per image (default: 100)
        temperature: Temperature for similarity computation (default: 1.0)
        positive_radius: Max spatial distance for positive pairs (default: 3.0)
        negative_radius: Min spatial distance for negative pairs (default: 11.0)
    """
    def __init__(
        self,
        patch_size=3,
        num_samples=100,
        temperature=1.0,
        positive_radius=3.0,
        negative_radius=11.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.temperature = temperature
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius

    def forward(self, latent):
        """
        Args:
            latent: [B, H, W, C] tensor (typically C=3 for RGB)

        Returns:
            Scalar loss value
        """
        B, H, W, C = latent.shape

        # Convert from [B, H, W, C] to [B, C, H, W] for F.unfold
        latent_chw = latent.permute(0, 3, 1, 2)  # [B, C, H, W]

        pad = self.patch_size // 2

        # Pad and extract patches for entire batch
        latent_padded = F.pad(latent_chw, (pad, pad, pad, pad), mode='replicate')
        patches = F.unfold(latent_padded, kernel_size=self.patch_size, stride=1)
        # patches: [B, C*patch_size^2, H*W]
        patches = patches.transpose(1, 2)  # [B, H*W, C*patch_size^2]

        # Pre-compute all patch coordinates (shared across batch)
        num_positions = H * W
        y_coords = torch.arange(H, device=latent.device).unsqueeze(1).expand(H, W).reshape(-1)
        x_coords = torch.arange(W, device=latent.device).unsqueeze(0).expand(H, W).reshape(-1)
        all_coords = torch.stack([y_coords, x_coords], dim=1).float()  # [H*W, 2]

        # Sample anchor positions (same for all images in batch for simplicity)
        num_samples = min(self.num_samples, num_positions)
        anchor_indices = torch.randperm(num_positions, device=latent.device)[:num_samples]

        # Get anchor patches for entire batch: [B, num_samples, C*patch_size^2]
        anchor_patches = patches[:, anchor_indices, :]

        # Compute anchor coordinates
        anchor_coords = torch.stack([
            anchor_indices // W,  # y coordinates
            anchor_indices % W    # x coordinates
        ], dim=1).float()  # [num_samples, 2]

        # Vectorized distance computation: [num_samples, H*W]
        # all_coords: [H*W, 2], anchor_coords: [num_samples, 2]
        distances = torch.cdist(anchor_coords.unsqueeze(0), all_coords.unsqueeze(0)).squeeze(0)
        # distances: [num_samples, H*W]

        # Define positive and negative masks for all anchors at once
        pos_mask = (distances > 0) & (distances <= self.positive_radius)  # [num_samples, H*W]
        neg_mask = distances > self.negative_radius  # [num_samples, H*W]

        # Normalize patches once for entire batch
        anchor_patches_norm = F.normalize(anchor_patches, dim=2)  # [B, num_samples, C*patch_size^2]
        patches_norm = F.normalize(patches, dim=2)  # [B, H*W, C*patch_size^2]

        # Compute similarities for all anchors at once: [B, num_samples, H*W]
        similarities = torch.bmm(anchor_patches_norm, patches_norm.transpose(1, 2))

        # Apply temperature
        similarities = similarities / self.temperature

        # Compute InfoNCE loss for each anchor across the batch
        batch_loss = 0.0
        valid_samples = 0

        for i in range(num_samples):
            pos_m = pos_mask[i]  # [H*W]
            neg_m = neg_mask[i]  # [H*W]

            if pos_m.sum() > 0 and neg_m.sum() > 0:
                # Get similarities for this anchor across batch: [B, H*W]
                sim = similarities[:, i, :]

                # Extract positive and negative similarities
                pos_sims = sim[:, pos_m]  # [B, num_pos]
                neg_sims = sim[:, neg_m]  # [B, num_neg]

                # InfoNCE loss per batch item
                pos_exp = torch.exp(pos_sims)
                neg_exp = torch.exp(neg_sims)

                # Average over positives and negatives, then compute log ratio
                loss_per_batch = -torch.log(
                    pos_exp.mean(dim=1) / (pos_exp.mean(dim=1) + neg_exp.mean(dim=1) + 1e-8)
                )

                batch_loss += loss_per_batch.sum()
                valid_samples += B

        # Average over valid samples
        return batch_loss / max(valid_samples, 1)


class InfoNCELoss(nn.Module):
    """
    Combined loss for RGB latent training with InfoNCE coherence.

    Combines:
    - Reconstruction loss (text decoder cross-entropy)
    - InfoNCE patch coherence (spatial RGB patterns)
    - Magnitude loss (prevent collapse)
    - Spatial diversity loss (encourages different spatial patterns across batch)

    Args:
        lambda_recon: Weight for reconstruction loss (default: 5.0)
        lambda_infonce: Weight for InfoNCE loss (default: 2.0)
        lambda_magnitude: Weight for magnitude loss (default: 5.0)
        lambda_spatial_diversity: Weight for spatial diversity loss (default: 0.0)
        lambda_batch_infonce: Weight for batch InfoNCE loss (default: 0.0)
        patch_size: Patch size for InfoNCE (default: 3)
        num_samples: Number of anchor patches for InfoNCE (default: 100)
        temperature: Temperature for InfoNCE (default: 1.0)
        positive_radius: Positive pair radius for InfoNCE (default: 3.0)
        negative_radius: Negative pair radius for InfoNCE (default: 11.0)
        min_magnitude: Minimum magnitude target (default: 0.3)
        spatial_diversity_temperature: Temperature for spatial diversity (default: 0.5)
        batch_infonce_within_weight: Weight for within-image component of batch InfoNCE (default: 1.0)
        batch_infonce_across_weight: Weight for cross-image component of batch InfoNCE (default: 1.0)
        batch_infonce_temperature_within: Temperature for within-image similarity (default: 1.0)
        batch_infonce_temperature_across: Temperature for cross-image similarity (default: 0.5)
        batch_infonce_cross_image_radius: Spatial tolerance for cross-image negatives (default: 2.0)
        batch_infonce_num_cross_images: Number of other images to sample (default: 8)
        pad_token_id: Token ID to ignore in reconstruction loss (default: 50256 for GPT-2 EOS)
    """
    def __init__(
        self,
        lambda_recon=5.0,
        lambda_infonce=2.0,
        lambda_magnitude=5.0,
        lambda_spatial_diversity=0.0,
        lambda_batch_infonce=0.0,
        patch_size=3,
        num_samples=100,
        temperature=1.0,
        positive_radius=3.0,
        negative_radius=11.0,
        min_magnitude=0.3,
        spatial_diversity_temperature=0.5,
        batch_infonce_within_weight=1.0,
        batch_infonce_across_weight=1.0,
        batch_infonce_temperature_within=1.0,
        batch_infonce_temperature_across=0.5,
        batch_infonce_cross_image_radius=2.0,
        batch_infonce_num_cross_images=8,
        pad_token_id=50256  # GPT-2 EOS token (used as padding)
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_infonce = lambda_infonce
        self.lambda_magnitude = lambda_magnitude
        self.lambda_spatial_diversity = lambda_spatial_diversity
        self.lambda_batch_infonce = lambda_batch_infonce

        # Initialize loss components
        self.infonce_loss = InfoNCEPatchLoss(
            patch_size=patch_size,
            num_samples=num_samples,
            temperature=temperature,
            positive_radius=positive_radius,
            negative_radius=negative_radius
        )
        self.magnitude_loss = MagnitudeLoss(min_magnitude=min_magnitude)

        # Spatial diversity loss (only if enabled)
        if self.lambda_spatial_diversity > 0:
            self.spatial_diversity_loss = SpatialContrastiveLoss(
                temperature=spatial_diversity_temperature
            )
        else:
            self.spatial_diversity_loss = None

        # Batch InfoNCE loss (only if enabled)
        if self.lambda_batch_infonce > 0:
            self.batch_infonce_loss = BatchInfoNCELoss(
                within_weight=batch_infonce_within_weight,
                across_weight=batch_infonce_across_weight,
                patch_size=patch_size,
                num_samples=num_samples,
                temperature_within=batch_infonce_temperature_within,
                temperature_across=batch_infonce_temperature_across,
                positive_radius=positive_radius,
                negative_radius=negative_radius,
                cross_image_radius=batch_infonce_cross_image_radius,
                num_cross_images=batch_infonce_num_cross_images
            )
        else:
            self.batch_infonce_loss = None

        # Ignore padding tokens in reconstruction loss
        self.recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, latent, logits=None, target_ids=None, return_components=False):
        """
        Compute combined loss.

        Args:
            latent: [B, H, W, C] RGB latent tensor
            logits: [B, seq_len, vocab_size] decoder output (optional)
            target_ids: [B, seq_len] target token IDs (optional)
            return_components: If True, return dict with all components. If False, return just loss scalar.

        Returns:
            If return_components=True:
                Dictionary containing:
                    - loss: Combined weighted loss (main loss key for train.py)
                    - recon_loss: Reconstruction loss component
                    - infonce_loss: InfoNCE loss component
                    - magnitude_loss: Magnitude loss component
                    - spatial_diversity_loss: Spatial diversity loss component (if enabled)
                    - batch_infonce_loss: Batch InfoNCE loss component (if enabled)
            If return_components=False:
                Scalar tensor (combined loss)
        """
        # 1. InfoNCE patch coherence
        infonce = self.infonce_loss(latent)

        # 2. Magnitude
        magnitude = self.magnitude_loss(latent)

        # 3. Spatial diversity (if enabled)
        if self.spatial_diversity_loss is not None:
            spatial_diversity = self.spatial_diversity_loss(latent)
        else:
            spatial_diversity = torch.tensor(0.0, device=latent.device)

        # 4. Batch InfoNCE (if enabled)
        if self.batch_infonce_loss is not None:
            batch_infonce = self.batch_infonce_loss(latent)
        else:
            batch_infonce = torch.tensor(0.0, device=latent.device)

        # 5. Reconstruction (if decoder outputs provided)
        if logits is not None and target_ids is not None:
            B, seq_len, vocab_size = logits.shape
            recon = self.recon_loss(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )
        else:
            recon = torch.tensor(0.0, device=latent.device)

        # Combine losses
        total_loss = (
            self.lambda_recon * recon +
            self.lambda_infonce * infonce +
            self.lambda_magnitude * magnitude +
            self.lambda_spatial_diversity * spatial_diversity +
            self.lambda_batch_infonce * batch_infonce
        )

        if return_components:
            components_dict = {
                'recon_loss': recon.item(),
                'infonce_loss': infonce.item(),
                'magnitude_loss': magnitude.item(),
            }
            # Only include optional losses if they're enabled
            if self.spatial_diversity_loss is not None:
                components_dict['spatial_diversity_loss'] = spatial_diversity.item()
            if self.batch_infonce_loss is not None:
                components_dict['batch_infonce_loss'] = batch_infonce.item()

            return {
                'loss': total_loss,
                'components': components_dict,
                'metrics': {}  # Placeholder for additional metrics if needed
            }
        else:
            return total_loss
