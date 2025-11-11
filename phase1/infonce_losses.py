"""
InfoNCE-based losses for RGB latent training.

This module implements:
- InfoNCE patch coherence loss (spatial RGB patch similarity)
- Magnitude loss (prevents collapse to zero)
- Combined wrapper for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        batch_loss = 0.0
        pad = self.patch_size // 2

        for b in range(B):
            # Extract single image
            img = latent_chw[b:b+1]  # [1, C, H, W]

            # Pad and extract patches
            img_padded = F.pad(img, (pad, pad, pad, pad), mode='replicate')
            patches = F.unfold(img_padded, kernel_size=self.patch_size, stride=1)
            # patches: [1, C*patch_size^2, H*W]
            patches = patches.squeeze(0).t()  # [H*W, C*patch_size^2]

            # Sample anchor positions
            num_positions = H * W
            num_samples = min(self.num_samples, num_positions)

            anchor_indices = torch.randperm(num_positions, device=latent.device)[:num_samples]
            anchor_patches = patches[anchor_indices]  # [num_samples, C*patch_size^2]

            # Compute anchor coordinates
            anchor_coords = torch.stack([
                anchor_indices // W,  # y coordinates
                anchor_indices % W    # x coordinates
            ], dim=1).float()  # [num_samples, 2]

            # All patch coordinates
            y_coords = torch.arange(H, device=latent.device).unsqueeze(1).expand(H, W).reshape(-1)
            x_coords = torch.arange(W, device=latent.device).unsqueeze(0).expand(H, W).reshape(-1)
            all_coords = torch.stack([y_coords, x_coords], dim=1).float()  # [H*W, 2]

            # Compute InfoNCE for each anchor
            for i in range(num_samples):
                anchor_patch = anchor_patches[i:i+1]  # [1, C*patch_size^2]
                anchor_coord = anchor_coords[i:i+1]   # [1, 2]

                # Compute spatial distances
                distances = torch.norm(all_coords - anchor_coord, dim=1)  # [H*W]

                # Define positive and negative masks
                pos_mask = (distances > 0) & (distances <= self.positive_radius)
                neg_mask = distances > self.negative_radius

                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    # Compute cosine similarities on full RGB patches
                    anchor_norm = F.normalize(anchor_patch, dim=1)
                    patches_norm = F.normalize(patches, dim=1)
                    similarities = torch.matmul(anchor_norm, patches_norm.t()).squeeze(0)  # [H*W]

                    # Apply temperature
                    pos_sims = similarities[pos_mask] / self.temperature
                    neg_sims = similarities[neg_mask] / self.temperature

                    # InfoNCE loss: -log(mean(exp(pos)) / (mean(exp(pos)) + mean(exp(neg))))
                    pos_exp = torch.exp(pos_sims)
                    neg_exp = torch.exp(neg_sims)

                    batch_loss += -torch.log(pos_exp.mean() / (pos_exp.mean() + neg_exp.mean() + 1e-8))

        # Average over batch and samples
        total_samples = B * self.num_samples
        return batch_loss / max(total_samples, 1)


class InfoNCELoss(nn.Module):
    """
    Combined loss for RGB latent training with InfoNCE coherence.

    Combines:
    - Reconstruction loss (text decoder cross-entropy)
    - InfoNCE patch coherence (spatial RGB patterns)
    - Magnitude loss (prevent collapse)

    Args:
        lambda_recon: Weight for reconstruction loss (default: 5.0)
        lambda_infonce: Weight for InfoNCE loss (default: 2.0)
        lambda_magnitude: Weight for magnitude loss (default: 5.0)
        patch_size: Patch size for InfoNCE (default: 3)
        num_samples: Number of anchor patches for InfoNCE (default: 100)
        temperature: Temperature for InfoNCE (default: 1.0)
        positive_radius: Positive pair radius for InfoNCE (default: 3.0)
        negative_radius: Negative pair radius for InfoNCE (default: 11.0)
        min_magnitude: Minimum magnitude target (default: 0.3)
    """
    def __init__(
        self,
        lambda_recon=5.0,
        lambda_infonce=2.0,
        lambda_magnitude=5.0,
        patch_size=3,
        num_samples=100,
        temperature=1.0,
        positive_radius=3.0,
        negative_radius=11.0,
        min_magnitude=0.3
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_infonce = lambda_infonce
        self.lambda_magnitude = lambda_magnitude

        # Initialize loss components
        self.infonce_loss = InfoNCEPatchLoss(
            patch_size=patch_size,
            num_samples=num_samples,
            temperature=temperature,
            positive_radius=positive_radius,
            negative_radius=negative_radius
        )
        self.magnitude_loss = MagnitudeLoss(min_magnitude=min_magnitude)
        self.recon_loss = nn.CrossEntropyLoss()

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
            If return_components=False:
                Scalar tensor (combined loss)
        """
        # 1. InfoNCE patch coherence
        infonce = self.infonce_loss(latent)

        # 2. Magnitude
        magnitude = self.magnitude_loss(latent)

        # 3. Reconstruction (if decoder outputs provided)
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
            self.lambda_magnitude * magnitude
        )

        if return_components:
            return {
                'loss': total_loss,
                'components': {
                    'recon_loss': recon.item(),
                    'infonce_loss': infonce.item(),
                    'magnitude_loss': magnitude.item()
                }
            }
        else:
            return total_loss
