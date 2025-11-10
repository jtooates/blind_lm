"""
Experimental testbed for finding the right loss cocktail to create image-like latents.
Tests contrastive learning on Gaussian noise to produce shapes with spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import matplotlib.patches as patches


class SpatialContrastiveLoss(nn.Module):
    """
    Contrastive loss that encourages:
    - Similar latents → similar spatial patterns
    - Different latents → different spatial patterns
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, H, W] or [B, H, W, C] batch of latents
        Returns:
            contrastive loss value
        """
        B = latents.shape[0]

        # Flatten to [B, -1]
        flat = latents.reshape(B, -1)

        # Compute pairwise cosine similarities
        flat_norm = F.normalize(flat, dim=1)
        sim_matrix = torch.matmul(flat_norm, flat_norm.T) / self.temperature

        # Create labels: similar indices should be similar
        # For now, use distance in batch as proxy
        labels = torch.arange(B)

        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class ShapeEmergenceLoss(nn.Module):
    """
    Encourage emergence of distinct shapes/objects in the latent.
    """

    def __init__(self):
        super().__init__()

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply losses that encourage shape-like structures.

        Args:
            latent: [B, H, W] grayscale image
        Returns:
            dict of loss components
        """
        losses = {}

        # 1. Sparsity loss - most pixels should be background (0)
        sparsity = torch.mean(torch.abs(latent))
        losses['sparsity'] = sparsity * 0.1

        # 2. Local coherence - nearby pixels should be similar (objects)
        dx = latent[:, 1:, :] - latent[:, :-1, :]
        dy = latent[:, :, 1:] - latent[:, :, :-1]
        tv = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        losses['tv'] = tv * 0.05

        # 3. Object-ness - encourage connected regions
        # Use threshold and connected components proxy
        binary = (torch.abs(latent) > 0.1).float()

        # Encourage binary values to form connected regions
        # by penalizing isolated pixels
        kernel = torch.ones(1, 1, 3, 3).to(latent.device) / 9.0
        if latent.dim() == 3:
            binary = binary.unsqueeze(1)
        smoothed = F.conv2d(binary, kernel, padding=1)
        if latent.dim() == 3:
            smoothed = smoothed.squeeze(1)

        # Pixels that are on but have few neighbors are penalized
        isolated = binary * (1 - smoothed)
        losses['isolation'] = torch.mean(isolated) * 0.1

        # 4. Distinctness - different regions should have different values
        # Encourage bimodal distribution (background + objects)
        values = latent.reshape(-1)
        mean_val = torch.mean(values)
        variance = torch.var(values)
        losses['variance'] = (1.0 - variance) * 0.05  # Encourage variance

        # 5. Edge sharpness - transitions should be sharp
        grad_x = torch.abs(latent[:, 1:, :] - latent[:, :-1, :])
        grad_y = torch.abs(latent[:, :, 1:] - latent[:, :, :-1])

        # Encourage gradients to be either 0 (smooth) or high (edge)
        edge_loss = -torch.mean(grad_x**2) - torch.mean(grad_y**2)
        losses['edges'] = edge_loss * 0.01

        return losses


class PositionalRelationLoss(nn.Module):
    """
    Encourage spatial relationships between bright regions (objects).
    """

    def __init__(self):
        super().__init__()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Encourage multiple distinct objects with spatial relationships.
        """
        B, H, W = latent.shape

        # Find centers of mass for bright regions
        # Threshold to get object regions
        objects = (latent > 0.2).float()

        # Compute center of mass for each image
        y_coords = torch.arange(H).view(1, H, 1).float().to(latent.device)
        x_coords = torch.arange(W).view(1, 1, W).float().to(latent.device)

        total_mass = objects.sum(dim=(1, 2), keepdim=True) + 1e-6

        y_center = (objects * y_coords).sum(dim=(1, 2)) / total_mass.squeeze()
        x_center = (objects * x_coords).sum(dim=(1, 2)) / total_mass.squeeze()

        # Encourage objects to be separated (not all in center)
        center_y = H / 2
        center_x = W / 2

        dist_from_center = torch.sqrt((y_center - center_y)**2 + (x_center - center_x)**2)

        # Loss encourages objects away from center
        loss = torch.mean(1.0 / (dist_from_center + 1.0))

        return loss


class LatentToImageExperiment:
    """
    Experiment with different loss combinations to create image-like patterns.
    """

    def __init__(self, latent_size=(32, 32), device='cpu'):
        self.H, self.W = latent_size
        self.device = device

        # Loss components
        self.shape_loss = ShapeEmergenceLoss()
        self.contrastive_loss = SpatialContrastiveLoss()
        self.position_loss = PositionalRelationLoss()

    def generate_latent_batch(self, batch_size: int, seed: int = None) -> torch.Tensor:
        """Generate batch of Gaussian noise latents."""
        if seed is not None:
            torch.manual_seed(seed)

        # Generate with some structure - not pure noise
        latents = []
        for i in range(batch_size):
            # Base noise
            noise = torch.randn(self.H, self.W) * 0.1

            # Add some blob-like structures
            n_blobs = np.random.randint(2, 5)
            for _ in range(n_blobs):
                # Random position
                cy = np.random.randint(5, self.H - 5)
                cx = np.random.randint(5, self.W - 5)

                # Random size
                size = np.random.randint(3, 8)

                # Create Gaussian blob
                y, x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
                blob = torch.exp(-((y - cy)**2 + (x - cx)**2) / (2 * size**2))

                # Random intensity
                intensity = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1])
                noise += blob * intensity

            latents.append(noise)

        return torch.stack(latents).to(self.device)

    def optimize_latents(
        self,
        initial_latents: torch.Tensor,
        num_steps: int = 100,
        lr: float = 0.01,
        loss_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Optimize latents using the loss cocktail.

        Args:
            initial_latents: [B, H, W] starting latents
            num_steps: optimization steps
            lr: learning rate
            loss_weights: dict of loss component weights

        Returns:
            optimized_latents, loss_history
        """
        if loss_weights is None:
            loss_weights = {
                'shape': 1.0,
                'contrastive': 0.5,
                'position': 0.1
            }

        latents = initial_latents.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latents], lr=lr)

        loss_history = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute losses
            total_loss = 0

            # Shape emergence losses
            shape_losses = self.shape_loss(latents)
            for name, loss in shape_losses.items():
                total_loss += loss * loss_weights.get('shape', 1.0)

            # Contrastive loss (if batch size > 1)
            if latents.shape[0] > 1:
                contrast_loss = self.contrastive_loss(latents)
                total_loss += contrast_loss * loss_weights.get('contrastive', 0.5)

            # Position loss
            position_loss = self.position_loss(latents)
            total_loss += position_loss * loss_weights.get('position', 0.1)

            # Backward
            total_loss.backward()
            optimizer.step()

            # Clamp values to reasonable range
            with torch.no_grad():
                latents.clamp_(-1, 1)

            loss_history.append(total_loss.item())

            if step % 20 == 0:
                print(f"Step {step}: Loss = {total_loss.item():.4f}")

        return latents.detach(), loss_history

    def visualize_results(
        self,
        original: torch.Tensor,
        optimized: torch.Tensor,
        save_path: str = None
    ):
        """Visualize original vs optimized latents."""
        B = original.shape[0]

        fig, axes = plt.subplots(2, B, figsize=(B * 3, 6))

        if B == 1:
            axes = axes.reshape(2, 1)

        for i in range(B):
            # Original
            ax = axes[0, i]
            im = ax.imshow(original[i].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'Original {i+1}')
            ax.axis('off')

            # Optimized
            ax = axes[1, i]
            im = ax.imshow(optimized[i].cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'Optimized {i+1}')
            ax.axis('off')

        plt.suptitle('Latent Optimization: Gaussian Noise → Spatial Patterns')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def run_experiment(
        self,
        batch_size: int = 4,
        num_steps: int = 200,
        loss_configs: List[Dict] = None
    ):
        """
        Run experiments with different loss configurations.
        """
        if loss_configs is None:
            loss_configs = [
                {'shape': 1.0, 'contrastive': 0.0, 'position': 0.0},  # Shape only
                {'shape': 1.0, 'contrastive': 0.5, 'position': 0.0},  # + Contrastive
                {'shape': 1.0, 'contrastive': 0.5, 'position': 0.2},  # + Position
                {'shape': 0.5, 'contrastive': 1.0, 'position': 0.2},  # Contrastive-heavy
            ]

        results = []

        for i, weights in enumerate(loss_configs):
            print(f"\n{'='*50}")
            print(f"Experiment {i+1}: Weights = {weights}")
            print('='*50)

            # Generate initial latents
            initial = self.generate_latent_batch(batch_size, seed=42 + i)

            # Optimize
            optimized, history = self.optimize_latents(
                initial,
                num_steps=num_steps,
                loss_weights=weights
            )

            # Visualize
            self.visualize_results(
                initial,
                optimized,
                save_path=f'experiment_{i+1}.png'
            )

            results.append({
                'weights': weights,
                'initial': initial,
                'optimized': optimized,
                'loss_history': history
            })

        return results


if __name__ == "__main__":
    # Run experiments
    experiment = LatentToImageExperiment(latent_size=(32, 32))

    print("Testing different loss cocktails for creating spatial patterns...")
    print("Goal: Transform Gaussian noise into shape-like patterns")

    # Test different loss combinations
    results = experiment.run_experiment(
        batch_size=4,
        num_steps=200,
        loss_configs=[
            {'shape': 1.0, 'contrastive': 0.0, 'position': 0.0},  # Baseline
            {'shape': 1.0, 'contrastive': 1.0, 'position': 0.0},  # Add contrastive
            {'shape': 0.5, 'contrastive': 2.0, 'position': 0.5},  # Strong contrastive
        ]
    )

    print("\n" + "="*50)
    print("Experiments complete! Check the generated images.")
    print("="*50)