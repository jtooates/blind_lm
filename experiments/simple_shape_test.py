"""
Simple test: Can we make Gaussian noise look like shapes using only losses?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def create_shape_losses(latent):
    """
    Create losses that encourage shape-like structures.

    Args:
        latent: [B, H, W] tensor to optimize

    Returns:
        dict of loss components
    """
    B, H, W = latent.shape
    losses = {}

    # 1. Sparsity - most pixels should be background (near 0)
    losses['sparsity'] = torch.mean(torch.abs(latent))

    # 2. Binary-ness - pixels should be either on or off
    # Encourage values near -1 or 1, not in between
    distance_from_binary = torch.min(
        (latent - 1.0)**2,  # Distance from 1
        (latent + 1.0)**2   # Distance from -1
    )
    losses['binary'] = torch.mean(distance_from_binary)

    # 3. Smoothness within objects (Total Variation)
    dx = torch.abs(latent[:, 1:, :] - latent[:, :-1, :])
    dy = torch.abs(latent[:, :, 1:] - latent[:, :, :-1])
    losses['tv'] = torch.mean(dx) + torch.mean(dy)

    # 4. Multiple objects - encourage exactly 2-3 bright regions
    # Count connected components (approximate)
    threshold = 0.5
    binary_mask = (latent > threshold).float()

    # Encourage some but not too many bright pixels
    bright_ratio = torch.mean(binary_mask)
    target_ratio = 0.2  # Want ~20% of pixels bright
    losses['object_size'] = (bright_ratio - target_ratio)**2

    # 5. Separation - bright regions should be separated
    # Use distance transform approximation
    kernel = torch.ones(1, 1, 5, 5).to(latent.device) / 25.0
    blurred = F.conv2d(
        binary_mask.unsqueeze(1),
        kernel,
        padding=2
    ).squeeze(1)

    # Penalize when bright regions are too connected
    losses['separation'] = torch.mean(blurred * binary_mask)

    return losses


def contrastive_shape_loss(latents):
    """
    Contrastive loss: different latents should look different.

    Args:
        latents: [B, H, W] batch of latents

    Returns:
        contrastive loss
    """
    B = latents.shape[0]
    if B < 2:
        return torch.tensor(0.0)

    # Flatten
    flat = latents.reshape(B, -1)

    # Normalize
    flat_norm = F.normalize(flat, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(flat_norm, flat_norm.T)

    # We want different latents to be different
    # So penalize high similarity for different indices
    mask = 1.0 - torch.eye(B).to(latents.device)

    # Loss: minimize similarity between different samples
    loss = torch.mean(torch.abs(sim_matrix) * mask)

    return loss


def optimize_noise_to_shapes(
    batch_size=4,
    image_size=(32, 32),
    num_steps=500,
    lr=0.01,
    device='cpu'
):
    """
    Main experiment: optimize Gaussian noise to look like shapes.
    """
    H, W = image_size

    # Initialize with structured noise (not pure random)
    latents = []
    for i in range(batch_size):
        # Start with weak noise
        noise = torch.randn(H, W) * 0.1

        # Add a few random blobs to break symmetry
        for _ in range(2):
            y = np.random.randint(5, H-5)
            x = np.random.randint(5, W-5)
            size = np.random.randint(3, 7)

            yy, xx = torch.meshgrid(
                torch.arange(H) - y,
                torch.arange(W) - x,
                indexing='ij'
            )
            blob = torch.exp(-(yy**2 + xx**2) / (2 * size**2))
            noise += blob * np.random.uniform(0.5, 1.5)

        latents.append(noise)

    latents = torch.stack(latents).to(device)
    latents.requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.Adam([latents], lr=lr)

    # Loss weights (tune these!)
    weights = {
        'sparsity': 0.5,
        'binary': 0.3,
        'tv': 0.1,
        'object_size': 1.0,
        'separation': 0.5,
        'contrastive': 2.0
    }

    losses_history = []

    print("Optimizing noise to look like shapes...")
    print(f"Loss weights: {weights}")

    for step in range(num_steps):
        optimizer.zero_grad()

        # Shape losses
        shape_losses = create_shape_losses(latents)

        # Contrastive loss
        contrast_loss = contrastive_shape_loss(latents)

        # Total loss
        total_loss = 0
        for name, loss in shape_losses.items():
            total_loss += loss * weights.get(name, 0.1)
        total_loss += contrast_loss * weights.get('contrastive', 1.0)

        # Optimize
        total_loss.backward()
        optimizer.step()

        # Clamp to valid range
        with torch.no_grad():
            latents.clamp_(-1.5, 1.5)

        losses_history.append(total_loss.item())

        if step % 50 == 0:
            print(f"Step {step:3d}: Loss = {total_loss.item():.4f}")
            # Print individual losses
            for name, loss in shape_losses.items():
                print(f"  {name}: {loss.item():.4f}")
            print(f"  contrastive: {contrast_loss.item():.4f}")

    return latents.detach(), losses_history


def visualize_results(optimized_latents, save_path='shape_results.png'):
    """
    Visualize the optimized latents.
    """
    B = optimized_latents.shape[0]

    fig, axes = plt.subplots(1, B, figsize=(B * 3, 3))

    if B == 1:
        axes = [axes]

    for i in range(B):
        ax = axes[i]

        # Show as grayscale image
        im = ax.imshow(
            optimized_latents[i].cpu().numpy(),
            cmap='gray',
            vmin=-1.5,
            vmax=1.5
        )
        ax.set_title(f'Latent {i+1}')
        ax.axis('off')

    plt.suptitle('Gaussian Noise → Shapes (via losses only)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()

    # Also save individual latents as separate images
    for i in range(B):
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.imshow(
            optimized_latents[i].cpu().numpy(),
            cmap='gray',
            vmin=-1.5,
            vmax=1.5
        )
        ax2.axis('off')
        plt.savefig(f'latent_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)


if __name__ == "__main__":
    print("="*60)
    print("EXPERIMENT: Can losses alone create shape-like patterns?")
    print("="*60)

    # Run optimization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    optimized, history = optimize_noise_to_shapes(
        batch_size=6,
        image_size=(32, 32),
        num_steps=300,
        lr=0.02,
        device=device
    )

    # Visualize
    print("\n" + "="*60)
    print("Optimization complete! Visualizing results...")
    print("="*60)

    visualize_results(optimized, 'shape_results.png')

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel('Step')
    plt.ylabel('Total Loss')
    plt.title('Loss During Optimization')
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nResults saved to:")
    print("  - shape_results.png (all latents)")
    print("  - latent_1.png, latent_2.png, ... (individual)")
    print("  - loss_curve.png (optimization history)")

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)