"""
Phase 1: Visualization Tools
Monitoring tools for image prior metrics and latent quality.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, List, Dict
import json


def plot_power_spectra(latents: torch.Tensor, save_path: Optional[Path] = None):
    """
    Plot log-log power spectra for each channel with fitted slopes.

    Args:
        latents: [B, H, W, C] visual latents
        save_path: Optional path to save figure
    """
    from losses import SpectrumLoss

    spec_loss = SpectrumLoss()

    # Average over batch
    latents_avg = latents.mean(dim=0, keepdim=True)  # [1, H, W, C]

    # Compute spectrum
    freqs, power = spec_loss.compute_radial_spectrum(latents_avg)
    freqs = freqs.cpu().numpy()
    power = power.squeeze(0).cpu().numpy()  # [C, n_bins]

    # Fit slopes
    slopes = spec_loss.fit_power_law(
        torch.from_numpy(freqs),
        torch.from_numpy(power).unsqueeze(0)
    ).squeeze(0).numpy()

    C = power.shape[0]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for c in range(C):
        ax = axes[c]

        # Plot log-log
        valid = (freqs > 0) & (power[c] > 0)
        ax.loglog(freqs[valid], power[c][valid], 'b.-', alpha=0.6, label='Data')

        # Plot fitted line
        if slopes[c] > 0:
            fitted = power[c][valid][0] * (freqs[valid] / freqs[valid][0]) ** (-slopes[c])
            ax.loglog(freqs[valid], fitted, 'r--', alpha=0.8,
                     label=f'α = {slopes[c]:.2f}')

        # Target range
        ax.axhspan(1e-6, 1e6, alpha=0.1, color='green')

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_title(f'Channel {c+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Power Spectra (Target: α ∈ [1.5, 2.5])', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return slopes


def plot_channel_montage(latents: torch.Tensor, texts: List[str],
                         save_path: Optional[Path] = None):
    """
    Plot montage of all channels for fixed sentences.

    Args:
        latents: [B, H, W, C] visual latents
        texts: List of sentence strings
        save_path: Optional path to save figure
    """
    B, H, W, C = latents.shape
    latents_np = latents.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(B, C, figsize=(C*2, B*2))

    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        for c in range(C):
            ax = axes[b, c]

            # Show channel
            img = latents_np[b, :, :, c]
            im = ax.imshow(img, cmap='viridis', aspect='auto')
            ax.axis('off')

            # Title
            if b == 0:
                ax.set_title(f'Ch {c+1}', fontsize=10)

            # Text on left
            if c == 0:
                # Truncate text if too long
                text = texts[b][:30] + '...' if len(texts[b]) > 30 else texts[b]
                ax.text(-0.1, 0.5, text, transform=ax.transAxes,
                       rotation=90, va='center', ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gradient_histograms(latents: torch.Tensor, save_path: Optional[Path] = None):
    """
    Plot gradient histograms to visualize heavy-tailed distribution.

    Args:
        latents: [B, H, W, C] visual latents
        save_path: Optional path to save figure
    """
    # Convert to BCHW
    if latents.shape[-1] < latents.shape[-2]:
        latents = latents.permute(0, 3, 1, 2)

    # Compute gradients
    grad_h = latents[:, :, 1:, :] - latents[:, :, :-1, :]
    grad_w = latents[:, :, :, 1:] - latents[:, :, :, :-1]

    # Flatten
    grad_h_flat = grad_h.flatten().cpu().numpy()
    grad_w_flat = grad_w.flatten().cpu().numpy()

    # Compute kurtosis
    from scipy.stats import kurtosis
    kurt_h = kurtosis(grad_h_flat)
    kurt_w = kurtosis(grad_w_flat)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Horizontal gradients
    axes[0].hist(grad_h_flat, bins=100, alpha=0.7, color='blue', density=True)
    axes[0].set_xlabel('Gradient Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Horizontal Gradients (κ = {kurt_h:.2f})')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Vertical gradients
    axes[1].hist(grad_w_flat, bins=100, alpha=0.7, color='green', density=True)
    axes[1].set_xlabel('Gradient Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Vertical Gradients (κ = {kurt_w:.2f})')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Gradient Distributions (Target: κ > 3)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {'kurtosis_h': kurt_h, 'kurtosis_w': kurt_w}


def plot_channel_covariance(latents: torch.Tensor, save_path: Optional[Path] = None):
    """
    Plot channel covariance matrix.

    Args:
        latents: [B, H, W, C] visual latents
        save_path: Optional path to save figure
    """
    from losses import ChannelDecorrelationLoss

    decorr_loss = ChannelDecorrelationLoss()
    _, corr_matrix = decorr_loss(latents)
    corr_matrix_np = corr_matrix.cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # Labels
    C = corr_matrix_np.shape[0]
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels([f'Ch{i+1}' for i in range(C)])
    ax.set_yticklabels([f'Ch{i+1}' for i in range(C)])

    # Add values
    for i in range(C):
        for j in range(C):
            text = ax.text(j, i, f'{corr_matrix_np[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Channel Correlation Matrix (Target: Diagonal)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(metrics_history: List[Dict], save_path: Optional[Path] = None):
    """
    Plot training curves for all loss components.

    Args:
        metrics_history: List of metric dictionaries from training
        save_path: Optional path to save figure
    """
    if not metrics_history:
        print("No metrics history to plot")
        return

    # Extract data
    steps = [m['step'] for m in metrics_history]

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Total loss
    ax1 = fig.add_subplot(gs[0, :])
    losses = [m['eval_loss'] for m in metrics_history]
    ax1.plot(steps, losses, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Loss components
    components = ['spectrum', 'tv', 'wavelet', 'kurtosis', 'decorrelation', 'variance']
    component_axes = [
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])
    ]

    for comp, ax in zip(components, component_axes):
        if 'eval_components' in metrics_history[0]:
            values = [m['eval_components'][comp] for m in metrics_history]
            ax.plot(steps, values, linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(comp.capitalize(), fontsize=11)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_slope_evolution(metrics_history: List[Dict], save_path: Optional[Path] = None):
    """
    Plot evolution of power spectrum slopes over training.

    Args:
        metrics_history: List of metric dictionaries from training
        save_path: Optional path to save figure
    """
    if not metrics_history:
        print("No metrics history to plot")
        return

    steps = [m['step'] for m in metrics_history]

    # Extract slopes per channel
    slopes_per_channel = []
    for m in metrics_history:
        if 'eval_metrics' in m and 'slopes' in m['eval_metrics']:
            slopes = m['eval_metrics']['slopes']
            if torch.is_tensor(slopes):
                slopes = slopes.cpu().numpy()
            slopes_per_channel.append(slopes)

    if not slopes_per_channel:
        print("No slope data to plot")
        return

    slopes_per_channel = np.array(slopes_per_channel)  # [n_steps, n_channels]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual channel slopes
    ax1 = axes[0]
    n_channels = slopes_per_channel.shape[1]
    for c in range(n_channels):
        ax1.plot(steps, slopes_per_channel[:, c], label=f'Ch {c+1}', linewidth=2)

    # Target range
    ax1.axhspan(1.5, 2.5, alpha=0.2, color='green', label='Target')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Slope α')
    ax1.set_title('Power Spectrum Slopes per Channel', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mean slope
    ax2 = axes[1]
    mean_slopes = slopes_per_channel.mean(axis=1)
    std_slopes = slopes_per_channel.std(axis=1)

    ax2.plot(steps, mean_slopes, 'b-', linewidth=2, label='Mean')
    ax2.fill_between(steps,
                     mean_slopes - std_slopes,
                     mean_slopes + std_slopes,
                     alpha=0.3, label='±1 std')
    ax2.axhspan(1.5, 2.5, alpha=0.2, color='green', label='Target')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Slope α')
    ax2.set_title('Mean Slope with Std Dev', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_evaluation_report(model, eval_set, device='cpu', output_dir=None):
    """
    Create comprehensive evaluation report with all visualizations.

    Args:
        model: Trained model
        eval_set: FixedEvaluationSet
        device: Device to run on
        output_dir: Directory to save visualizations
    """
    model.eval()

    # Get batch
    batch = eval_set.get_batch(device=device)

    # Forward pass
    with torch.no_grad():
        latents = model(batch['input_ids'], batch['attention_mask'])

    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    print("Generating visualizations...")

    # 1. Power spectra
    slopes = plot_power_spectra(
        latents,
        save_path=output_dir / 'power_spectra.png' if output_dir else None
    )

    # 2. Channel montage
    plot_channel_montage(
        latents,
        batch['text'],
        save_path=output_dir / 'channel_montage.png' if output_dir else None
    )

    # 3. Gradient histograms
    kurt_metrics = plot_gradient_histograms(
        latents,
        save_path=output_dir / 'gradient_histograms.png' if output_dir else None
    )

    # 4. Channel covariance
    plot_channel_covariance(
        latents,
        save_path=output_dir / 'channel_covariance.png' if output_dir else None
    )

    # Summary
    summary = {
        'slopes': slopes.tolist() if hasattr(slopes, 'tolist') else slopes,
        'mean_slope': float(np.mean(slopes)),
        'slopes_in_range': int(np.sum((slopes >= 1.5) & (slopes <= 2.5))),
        'kurtosis_h': float(kurt_metrics['kurtosis_h']),
        'kurtosis_w': float(kurt_metrics['kurtosis_w'])
    }

    if output_dir:
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print("\nEvaluation Summary:")
    print(f"  Mean slope: {summary['mean_slope']:.2f}")
    print(f"  Slopes in target range [1.5, 2.5]: {summary['slopes_in_range']}/6")
    print(f"  Gradient kurtosis (h): {summary['kurtosis_h']:.2f}")
    print(f"  Gradient kurtosis (w): {summary['kurtosis_w']:.2f}")

    return summary


if __name__ == "__main__":
    # Test visualizations with random data
    print("Testing visualization tools with random data...")

    # Create random latents
    B, H, W, C = 16, 32, 32, 6
    latents = torch.randn(B, H, W, C)

    # Create dummy texts
    texts = [f"Sentence {i+1}" for i in range(B)]

    # Test plots
    print("\n1. Testing power spectra plot...")
    plot_power_spectra(latents)

    print("\n2. Testing channel montage...")
    plot_channel_montage(latents[:4], texts[:4])

    print("\n3. Testing gradient histograms...")
    plot_gradient_histograms(latents)

    print("\n4. Testing channel covariance...")
    plot_channel_covariance(latents)

    print("\nAll visualization tests passed!")
