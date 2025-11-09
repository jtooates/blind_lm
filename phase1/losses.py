"""
Phase 1: Image Prior Losses
Analytic priors to make the latent grid look image-like.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SpectrumLoss(nn.Module):
    """
    1/f spectrum loss - encourages natural image power spectrum.
    Target slope α ∈ [1.5, 2.5] for natural images.
    """

    def __init__(self, target_slope: float = 2.0, slope_weight: float = 1.0):
        super().__init__()
        self.target_slope = target_slope
        self.slope_weight = slope_weight

    def compute_radial_spectrum(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute radially averaged power spectrum.

        Args:
            x: [B, H, W, C] or [B, C, H, W] visual latent
        Returns:
            freqs: frequency bins
            power: radially averaged power spectrum
        """
        # Handle both BHWC and BCHW formats
        if x.dim() == 4 and x.shape[-1] < x.shape[-2]:  # Likely BHWC
            x = x.permute(0, 3, 1, 2)  # Convert to BCHW

        B, C, H, W = x.shape

        # Compute 2D FFT
        fft = torch.fft.fft2(x, norm='ortho')
        power = torch.abs(fft) ** 2

        # Shift zero frequency to center
        power = torch.fft.fftshift(power, dim=(-2, -1))

        # Create frequency grid
        freq_y = torch.fft.fftfreq(H, d=1.0).to(x.device)
        freq_x = torch.fft.fftfreq(W, d=1.0).to(x.device)
        freq_y = torch.fft.fftshift(freq_y)
        freq_x = torch.fft.fftshift(freq_x)

        fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_radial = torch.sqrt(fy**2 + fx**2)

        # Radial binning
        n_bins = min(H, W) // 2
        bin_edges = torch.linspace(0, 0.5, n_bins + 1).to(x.device)
        radial_profile = []

        for i in range(n_bins):
            mask = (freq_radial >= bin_edges[i]) & (freq_radial < bin_edges[i + 1])
            if mask.sum() > 0:
                radial_profile.append(power[..., mask].mean(dim=-1))
            else:
                radial_profile.append(torch.zeros(B, C).to(x.device))

        radial_profile = torch.stack(radial_profile, dim=-1)  # [B, C, n_bins]
        freq_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return freq_centers[1:], radial_profile[..., 1:]  # Skip DC component

    def fit_power_law(self, freqs: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        """
        Fit power law to spectrum: power = A * freq^(-α)
        Returns slope α for each channel.
        """
        # Take log and fit line
        log_freqs = torch.log10(freqs + 1e-10)
        log_power = torch.log10(power + 1e-10)

        # Fit using least squares (for each batch and channel)
        # slope = - α (negative because power decreases with frequency)
        B, C, N = log_power.shape

        # Compute slope for each channel
        slopes = []
        for b in range(B):
            for c in range(C):
                valid = ~torch.isnan(log_power[b, c]) & ~torch.isinf(log_power[b, c])
                if valid.sum() > 1:
                    x = log_freqs[valid]
                    y = log_power[b, c, valid]

                    # Linear regression
                    x_mean = x.mean()
                    y_mean = y.mean()
                    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
                    slopes.append(-slope)  # Negative to get positive α
                else:
                    slopes.append(torch.tensor(0.0).to(log_power.device))

        slopes = torch.stack(slopes).reshape(B, C)
        return slopes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [B, H, W, C] visual latent
        Returns:
            loss: spectrum matching loss
            metrics: dict with slopes per channel
        """
        freqs, power = self.compute_radial_spectrum(x)
        slopes = self.fit_power_law(freqs, power)

        # Loss: penalize deviation from target slope
        slope_loss = (slopes - self.target_slope).abs().mean()

        # Also penalize very flat or very steep spectra
        too_flat = F.relu(1.5 - slopes)  # Penalize α < 1.5
        too_steep = F.relu(slopes - 2.5)  # Penalize α > 2.5
        range_loss = (too_flat + too_steep).mean()

        loss = self.slope_weight * slope_loss + range_loss

        metrics = {
            'slopes': slopes.detach(),
            'mean_slope': slopes.mean().item(),
            'slope_std': slopes.std().item()
        }

        return loss, metrics


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss - promotes smoothness with edges.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] visual latent
        Returns:
            tv_loss: total variation loss
        """
        # Handle both BHWC and BCHW formats
        if x.dim() == 4 and x.shape[-1] < x.shape[-2]:  # Likely BHWC
            x = x.permute(0, 3, 1, 2)  # Convert to BCHW

        # Compute gradients
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        # Sum of absolute gradients
        tv_h = diff_h.sum(dim=(2, 3))  # Sum over spatial dims
        tv_w = diff_w.sum(dim=(2, 3))

        tv = tv_h + tv_w  # [B, C]

        if self.reduction == 'mean':
            return tv.mean()
        elif self.reduction == 'sum':
            return tv.sum()
        else:
            return tv


class WaveletSparsityLoss(nn.Module):
    """
    Wavelet sparsity loss - promotes sparse edge-like features.
    Simple version using Haar wavelets.
    """

    def __init__(self, sparsity_weight: float = 0.1):
        super().__init__()
        self.sparsity_weight = sparsity_weight

        # Haar wavelet filters
        self.register_buffer('low_filter', torch.tensor([[1, 1], [1, 1]]) / 2.0)
        self.register_buffer('high_h_filter', torch.tensor([[-1, -1], [1, 1]]) / 2.0)
        self.register_buffer('high_v_filter', torch.tensor([[-1, 1], [-1, 1]]) / 2.0)
        self.register_buffer('high_d_filter', torch.tensor([[1, -1], [-1, 1]]) / 2.0)

    def wavelet_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Apply one level of Haar wavelet transform"""
        # Handle both BHWC and BCHW formats
        if x.dim() == 4 and x.shape[-1] < x.shape[-2]:  # Likely BHWC
            x = x.permute(0, 3, 1, 2)  # Convert to BCHW

        B, C, H, W = x.shape

        # Apply filters using convolution
        # Need to reshape filters for conv2d
        filters = torch.stack([
            self.low_filter,
            self.high_h_filter,
            self.high_v_filter,
            self.high_d_filter
        ]).unsqueeze(1)  # [4, 1, 2, 2]

        # Apply to each channel
        coeffs = []
        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [B, 1, H, W]
            coeff = F.conv2d(x_c, filters, stride=2, padding=0)  # [B, 4, H/2, W/2]
            coeffs.append(coeff)

        coeffs = torch.cat(coeffs, dim=1)  # [B, C*4, H/2, W/2]

        # Split into LL, LH, HL, HH
        coeffs = coeffs.reshape(B, C, 4, H//2, W//2)
        ll = coeffs[:, :, 0]  # Low-low (approximation)
        lh = coeffs[:, :, 1]  # Low-high (horizontal edges)
        hl = coeffs[:, :, 2]  # High-low (vertical edges)
        hh = coeffs[:, :, 3]  # High-high (diagonal edges)

        return ll, lh, hl, hh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] visual latent
        Returns:
            loss: wavelet sparsity loss
        """
        _, lh, hl, hh = self.wavelet_transform(x)

        # L1 sparsity on high-frequency coefficients
        sparsity = (
            torch.abs(lh).mean() +
            torch.abs(hl).mean() +
            torch.abs(hh).mean()
        ) / 3.0

        # We actually want some edges, so we penalize too little or too much
        target_sparsity = 0.3  # Target level of edge activity
        loss = self.sparsity_weight * torch.abs(sparsity - target_sparsity)

        return loss


class GradientKurtosisLoss(nn.Module):
    """
    Heavy-tailed gradient loss - natural images have kurtosis > 3.
    """

    def __init__(self, target_kurtosis: float = 6.0):
        super().__init__()
        self.target_kurtosis = target_kurtosis

    def compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis of a distribution"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + 1e-8)
        kurtosis = (normalized ** 4).mean(dim=-1)
        return kurtosis

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [B, H, W, C] visual latent
        Returns:
            loss: kurtosis matching loss
            metrics: dict with kurtosis values
        """
        # Handle both BHWC and BCHW formats
        if x.dim() == 4 and x.shape[-1] < x.shape[-2]:  # Likely BHWC
            x = x.permute(0, 3, 1, 2)  # Convert to BCHW

        # Compute gradients
        grad_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_w = x[:, :, :, 1:] - x[:, :, :, :-1]

        # Flatten spatial dimensions
        grad_h_flat = grad_h.reshape(x.shape[0], x.shape[1], -1)
        grad_w_flat = grad_w.reshape(x.shape[0], x.shape[1], -1)

        # Compute kurtosis for each channel
        kurt_h = self.compute_kurtosis(grad_h_flat)  # [B, C]
        kurt_w = self.compute_kurtosis(grad_w_flat)  # [B, C]

        avg_kurt = (kurt_h + kurt_w) / 2.0

        # Loss: encourage heavy tails (kurtosis > 3)
        loss = F.relu(3.0 - avg_kurt).mean()  # Penalize if kurtosis < 3

        # Also penalize if kurtosis is too extreme
        loss += 0.1 * F.relu(avg_kurt - 20.0).mean()  # Penalize if > 20

        metrics = {
            'kurtosis_h': kurt_h.mean().item(),
            'kurtosis_w': kurt_w.mean().item(),
            'kurtosis_avg': avg_kurt.mean().item()
        }

        return loss, metrics


class ChannelDecorrelationLoss(nn.Module):
    """
    Channel decorrelation loss - encourages channels to be independent.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, H, W, C] visual latent
        Returns:
            loss: decorrelation loss
            corr_matrix: correlation matrix for visualization
        """
        B, H, W, C = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(B, H * W, C)  # [B, H*W, C]

        # Compute correlation matrix for each batch
        corr_matrices = []
        for b in range(B):
            # Center the data
            x_b = x_flat[b]  # [H*W, C]
            x_centered = x_b - x_b.mean(dim=0, keepdim=True)

            # Compute correlation
            cov = torch.mm(x_centered.T, x_centered) / (H * W - 1)
            std = torch.sqrt(torch.diag(cov) + 1e-8)
            corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
            corr_matrices.append(corr)

        corr_matrix = torch.stack(corr_matrices).mean(dim=0)  # Average over batch

        # Loss: minimize off-diagonal correlations
        off_diagonal = corr_matrix - torch.eye(C).to(x.device)
        loss = (off_diagonal ** 2).mean()

        return loss, corr_matrix.detach()


class VarianceRegularizationLoss(nn.Module):
    """
    Variance regularization - maintains target variance level.
    """

    def __init__(self, target_var: float = 1.0):
        super().__init__()
        self.target_var = target_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] visual latent
        Returns:
            loss: variance regularization loss
        """
        B, H, W, C = x.shape

        # Compute variance per channel
        x_flat = x.reshape(B, H * W, C)
        var = x_flat.var(dim=1)  # [B, C]

        # Loss: match target variance
        loss = (var - self.target_var).abs().mean()

        return loss


class ReconstructionLoss(nn.Module):
    """
    Cross-entropy reconstruction loss for text decoder.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, L, V] predicted logits
            target_ids: [B, L] target token IDs
        Returns:
            loss: cross-entropy loss
        """
        # Reshape for cross-entropy: [B*L, V] and [B*L]
        B, L, V = logits.shape
        logits_flat = logits.reshape(B * L, V)
        target_flat = target_ids.reshape(B * L)

        # Cross-entropy loss
        loss = F.cross_entropy(logits_flat, target_flat, ignore_index=self.ignore_index)

        return loss


class ImagePriorLoss(nn.Module):
    """
    Combined image prior loss for Phase 1.
    Now includes optional reconstruction loss.
    """

    def __init__(
        self,
        lambda_spec: float = 0.5,
        lambda_tv: float = 0.1,
        lambda_wav: float = 0.1,
        lambda_kurt: float = 0.05,
        lambda_cov: float = 0.05,
        lambda_var: float = 0.05,
        lambda_recon: float = 0.0,  # NEW: reconstruction loss weight
        target_slope: float = 2.0,
        target_var: float = 1.0
    ):
        super().__init__()

        # Loss weights
        self.lambda_spec = lambda_spec
        self.lambda_tv = lambda_tv
        self.lambda_wav = lambda_wav
        self.lambda_kurt = lambda_kurt
        self.lambda_cov = lambda_cov
        self.lambda_var = lambda_var
        self.lambda_recon = lambda_recon

        # Individual losses
        self.spectrum_loss = SpectrumLoss(target_slope)
        self.tv_loss = TotalVariationLoss()
        self.wavelet_loss = WaveletSparsityLoss()
        self.kurtosis_loss = GradientKurtosisLoss()
        self.decorr_loss = ChannelDecorrelationLoss()
        self.var_loss = VarianceRegularizationLoss(target_var)
        self.recon_loss = ReconstructionLoss()  # NEW

    def forward(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> dict:
        """
        Args:
            x: [B, H, W, C] visual latent
            logits: [B, L, V] predicted logits (optional, for reconstruction)
            target_ids: [B, L] target token IDs (optional, for reconstruction)
            return_components: whether to return individual loss components
        Returns:
            dict with 'loss' and optionally component losses and metrics
        """
        # Compute image prior losses
        spec_loss, spec_metrics = self.spectrum_loss(x)
        tv_loss = self.tv_loss(x)
        wav_loss = self.wavelet_loss(x)
        kurt_loss, kurt_metrics = self.kurtosis_loss(x)
        cov_loss, corr_matrix = self.decorr_loss(x)
        var_loss = self.var_loss(x)

        # Combined loss
        total_loss = (
            self.lambda_spec * spec_loss +
            self.lambda_tv * tv_loss +
            self.lambda_wav * wav_loss +
            self.lambda_kurt * kurt_loss +
            self.lambda_cov * cov_loss +
            self.lambda_var * var_loss
        )

        # Add reconstruction loss if provided
        recon_loss_value = torch.tensor(0.0, device=x.device)
        if logits is not None and target_ids is not None and self.lambda_recon > 0:
            recon_loss_value = self.recon_loss(logits, target_ids)
            total_loss = total_loss + self.lambda_recon * recon_loss_value

        result = {
            'loss': total_loss,
            'metrics': {
                **spec_metrics,
                **kurt_metrics,
                'corr_matrix': corr_matrix
            }
        }

        if return_components:
            result['components'] = {
                'spectrum': spec_loss.item(),
                'tv': tv_loss.item(),
                'wavelet': wav_loss.item(),
                'kurtosis': kurt_loss.item(),
                'decorrelation': cov_loss.item(),
                'variance': var_loss.item(),
                'reconstruction': recon_loss_value.item() if self.lambda_recon > 0 else 0.0
            }

        return result


if __name__ == "__main__":
    # Test the losses
    print("Testing image prior losses...")

    # Create random input
    batch_size = 2
    H, W, C = 32, 32, 6
    x = torch.randn(batch_size, H, W, C)

    # Test individual losses
    print("\n1. Spectrum Loss:")
    spec_loss = SpectrumLoss()
    loss, metrics = spec_loss(x)
    print(f"   Loss: {loss:.4f}")
    print(f"   Mean slope: {metrics['mean_slope']:.2f}")

    print("\n2. Total Variation Loss:")
    tv_loss = TotalVariationLoss()
    loss = tv_loss(x)
    print(f"   Loss: {loss:.4f}")

    print("\n3. Wavelet Sparsity Loss:")
    wav_loss = WaveletSparsityLoss()
    loss = wav_loss(x)
    print(f"   Loss: {loss:.4f}")

    print("\n4. Gradient Kurtosis Loss:")
    kurt_loss = GradientKurtosisLoss()
    loss, metrics = kurt_loss(x)
    print(f"   Loss: {loss:.4f}")
    print(f"   Avg kurtosis: {metrics['kurtosis_avg']:.2f}")

    print("\n5. Channel Decorrelation Loss:")
    decorr_loss = ChannelDecorrelationLoss()
    loss, corr = decorr_loss(x)
    print(f"   Loss: {loss:.4f}")
    print(f"   Corr matrix shape: {corr.shape}")

    print("\n6. Variance Regularization Loss:")
    var_loss = VarianceRegularizationLoss()
    loss = var_loss(x)
    print(f"   Loss: {loss:.4f}")

    print("\n7. Combined Image Prior Loss:")
    combined_loss = ImagePriorLoss()
    result = combined_loss(x, return_components=True)
    print(f"   Total loss: {result['loss']:.4f}")
    print("   Components:")
    for name, value in result['components'].items():
        print(f"      {name}: {value:.4f}")