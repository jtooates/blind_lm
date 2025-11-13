"""
Benchmark script to measure speedup of vectorized batch InfoNCE loss.
"""

import torch
import time
from phase1.batch_infonce_loss import BatchInfoNCELoss

def benchmark_loss(loss_fn, latents, num_iterations=10):
    """Benchmark loss computation time."""
    # Warmup
    for _ in range(3):
        _ = loss_fn(latents)

    # Actual timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    for _ in range(num_iterations):
        loss = loss_fn(latents)
        # Simulate backward pass
        loss.backward() if latents.requires_grad else None

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / num_iterations
    return avg_time


if __name__ == "__main__":
    print("=" * 60)
    print("Batch InfoNCE Loss Performance Benchmark")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test configurations matching training
    configs = [
        {"B": 4, "H": 32, "W": 32, "C": 3, "num_samples": 25, "num_cross_images": 8},
        {"B": 8, "H": 32, "W": 32, "C": 3, "num_samples": 25, "num_cross_images": 8},
        {"B": 16, "H": 32, "W": 32, "C": 3, "num_samples": 25, "num_cross_images": 8},
    ]

    for config in configs:
        B, H, W, C = config["B"], config["H"], config["W"], config["C"]
        num_samples = config["num_samples"]
        num_cross_images = config["num_cross_images"]

        print(f"\n{'─' * 60}")
        print(f"Config: B={B}, H={H}, W={W}, C={C}")
        print(f"        num_samples={num_samples}, num_cross_images={num_cross_images}")
        print(f"{'─' * 60}")

        # Create test data
        latents = torch.randn(B, H, W, C, device=device, requires_grad=True)

        # Initialize loss function
        loss_fn = BatchInfoNCELoss(
            within_weight=1.0,
            across_weight=1.0,
            patch_size=3,
            num_samples=num_samples,
            temperature_within=1.0,
            temperature_across=0.5,
            positive_radius=3.0,
            negative_radius=11.0,
            cross_image_radius=2.0,
            num_cross_images=num_cross_images
        ).to(device)

        # Benchmark
        avg_time = benchmark_loss(loss_fn, latents, num_iterations=10)

        print(f"Average time per forward+backward: {avg_time*1000:.2f} ms")
        print(f"Throughput: {1/avg_time:.2f} iterations/sec")

    print("\n" + "=" * 60)
    print("Key optimizations in vectorized version:")
    print("  1. Batch matrix multiplication for within-image similarities")
    print("  2. Einsum for cross-image similarities (all anchors at once)")
    print("  3. Precomputed masks for all anchors simultaneously")
    print("  4. Reduced from O(B×N×K) loops to O(B+N) loops")
    print("=" * 60)
