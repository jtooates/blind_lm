"""Quick test that coherence loss functions work correctly"""
import torch
import torch.nn.functional as F

def coherence_loss_autocorr(latent):
    """Spatial Autocorrelation Loss"""
    dx = latent[:, 1:, :] * latent[:, :-1, :]
    dy = latent[:, :, 1:] * latent[:, :, :-1]
    autocorr = torch.mean(dx) + torch.mean(dy)
    return -autocorr

def coherence_loss_perimeter(latent, threshold=0.5, temperature=0.1):
    """Perimeter-to-Area Loss"""
    binary_soft = torch.sigmoid((latent - threshold) / temperature)
    dx = torch.abs(binary_soft[:, 1:, :] - binary_soft[:, :-1, :])
    dy = torch.abs(binary_soft[:, :, 1:] - binary_soft[:, :, :-1])
    edge_length = torch.sum(dx) + torch.sum(dy)
    area = torch.sum(binary_soft) + 1e-6
    return edge_length / area

def coherence_loss_morphological(latent):
    """Morphological Smoothness Loss"""
    latent_4d = latent.unsqueeze(1)
    pooled = F.max_pool2d(latent_4d, kernel_size=3, stride=1, padding=1)
    unpooled = -F.max_pool2d(-pooled, kernel_size=3, stride=1, padding=1)
    unpooled = unpooled.squeeze(1)
    return torch.mean((latent - unpooled) ** 2)

# Test
print("Testing coherence losses...")
latent = torch.randn(4, 32, 32)
latent.requires_grad_(True)

# Test autocorr
loss1 = coherence_loss_autocorr(latent)
print(f"✓ autocorr loss: {loss1.item():.4f}")
loss1.backward()
assert latent.grad is not None
latent.grad.zero_()

# Test perimeter
loss2 = coherence_loss_perimeter(latent)
print(f"✓ perimeter loss: {loss2.item():.4f}")
loss2.backward()
assert latent.grad is not None
latent.grad.zero_()

# Test morphological
loss3 = coherence_loss_morphological(latent)
print(f"✓ morphological loss: {loss3.item():.4f}")
loss3.backward()
assert latent.grad is not None

print("\n✓ All coherence losses are differentiable and working!")
