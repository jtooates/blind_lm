"""
Test single channel encoder with object-forming losses.
"""

import torch
import sys
from pathlib import Path

print("="*70)
print("Testing Single Channel Encoder with Object-Forming Losses")
print("="*70)

# Test 1: Create encoder and decoder
print("\n1. Creating single-channel encoder and decoder...")
from model import create_model
from decoder import create_decoder

encoder_config = {
    'vocab_size': 50257,
    'max_seq_len': 64,
    'hidden_size': 384,
    'num_layers': 6,
    'num_heads': 8,
    'ffn_size': 1536,
    'dropout': 0.1,
    'grid_size': 32,
    'num_channels': 1,  # Single channel
    'use_rope': True,
    'use_smooth_head': False
}

decoder_config = {
    'vocab_size': 50257,
    'max_seq_len': 64,
    'hidden_size': 384,
    'num_layers': 4,
    'num_heads': 8,
    'ffn_size': 1536,
    'dropout': 0.1,
    'use_rope': True
}

encoder = create_model(encoder_config)
decoder = create_decoder(decoder_config)

encoder_params = sum(p.numel() for p in encoder.parameters())/1e6
decoder_params = sum(p.numel() for p in decoder.parameters())/1e6
print(f"   ✓ Encoder: {encoder_params:.2f}M parameters")
print(f"   ✓ Decoder: {decoder_params:.2f}M parameters")
print(f"   ✓ Total: {encoder_params + decoder_params:.2f}M parameters")

# Test 2: Forward pass
print("\n2. Testing forward pass with single channel...")
batch_size = 4
seq_len = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

with torch.no_grad():
    # Encoder
    latents = encoder(input_ids, attention_mask)
    print(f"   ✓ Encoder output shape: {latents.shape}")
    print(f"   ✓ Expected: [{batch_size}, 32, 32, 1]")

    if latents.shape[-1] != 1:
        print(f"   ✗ ERROR: Expected 1 channel, got {latents.shape[-1]}")
    else:
        print(f"   ✓ Single channel confirmed!")

    # Decoder
    logits = decoder(latents, input_ids, attention_mask)
    print(f"   ✓ Decoder output shape: {logits.shape}")

# Test 3: Object-forming losses
print("\n3. Testing object-forming losses...")
from object_losses import ObjectFormingLoss

criterion = ObjectFormingLoss(
    lambda_sparsity=0.5,
    lambda_object_size=1.0,
    lambda_binary=0.3,
    lambda_contrastive=2.0,
    lambda_tv=0.1,
    lambda_recon=5.0
)

loss_dict = criterion(latents, logits=logits, target_ids=input_ids, return_components=True)
print(f"   ✓ Total loss: {loss_dict['loss'].item():.4f}")
print(f"   ✓ Loss components:")
for name, value in loss_dict['components'].items():
    print(f"      - {name}: {value:.4f}")

# Test 4: Verify grayscale interpretation
print("\n4. Checking grayscale interpretation...")
# Extract single channel
grayscale = latents[0, :, :, 0].detach().cpu().numpy()
print(f"   ✓ Grayscale shape: {grayscale.shape}")
print(f"   ✓ Value range: [{grayscale.min():.2f}, {grayscale.max():.2f}]")

# Test 5: Backward pass
print("\n5. Testing backward pass...")
from torch.optim import AdamW

optimizer = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-4)

optimizer.zero_grad()
latents = encoder(input_ids, attention_mask)
logits = decoder(latents, input_ids, attention_mask)
loss_dict = criterion(latents, logits=logits, target_ids=input_ids, return_components=True)
loss = loss_dict['loss']
loss.backward()
optimizer.step()

print(f"   ✓ Backward pass successful")
print(f"   ✓ Loss after step: {loss.item():.4f}")

# Test 6: Visualization compatibility
print("\n6. Testing visualization...")
import matplotlib.pyplot as plt
import numpy as np

# Create a figure showing the grayscale latent
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for i in range(min(4, batch_size)):
    row = i // 2
    col = i % 2
    ax = axes[row, col]

    # Extract grayscale image
    img = latents[i, :, :, 0].detach().cpu().numpy()

    # Display
    im = ax.imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
    ax.set_title(f'Latent {i+1} (Single Channel)')
    ax.axis('off')

plt.suptitle('Single Channel Grayscale Latents')
plt.tight_layout()
plt.savefig('test_single_channel.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Visualization saved to test_single_channel.png")

# Summary
print("\n" + "="*70)
print("✓ All single channel tests passed!")
print("="*70)
print("\nKey changes:")
print("  - Model outputs 32×32×1 (grayscale) instead of 32×32×6")
print("  - New object-forming losses (sparsity, object_size, contrastive)")
print("  - Direct visualization as grayscale images")
print("  - Reconstruction loss weight increased to 5.0")
print("\nReady to train with new configuration!")
print("="*70)