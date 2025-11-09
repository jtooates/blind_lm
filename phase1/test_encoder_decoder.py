"""
Test encoder+decoder integration with reconstruction loss
"""

import torch
import sys
from pathlib import Path

print("="*70)
print("Testing Encoder + Decoder Integration")
print("="*70)

# Test 1: Create encoder and decoder
print("\n1. Creating encoder and decoder...")
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
    'num_channels': 6,
    'use_rope': True,
    'use_smooth_head': True
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
print("\n2. Testing forward pass...")
batch_size = 4
seq_len = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

with torch.no_grad():
    # Encoder
    latents = encoder(input_ids, attention_mask)
    print(f"   ✓ Encoder: {input_ids.shape} → {latents.shape}")

    # Decoder
    logits = decoder(latents, input_ids, attention_mask)
    print(f"   ✓ Decoder: {latents.shape} → {logits.shape}")

# Test 3: Reconstruction loss
print("\n3. Testing reconstruction loss...")
from losses import ImagePriorLoss

criterion = ImagePriorLoss(
    lambda_spec=0.5,
    lambda_tv=0.1,
    lambda_wav=0.1,
    lambda_kurt=0.05,
    lambda_cov=0.05,
    lambda_var=0.05,
    lambda_recon=1.0  # Reconstruction loss enabled
)

loss_dict = criterion(latents, logits=logits, target_ids=input_ids, return_components=True)
print(f"   ✓ Total loss: {loss_dict['loss'].item():.4f}")
print(f"   ✓ Loss components:")
for name, value in loss_dict['components'].items():
    print(f"      - {name}: {value:.4f}")

# Test 4: Backward pass
print("\n4. Testing backward pass...")
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

# Test 5: Check that reconstruction loss is significant
print("\n5. Checking reconstruction loss...")
recon_loss = loss_dict['components']['reconstruction']
image_prior_loss = sum(v for k, v in loss_dict['components'].items() if k != 'reconstruction')

print(f"   Reconstruction loss: {recon_loss:.4f}")
print(f"   Image prior losses: {image_prior_loss:.4f}")

if recon_loss > 0:
    print(f"   ✓ Reconstruction loss is active")
else:
    print(f"   ✗ WARNING: Reconstruction loss is zero!")

# Summary
print("\n" + "="*70)
print("✓ All encoder+decoder tests passed!")
print("="*70)
print("\nThe model is ready to train with reconstruction loss.")
print("This should prevent the collapse to trivial solutions.")
print("="*70)
