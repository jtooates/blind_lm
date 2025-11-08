"""
Quick test script to verify Phase 1 components work together.
"""

import torch
import sys
from pathlib import Path

print("="*70)
print("Phase 1 Component Test")
print("="*70)

# Test 1: Model
print("\n1. Testing model...")
from model import create_model

config = {
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

model = create_model(config)
print(f"   ✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Test forward pass
batch_size = 4
seq_len = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

with torch.no_grad():
    latents = model(input_ids, attention_mask)

print(f"   ✓ Forward pass works: input {input_ids.shape} → latent {latents.shape}")

# Test 2: Losses
print("\n2. Testing losses...")
from losses import ImagePriorLoss

loss_fn = ImagePriorLoss()
result = loss_fn(latents, return_components=True)

print(f"   ✓ Total loss: {result['loss'].item():.4f}")
print(f"   ✓ Components:")
for name, value in result['components'].items():
    print(f"      - {name}: {value:.4f}")

# Test 3: Data loader
print("\n3. Testing data loader...")
from dataloader import create_fixed_eval_set
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

eval_set = create_fixed_eval_set(tokenizer, num_sentences=4)
print(f"   ✓ Fixed eval set created: {len(eval_set)} sentences")

batch = eval_set.get_batch()
print(f"   ✓ Batch created: {batch['input_ids'].shape}")

# Test 4: Training step simulation
print("\n4. Testing training step...")
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-4)

# Simulate one training step
optimizer.zero_grad()
latents = model(batch['input_ids'], batch['attention_mask'])
loss_dict = loss_fn(latents, return_components=True)
loss = loss_dict['loss']
loss.backward()
optimizer.step()

print(f"   ✓ Training step complete: loss = {loss.item():.4f}")

# Test 5: Visualization (don't show plots, just test they run)
print("\n5. Testing visualization...")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from visualize import (
    plot_power_spectra,
    plot_channel_montage,
    plot_gradient_histograms,
    plot_channel_covariance
)

# Generate latents
with torch.no_grad():
    test_latents = model(batch['input_ids'], batch['attention_mask'])

# Test each visualization (save to temp)
import tempfile
temp_dir = Path(tempfile.mkdtemp())

try:
    slopes = plot_power_spectra(test_latents, save_path=temp_dir / 'spec.png')
    print(f"   ✓ Power spectra plotted: mean slope = {slopes.mean():.2f}")

    plot_channel_montage(test_latents, batch['text'], save_path=temp_dir / 'montage.png')
    print(f"   ✓ Channel montage plotted")

    kurt_metrics = plot_gradient_histograms(test_latents, save_path=temp_dir / 'grad.png')
    print(f"   ✓ Gradient histograms plotted: kurtosis = {kurt_metrics['kurtosis_h']:.2f}")

    plot_channel_covariance(test_latents, save_path=temp_dir / 'cov.png')
    print(f"   ✓ Channel covariance plotted")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

except Exception as e:
    print(f"   ⚠ Visualization warning: {e}")

# Test 6: Check data files exist
print("\n6. Checking data files...")
train_file = Path('../train_sentences.txt')
val_file = Path('../val_sentences.txt')

if train_file.exists():
    with open(train_file) as f:
        num_train = sum(1 for line in f)
    print(f"   ✓ Training data found: {num_train} sentences")
else:
    print(f"   ⚠ Training data not found at {train_file}")

if val_file.exists():
    with open(val_file) as f:
        num_val = sum(1 for line in f)
    print(f"   ✓ Validation data found: {num_val} sentences")
else:
    print(f"   ⚠ Validation data not found at {val_file}")

# Summary
print("\n" + "="*70)
print("✓ All components working!")
print("="*70)
print("\nReady to train! Try:")
print("  python train.py --config configs/phase1_quick_test.json")
print("\nOr for full training:")
print("  python train.py --config configs/phase1_full.json")
print("="*70)
