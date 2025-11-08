# Phase 1: Image-like Latent Without Semantics

Train a text encoder to produce 2D visual latents (32×32×6) with natural image statistics, using only analytic priors (no semantic understanding).

## Overview

**Goal**: Make the latent grid V ∈ R^{H×W×C} look image-like under analytic priors only.

**Key Idea**: The training signal comes entirely from mathematical properties of natural images:
- 1/f² power spectrum
- Smooth regions with edges (total variation)
- Sparse wavelet coefficients
- Heavy-tailed gradients (kurtosis > 3)
- Decorrelated channels
- Stable variance

No reconstruction, no contrastive learning, no ground truth images - just image statistics!

## Architecture

- **Text Encoder**: 6-layer transformer (384 hidden dim, 8 heads) → 25.5M parameters
- **Grid Cross-Attention Head**: Maps token states to 32×32 spatial grid
- **Output**: 6-channel 2D latent with image-like properties

## Files

```
phase1/
├── model.py           # Text encoder architecture
├── losses.py          # Image prior losses
├── train.py           # Training script
├── dataloader.py      # Data loading utilities
├── visualize.py       # Visualization tools
├── configs/           # Configuration files
│   ├── phase1_quick_test.json  # Quick test (CPU, 500 steps)
│   └── phase1_full.json        # Full training (GPU, 50k steps)
└── README.md          # This file
```

## Quick Start

### 1. Generate Training Data

```bash
# From the parent directory
python generate_sentences.py --num 5000 --complexity 1 --seed 42 --output train_sentences.txt
python generate_sentences.py --num 500 --complexity 1 --seed 100 --output val_sentences.txt
```

### 2. Run Quick Test (CPU, ~5 minutes)

```bash
cd phase1
python train.py --config configs/phase1_quick_test.json
```

### 3. Run Full Training (GPU, ~hours)

```bash
cd phase1
python train.py --config configs/phase1_full.json
```

## Configuration

### Quick Test Config (`phase1_quick_test.json`)
- **Device**: CPU
- **Batch size**: 16
- **Steps**: 500 (2 epochs)
- **Purpose**: Verify everything works

### Full Config (`phase1_full.json`)
- **Device**: CUDA
- **Batch size**: 256
- **Steps**: 50,000 (10 epochs)
- **Purpose**: Actual Phase 1 training

### Key Hyperparameters

```json
{
  "training": {
    "lr": 2e-4,           # Learning rate
    "warmup_steps": 1000, # LR warmup
    "ema_decay": 0.999,   # EMA for model parameters
    "blur_warmup_steps": 2000  # Gaussian blur for first N steps
  },
  "loss": {
    "lambda_spec": 0.5,   # Spectrum loss weight
    "lambda_tv": 0.1,     # Total variation weight
    "lambda_wav": 0.1,    # Wavelet sparsity weight
    "lambda_kurt": 0.05,  # Gradient kurtosis weight
    "lambda_cov": 0.05,   # Channel decorrelation weight
    "lambda_var": 0.05    # Variance regularization weight
  }
}
```

## Monitoring Training

### During Training

The training script automatically logs:
- Total loss and components
- Learning rate
- Evaluation metrics every 500 steps

### After Training

Use visualization tools:

```python
from visualize import create_evaluation_report
from model import create_model
from dataloader import create_fixed_eval_set
from transformers import AutoTokenizer
import torch

# Load model
config = {...}  # Your config
model = create_model(config['model'])
checkpoint = torch.load('outputs/phase1/checkpoint_latest.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create eval set
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
eval_set = create_fixed_eval_set(tokenizer)

# Generate report
create_evaluation_report(model, eval_set, device='cpu', output_dir='./eval_report')
```

This generates:
- `power_spectra.png` - Log-log power spectra with fitted slopes
- `channel_montage.png` - Visual grid for 16 fixed sentences × 6 channels
- `gradient_histograms.png` - Gradient distributions showing kurtosis
- `channel_covariance.png` - Channel correlation matrix
- `evaluation_summary.json` - Numeric metrics

## Pass/Fail Criteria

**PASS when:**
- ≥ 4/6 channels have α ∈ [1.5, 2.5] for 3 consecutive evaluations
- TV plateaus and is non-zero
- Visuals stable across seeds (no speckle explosion)
- Channel covariance ≈ diagonal

**If FAIL:**
- Increase `lambda_tv` (e.g., +0.05) or reduce `lambda_var`
- Enable/extend blur warmup
- Reduce LR to 1e-4
- Keep the local smoothing head (`use_smooth_head: true`)

## Expected Metrics Evolution

### Epoch 0 (Random Initialization)
```
Spectrum slope α: ~0 (white noise)
TV loss: Very high (~2000)
Gradient kurtosis: ~3 (Gaussian)
Channel correlation: Already diagonal (good!)
```

### After Training
```
Spectrum slope α: 1.5-2.5 (natural images)
TV loss: Lower (~500-1000, non-zero)
Gradient kurtosis: >3 (heavy tails)
Channel correlation: Diagonal
Visuals: Smooth blobs with edges, no checkerboards
```

## Training Time Estimates

### Quick Test (CPU)
- **Steps**: 500
- **Time**: ~5-10 minutes
- **Purpose**: Verify code works

### Full Training (GPU)
- **Steps**: 50,000
- **Time**: ~2-4 hours (depending on GPU)
- **Purpose**: Achieve target metrics

## Troubleshooting

### Loss is NaN
- Reduce learning rate to 1e-4
- Enable gradient clipping (should be enabled by default)
- Check data isn't corrupted

### Checkerboard Artifacts
- Increase `lambda_tv` to 0.2
- Extend `blur_warmup_steps` to 5000
- Ensure `use_smooth_head: true`

### Spectrum Too Flat (α < 1.2)
- Increase `lambda_spec` to 1.0
- Add coordinate channels (see plan notes)
- Check that blur warmup is working

### Channels All Look Similar
- Increase `lambda_cov` to 0.1
- Reduce `lambda_var` to 0.01
- Check channel correlation matrix

## Next Steps

After Phase 1 passes:
- **Phase 2**: Add semantic meaning via contrastive learning
- **Phase 3**: Spatial jitter robustness
- **Phase 4**: Add text decoder
- **Phase 5**: Round-trip generation

## References

See `../visual-latent-plan-v2.md` for full project plan.
