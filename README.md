# Blind LM: Visual Latent Language Model

A self-supervised text encoder that learns 2D "visual" latent representations without ever seeing images.

## Overview

This project implements a novel approach to text encoding where the latent space has image-like properties. The key insight: by forcing text embeddings to have natural image statistics (1/f² spectrum, sparse edges, etc.), we create useful compositional structure without requiring ground truth images or labeled data.

**Training signal**: Pure mathematical properties of natural images
**No ground truth**: No images, no reconstruction targets, no labels
**Fully self-supervised**: Only text and analytic priors

## Project Status

✅ **Phase 0**: Dataset generation (synthetic colored blocks)
✅ **Phase 1**: Image-like latent encoder (ready to train)
🔲 **Phase 2**: Contrastive learning (paraphrases)
🔲 **Phase 3**: Spatial robustness
🔲 **Phase 4**: Text decoder
🔲 **Phase 5**: Round-trip generation

## Quick Start

### 1. Generate Training Data

```bash
# 10,000 training sentences
python generate_sentences.py --num 10000 --complexity 1 --seed 42 --output train_sentences.txt

# 1,000 validation sentences
python generate_sentences.py --num 1000 --complexity 1 --seed 100 --output val_sentences.txt
```

### 2. Train Phase 1

**Local (CPU) - Quick test:**
```bash
cd phase1
python train.py --config configs/phase1_quick_test.json
```

**Local (GPU) - Full training:**
```bash
cd phase1
python train.py --config configs/phase1_full.json
```

**Google Colab (Recommended):**
1. Upload `phase1_colab_training.ipynb` to Colab
2. Set runtime to GPU (T4)
3. Run all cells
4. Wait ~2-3 hours

See [COLAB_SETUP.md](COLAB_SETUP.md) for details.

### 3. View Results

```python
from phase1.visualize import create_evaluation_report
from phase1.model import create_model
from phase1.dataloader import create_fixed_eval_set
from transformers import AutoTokenizer
import torch

# Load model
model = create_model(config)
checkpoint = torch.load('outputs/phase1/checkpoint_latest.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate report
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
eval_set = create_fixed_eval_set(tokenizer)
create_evaluation_report(model, eval_set, output_dir='./eval_report')
```

## Repository Structure

```
blind_lm/
├── README.md                      # This file
├── COLAB_SETUP.md                # Colab instructions
├── PHASE1_SUMMARY.md             # Phase 1 details
├── visual-latent-plan-v2.md      # Full project plan
│
├── dataset_generator.py          # Scene graph generation
├── augmentations.py              # Paraphrases & counterfactuals
├── generate_sentences.py         # CLI for data generation
├── generate_with_augmentations.py # Full dataset pipeline
│
├── train_sentences.txt           # Training data (generated)
├── val_sentences.txt             # Validation data (generated)
│
├── phase1/                       # Phase 1: Image-like latents
│   ├── model.py                  # Text encoder (32.5M params)
│   ├── losses.py                 # Image prior losses
│   ├── train.py                  # Training script
│   ├── dataloader.py             # Data loading
│   ├── visualize.py              # Monitoring tools
│   ├── test_phase1.py            # Component tests
│   ├── README.md                 # Phase 1 docs
│   └── configs/                  # Training configs
│       ├── phase1_quick_test.json
│       ├── phase1_full.json
│       └── phase1_colab.json
│
└── phase1_colab_training.ipynb   # Colab notebook
```

## Phase 1: Image-like Latents

### Architecture

- **Text Encoder**: 6-layer transformer (384 hidden, 8 heads)
- **Grid Head**: Cross-attention to map tokens → 32×32×6 visual latent
- **Parameters**: ~32.5M

### Training Objective

Learn latents V ∈ R^{32×32×6} that have:
- 1/f² power spectrum (natural images)
- Smooth regions with edges (total variation)
- Sparse wavelet coefficients
- Heavy-tailed gradients (kurtosis > 3)
- Independent channels
- Stable variance

**No semantic understanding yet!** That comes in Phase 2.

### Pass Criteria

- ≥ 4/6 channels with power spectrum slope α ∈ [1.5, 2.5]
- Non-zero total variation
- Gradient kurtosis > 3
- Diagonal channel covariance
- No checkerboard artifacts

## Example Output

```python
# Input text
"the red block is on the blue cube"

# Output: 32×32×6 visual latent grid
# - Smooth blobs and edges
# - Natural image statistics
# - No semantic meaning (yet)
```

## Synthetic Dataset

The dataset consists of simple colored block descriptions:

```
the red block is on the blue cube
the green box is next to the yellow block
the purple cube is under the orange box
...
```

**Phase 0 Tools:**
- `SceneGenerator`: Creates spatial arrangements
- `TextRenderer`: Converts to natural language
- `MeaningPreservingAugmenter`: Paraphrases (synonym swaps, passive/active, etc.)
- `CounterfactualGenerator`: Meaning-breaking variations (color flips, relation changes, etc.)

## Requirements

```bash
pip install torch transformers matplotlib scipy tqdm
```

Or see `requirements.txt` (if created).

## Training Time

- **Quick test** (500 steps, CPU): ~5-10 minutes
- **Full training** (50k steps, GPU): ~2-4 hours
- **Colab T4 GPU**: ~2-3 hours

## Key Concepts

### Why "Visual" Latents?

The hypothesis: 2D spatial structure with image-like statistics provides useful inductive bias for compositional understanding, even without seeing actual images.

### Why No Decoder Yet?

Phase 1 focuses purely on latent structure. Adding semantics too early might cause trivial solutions. By first establishing good visual structure, we create a better foundation for later phases.

### Training Signal

```python
loss = (
    0.5 * spectrum_loss +      # 1/f² power spectrum
    0.1 * tv_loss +            # Smooth with edges
    0.1 * wavelet_loss +       # Sparse edges
    0.05 * kurtosis_loss +     # Heavy-tailed gradients
    0.05 * decorr_loss +       # Independent channels
    0.05 * variance_loss       # Stable variance
)
```

Pure image statistics - no semantic information!

## Visualization Tools

Phase 1 includes comprehensive monitoring:
- Power spectrum plots (log-log with fitted slopes)
- Channel montages (16 sentences × 6 channels)
- Gradient histograms (kurtosis visualization)
- Channel covariance heatmaps
- Training curves (loss components over time)

## Troubleshooting

See `phase1/README.md` for detailed troubleshooting:
- Loss is NaN → reduce learning rate
- Checkerboard artifacts → increase TV weight
- Spectrum too flat → increase spectrum weight
- Channels look similar → increase decorrelation weight

## Testing

```bash
# Test all components
cd phase1
python test_phase1.py
```

This verifies:
- ✅ Model architecture
- ✅ Loss computation
- ✅ Data loading
- ✅ Training step
- ✅ Visualizations

## Next Phases

### Phase 2: Semantic Meaning
Add contrastive learning:
- Paraphrases → similar latents
- Counterfactuals → different latents
- Uses augmentation data from Phase 0

### Phase 3: Spatial Robustness
Make latents invariant to small visual transforms (shifts, blur, cutout)

### Phase 4: Text Decoder
Add decoder to reconstruct text from latent (teacher forcing)

### Phase 5: Round-Trip Generation
Generate paraphrases without copying

## References

See `visual-latent-plan-v2.md` for complete implementation plan.

## License

[Your chosen license]

## Citation

```bibtex
@software{blind_lm,
  title={Blind LM: Visual Latent Language Model},
  author={[Your name]},
  year={2025},
  url={https://github.com/jtooates/blind_lm}
}
```

## Contact

Issues: https://github.com/jtooates/blind_lm/issues

---

**Current Status**: Phase 1 implementation complete and ready for training! 🚀
