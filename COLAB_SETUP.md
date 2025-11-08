# Google Colab Setup Guide

Quick guide to run Phase 1 training on Google Colab.

## Option 1: Use the Notebook (Recommended)

1. **Upload the notebook to Colab**:
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Upload `phase1_colab_training.ipynb`

2. **Or open from GitHub** (after you push):
   - Go to https://colab.research.google.com/
   - File → Open notebook → GitHub
   - Enter: `jtooates/blind_lm`
   - Select: `phase1_colab_training.ipynb`

3. **Set GPU runtime**:
   - Runtime → Change runtime type → T4 GPU

4. **Run all cells**:
   - Runtime → Run all
   - Or click through cells one by one

## Option 2: Manual Setup

If you prefer to set up manually:

### 1. Clone Repository
```python
!git clone https://github.com/jtooates/blind_lm.git
%cd blind_lm
```

### 2. Install Dependencies
```python
!pip install -q transformers scipy tqdm matplotlib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### 3. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Check GPU
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 5. Run Training
```python
%cd phase1
!python train.py --config configs/phase1_full.json --device cuda --output_dir /content/drive/MyDrive/blind_lm_outputs/phase1
```

## What to Expect

### Training Time
- **T4 GPU**: ~2-3 hours for 50k steps
- **Free Colab**: May disconnect after ~12 hours (checkpoints save automatically)

### Checkpoints
Saved every 2,000 steps to:
```
/content/drive/MyDrive/blind_lm_outputs/phase1/
```

### Progress Monitoring
Watch for:
- Loss decreasing (starts ~4-5, should go to ~0.5-1.0)
- Learning rate warming up to 2e-4
- Regular checkpoint saves

## Troubleshooting

### "No GPU available"
- Go to Runtime → Change runtime type → T4 GPU
- Restart runtime and run again

### "Session disconnected"
- Checkpoints are saved to Drive - safe to resume
- Rerun from the training cell (will load from latest checkpoint)

### "Out of memory"
- Reduce batch size in config (128 → 64)
- Or use gradient accumulation

### Warnings about tokenizers/schedulers
- Safe to ignore - these are PyTorch/HuggingFace deprecation warnings
- Code works correctly despite warnings

## After Training

### View Results
The notebook automatically generates:
- Power spectra plots
- Channel montage visualizations
- Gradient histograms
- Channel covariance heatmaps
- Evaluation summary JSON

### Download Results
Use the final cell to download a zip with:
- Final checkpoint
- All visualizations
- Evaluation metrics

## Tips

1. **Keep Drive mounted**: Prevents losing checkpoints if disconnected
2. **Monitor GPU usage**: Runtime → Manage sessions
3. **Use longer runtime**: Colab Pro gives longer sessions
4. **Resume training**: Just rerun the training cell - loads latest checkpoint automatically

## Files in Repository

Make sure these are pushed to GitHub:
```
blind_lm/
├── phase1_colab_training.ipynb  ← Upload this to Colab
├── dataset_generator.py
├── augmentations.py
├── generate_sentences.py
├── train_sentences.txt          ← Optional (can regenerate)
├── val_sentences.txt            ← Optional (can regenerate)
└── phase1/
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── dataloader.py
    ├── visualize.py
    └── configs/
        └── phase1_full.json
```

## Quick Start Checklist

- [ ] Push code to GitHub
- [ ] Open Colab notebook
- [ ] Set runtime to GPU
- [ ] Run all cells
- [ ] Wait 2-3 hours
- [ ] Check results
- [ ] Download checkpoints

That's it! The notebook handles everything else automatically.
