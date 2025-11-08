"""
Phase 1: Training Script
Train text encoder to produce image-like latents using only analytic priors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import time

from model import TextEncoder, create_model
from losses import ImagePriorLoss
from dataloader import create_dataloaders, create_fixed_eval_set


class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class GaussianBlur(nn.Module):
    """Apply Gaussian blur to latent for warmup stability"""

    def __init__(self, sigma=0.8):
        super().__init__()
        self.sigma = sigma

        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 2D Gaussian
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()

        self.register_buffer('kernel', gauss_2d)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C] latent
        Returns:
            blurred: [B, H, W, C] blurred latent
        """
        B, H, W, C = x.shape

        # Convert to BCHW
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply blur to each channel
        padding = self.kernel_size // 2
        blurred = F.conv2d(
            x,
            self.kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1),
            padding=padding,
            groups=C
        )

        # Convert back to BHWC
        blurred = blurred.permute(0, 2, 3, 1)

        return blurred


class Trainer:
    """Phase 1 Trainer"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Create model
        print("Creating model...")
        self.model = create_model(config['model']).to(self.device)
        print(f"Model has {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")

        # Create loss function
        self.criterion = ImagePriorLoss(
            lambda_spec=config['loss']['lambda_spec'],
            lambda_tv=config['loss']['lambda_tv'],
            lambda_wav=config['loss']['lambda_wav'],
            lambda_kurt=config['loss']['lambda_kurt'],
            lambda_cov=config['loss']['lambda_cov'],
            lambda_var=config['loss']['lambda_var']
        ).to(self.device)

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            betas=(config['training']['beta1'], config['training']['beta2']),
            weight_decay=config['training']['weight_decay']
        )

        # Create learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=config['training']['warmup_steps']
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['max_steps'] - config['training']['warmup_steps']
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config['training']['warmup_steps']]
        )

        # EMA
        self.ema = EMA(self.model, decay=config['training']['ema_decay'])

        # Gaussian blur for warmup
        self.blur = GaussianBlur(sigma=config['training'].get('blur_sigma', 0.8)).to(self.device)
        self.blur_warmup_steps = config['training'].get('blur_warmup_steps', 0)

        # Training state
        self.step = 0
        self.epoch = 0

        # Load dataloaders
        print("Loading data...")
        dataloaders, self.tokenizer = create_dataloaders(
            train_file=config['data']['train_file'],
            val_file=config['data'].get('val_file'),
            tokenizer_name=config['model'].get('tokenizer_name', 'gpt2'),
            batch_size=config['training']['batch_size'],
            max_length=config['model']['max_seq_len'],
            num_workers=config['data'].get('num_workers', 4),
            file_format=config['data'].get('file_format', 'txt')
        )

        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders.get('val')

        # Fixed evaluation set
        self.eval_set = create_fixed_eval_set(
            self.tokenizer,
            num_sentences=config['eval'].get('num_fixed_sentences', 16)
        )

        # Metrics tracking
        self.metrics_history = []

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Forward pass
        latents = self.model(input_ids, attention_mask)  # [B, H, W, C]

        # Apply blur during warmup
        if self.step < self.blur_warmup_steps:
            latents = self.blur(latents)

        # Compute loss
        loss_dict = self.criterion(latents, return_components=True)
        loss = loss_dict['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training'].get('grad_clip', 1.0)
        )

        self.optimizer.step()
        self.scheduler.step()

        # Update EMA
        self.ema.update()

        self.step += 1

        return {
            'loss': loss.item(),
            **loss_dict['components'],
            'lr': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on fixed set"""
        self.model.eval()

        # Apply EMA weights
        self.ema.apply_shadow()

        # Get fixed evaluation batch
        eval_batch = self.eval_set.get_batch(device=self.device)

        # Forward pass
        latents = self.model(eval_batch['input_ids'], eval_batch['attention_mask'])

        # Compute metrics
        loss_dict = self.criterion(latents, return_components=True)

        # Restore original weights
        self.ema.restore()

        return {
            'eval_loss': loss_dict['loss'].item(),
            'eval_metrics': loss_dict['metrics'],
            'eval_components': loss_dict['components'],
            'eval_latents': latents.cpu()
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.epoch += 1

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        epoch_metrics = []

        for batch in pbar:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['lr']:.2e}"
            })

            # Evaluate periodically
            if self.step % self.config['eval']['eval_interval'] == 0:
                eval_metrics = self.evaluate()
                self.metrics_history.append({
                    'step': self.step,
                    'epoch': self.epoch,
                    **eval_metrics
                })

                # Save checkpoint
                if self.step % self.config['eval']['save_interval'] == 0:
                    self.save_checkpoint()

        # Compute epoch statistics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if key != 'lr':
                avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)

        return avg_metrics

    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f'checkpoint_step_{self.step}.pt'

        # Apply EMA weights for saving
        self.ema.apply_shadow()

        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'config': self.config,
            'metrics_history': self.metrics_history
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Restore original weights
        self.ema.restore()

        # Also save as latest
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs...")
        print(f"Total steps: {self.config['training']['max_steps']}")
        print(f"Blur warmup steps: {self.blur_warmup_steps}")

        for epoch in range(self.config['training']['num_epochs']):
            epoch_metrics = self.train_epoch()

            print(f"\nEpoch {self.epoch} summary:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Check stopping criteria
            if self.step >= self.config['training']['max_steps']:
                print(f"Reached maximum steps ({self.config['training']['max_steps']})")
                break

        print("Training complete!")
        self.save_checkpoint()


def create_default_config():
    """Create default configuration"""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': './outputs/phase1',

        'model': {
            'vocab_size': 50257,  # GPT-2 tokenizer size
            'max_seq_len': 64,
            'hidden_size': 384,
            'num_layers': 6,
            'num_heads': 8,
            'ffn_size': 1536,
            'dropout': 0.1,
            'grid_size': 32,
            'num_channels': 6,
            'use_rope': True,
            'use_smooth_head': True,
            'tokenizer_name': 'gpt2'
        },

        'loss': {
            'lambda_spec': 0.5,
            'lambda_tv': 0.1,
            'lambda_wav': 0.1,
            'lambda_kurt': 0.05,
            'lambda_cov': 0.05,
            'lambda_var': 0.05
        },

        'training': {
            'batch_size': 256,
            'lr': 2e-4,
            'beta1': 0.9,
            'beta2': 0.95,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'num_epochs': 10,
            'max_steps': 50000,
            'ema_decay': 0.999,
            'grad_clip': 1.0,
            'blur_sigma': 0.8,
            'blur_warmup_steps': 2000
        },

        'data': {
            'train_file': '../train_sentences.txt',
            'val_file': None,
            'num_workers': 4,
            'file_format': 'txt'
        },

        'eval': {
            'eval_interval': 500,
            'save_interval': 2000,
            'num_fixed_sentences': 16
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Training')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--train_file', type=str, help='Path to training data')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()

    # Override with command line args
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.device:
        config['device'] = args.device

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
