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
from decoder_nonar import NonAutoregressiveTextDecoder, create_decoder
from infonce_losses import InfoNCELoss
from dataloader import create_dataloaders, create_fixed_eval_set


class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, models, decay=0.999):
        """
        Args:
            models: list of models or dict of {'name': model}
            decay: EMA decay rate
        """
        if isinstance(models, dict):
            self.models = models
        elif isinstance(models, (list, tuple)):
            self.models = {f'model_{i}': m for i, m in enumerate(models)}
        else:
            self.models = {'model': models}

        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for model_name, model in self.models.items():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    full_name = f'{model_name}.{name}'
                    self.shadow[full_name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for model_name, model in self.models.items():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    full_name = f'{model_name}.{name}'
                    self.shadow[full_name] = self.decay * self.shadow[full_name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for model_name, model in self.models.items():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    full_name = f'{model_name}.{name}'
                    self.backup[full_name] = param.data
                    param.data = self.shadow[full_name]

    def restore(self):
        """Restore original parameters"""
        for model_name, model in self.models.items():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    full_name = f'{model_name}.{name}'
                    param.data = self.backup[full_name]
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

        # Create encoder
        print("Creating encoder...")
        self.encoder = create_model(config['model']).to(self.device)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())/1e6
        print(f"Encoder has {encoder_params:.2f}M parameters")

        # Create decoder
        print("Creating decoder...")
        decoder_config = config.get('decoder', config['model']).copy()  # Use encoder config as default
        # Ensure decoder knows the number of visual channels from encoder
        decoder_config['num_visual_channels'] = config['model'].get('num_channels', 3)
        self.decoder = create_decoder(decoder_config).to(self.device)
        decoder_params = sum(p.numel() for p in self.decoder.parameters())/1e6
        print(f"Decoder has {decoder_params:.2f}M parameters")
        print(f"Total parameters: {encoder_params + decoder_params:.2f}M")

        # Create loss function - InfoNCE with RGB patch coherence
        self.criterion = InfoNCELoss(
            lambda_recon=config['loss'].get('lambda_recon', 5.0),
            lambda_infonce=config['loss'].get('lambda_infonce', 2.0),
            lambda_magnitude=config['loss'].get('lambda_magnitude', 5.0),
            lambda_spatial_diversity=config['loss'].get('lambda_spatial_diversity', 0.0),
            patch_size=config['loss'].get('infonce_patch_size', 3),
            num_samples=config['loss'].get('infonce_num_samples', 100),
            temperature=config['loss'].get('infonce_temperature', 1.0),
            positive_radius=config['loss'].get('infonce_positive_radius', 3.0),
            negative_radius=config['loss'].get('infonce_negative_radius', 11.0),
            min_magnitude=config['loss'].get('min_magnitude', 0.3),
            spatial_diversity_temperature=config['loss'].get('spatial_diversity_temperature', 0.5)
        ).to(self.device)

        # Create optimizer (both encoder and decoder)
        all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = AdamW(
            all_params,
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

        # EMA (both encoder and decoder)
        self.ema = EMA({'encoder': self.encoder, 'decoder': self.decoder}, decay=config['training']['ema_decay'])

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
        self.encoder.train()
        self.decoder.train()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Encoder forward pass
        latents = self.encoder(input_ids, attention_mask)  # [B, H, W, C]

        # Apply blur during warmup
        if self.step < self.blur_warmup_steps:
            latents = self.blur(latents)

        # Decoder forward pass (teacher forcing)
        # For teacher forcing, input is shifted by 1: [BOS, tok1, tok2, ...] predicts [tok1, tok2, ..., EOS]
        # For simplicity, we'll use the same input_ids (model learns to copy)
        logits = self.decoder(latents, input_ids, attention_mask)  # [B, L, V]

        # Compute loss (image priors + reconstruction)
        loss_dict = self.criterion(
            latents,
            logits=logits,
            target_ids=input_ids,
            return_components=True
        )
        loss = loss_dict['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (both encoder and decoder)
        all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        torch.nn.utils.clip_grad_norm_(
            all_params,
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

    def _print_reconstruction_examples(self, original_texts, predicted_texts, latents=None, num_examples=5):
        """Print side-by-side comparison of original and reconstructed text with RGB latent visualization"""
        print("\n" + "="*70)
        print(f"Reconstruction Examples (Step {self.step})")
        print("="*70)

        num_to_show = min(num_examples, len(original_texts))

        # Display RGB latents if provided
        if latents is not None:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create figure with subplots for RGB images
            fig, axes = plt.subplots(1, num_to_show, figsize=(3 * num_to_show, 3))
            if num_to_show == 1:
                axes = [axes]

            for i in range(num_to_show):
                # Convert latent [H, W, C] to RGB for display
                latent_rgb = latents[i].numpy()  # Already on CPU from evaluate()
                # Normalize from [-1.5, 1.5] to [0, 1]
                latent_rgb = (latent_rgb + 1.5) / 3.0
                latent_rgb = np.clip(latent_rgb, 0, 1)

                axes[i].imshow(latent_rgb)
                axes[i].set_title(f'[{i+1}]', fontsize=10)
                axes[i].axis('off')

            plt.suptitle('RGB Latents', fontsize=12, fontweight='bold')
            plt.tight_layout()

            # Display in notebook by rendering to bytes and using IPython.display.Image
            try:
                from IPython.display import display, Image as IPImage
                import io
                import base64

                # Render figure to PNG bytes
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_data = buf.read()
                buf.close()

                # Try multiple display methods for compatibility
                try:
                    # Method 1: Direct display with Image object
                    from IPython.display import HTML
                    encoded = base64.b64encode(img_data).decode('ascii')
                    html = f'<img src="data:image/png;base64,{encoded}"/>'
                    display(HTML(html))
                except:
                    # Method 2: Fallback to IPImage
                    display(IPImage(data=img_data))

                plt.close(fig)
            except (ImportError, NameError):
                # Fallback for non-notebook environments
                plt.show()
                plt.close(fig)

        # Print text reconstructions
        for i in range(num_to_show):
            orig = original_texts[i]
            pred = predicted_texts[i]

            # Check if exact match
            match_symbol = "✓" if orig == pred else "✗"

            print(f"\n[{i+1}] {match_symbol}")
            print(f"  Original:    {orig}")
            print(f"  Predicted:   {pred}")

        # Compute accuracy
        exact_matches = sum(1 for o, p in zip(original_texts, predicted_texts) if o == p)
        accuracy = exact_matches / len(original_texts) * 100
        print(f"\n  Exact match accuracy: {exact_matches}/{len(original_texts)} ({accuracy:.1f}%)")
        print("="*70 + "\n")

    @torch.no_grad()
    def _show_training_examples(self, num_examples=5):
        """Show reconstruction examples from training data with RGB latents"""
        self.encoder.eval()
        self.decoder.eval()

        # Apply EMA weights
        self.ema.apply_shadow()

        # Get a batch from training data
        train_batch = next(iter(self.train_loader))
        train_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in train_batch.items()}

        # Take only num_examples from the batch
        input_ids = train_batch['input_ids'][:num_examples]
        attention_mask = train_batch['attention_mask'][:num_examples]
        original_texts = train_batch['text'][:num_examples]

        # Forward pass
        latents = self.encoder(input_ids, attention_mask)
        logits = self.decoder(latents, input_ids, attention_mask)

        # Decode predictions (argmax)
        predicted_ids = logits.argmax(dim=-1)

        # Decode to text
        predicted_texts = []
        for i in range(predicted_ids.shape[0]):
            pred_text = self.tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
            predicted_texts.append(pred_text)

        # Restore original weights
        self.ema.restore()

        # Display with a different title
        print("\n" + "="*70)
        print(f"Training Data Examples (Step {self.step})")
        print("="*70)

        # Display RGB latents
        import matplotlib.pyplot as plt
        import numpy as np

        # Create figure with subplots for RGB images
        fig, axes = plt.subplots(1, num_examples, figsize=(3 * num_examples, 3))
        if num_examples == 1:
            axes = [axes]

        for i in range(num_examples):
            # Convert latent [H, W, C] to RGB for display
            latent_rgb = latents[i].cpu().numpy()
            # Normalize from [-1.5, 1.5] to [0, 1]
            latent_rgb = (latent_rgb + 1.5) / 3.0
            latent_rgb = np.clip(latent_rgb, 0, 1)

            axes[i].imshow(latent_rgb)
            axes[i].set_title(f'[{i+1}]', fontsize=10)
            axes[i].axis('off')

        plt.suptitle('RGB Latents (Training Data)', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Display in notebook
        try:
            from IPython.display import display, Image as IPImage
            import io
            import base64

            # Render figure to PNG bytes
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_data = buf.read()
            buf.close()

            # Try multiple display methods for compatibility
            try:
                # Method 1: Direct display with Image object
                from IPython.display import HTML
                encoded = base64.b64encode(img_data).decode('ascii')
                html = f'<img src="data:image/png;base64,{encoded}"/>'
                display(HTML(html))
            except:
                # Method 2: Fallback to IPImage
                display(IPImage(data=img_data))

            plt.close(fig)
        except (ImportError, NameError):
            plt.show()
            plt.close(fig)

        # Print text reconstructions
        for i in range(num_examples):
            orig = original_texts[i]
            pred = predicted_texts[i]

            # Check if exact match
            match_symbol = "✓" if orig == pred else "✗"

            print(f"\n[{i+1}] {match_symbol}")
            print(f"  Original:    {orig}")
            print(f"  Predicted:   {pred}")

        # Compute accuracy
        exact_matches = sum(1 for o, p in zip(original_texts, predicted_texts) if o == p)
        accuracy = exact_matches / len(original_texts) * 100
        print(f"\n  Exact match accuracy: {exact_matches}/{len(original_texts)} ({accuracy:.1f}%)")
        print("="*70 + "\n")

        # Return to training mode
        self.encoder.train()
        self.decoder.train()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on fixed set"""
        self.encoder.eval()
        self.decoder.eval()

        # Apply EMA weights
        self.ema.apply_shadow()

        # Get fixed evaluation batch
        eval_batch = self.eval_set.get_batch(device=self.device)

        # Forward pass
        latents = self.encoder(eval_batch['input_ids'], eval_batch['attention_mask'])
        logits = self.decoder(latents, eval_batch['input_ids'], eval_batch['attention_mask'])

        # Compute metrics
        loss_dict = self.criterion(
            latents,
            logits=logits,
            target_ids=eval_batch['input_ids'],
            return_components=True
        )

        # Decode predictions (argmax)
        predicted_ids = logits.argmax(dim=-1)  # [B, L]

        # Decode to text
        original_texts = eval_batch['text']
        predicted_texts = []
        for i in range(predicted_ids.shape[0]):
            # Decode tokens to text
            pred_text = self.tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
            predicted_texts.append(pred_text)

        # Restore original weights
        self.ema.restore()

        return {
            'eval_loss': loss_dict['loss'].item(),
            'eval_metrics': loss_dict['metrics'],
            'eval_components': loss_dict['components'],
            'eval_latents': latents.cpu(),
            'original_texts': original_texts,
            'predicted_texts': predicted_texts
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.epoch += 1

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        epoch_metrics = []

        for batch in pbar:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            # Update progress bar with all loss components
            postfix_dict = {
                'loss': f"{metrics['loss']:.4f}",
                'recon': f"{metrics.get('recon_loss', 0):.3f}",
                'info': f"{metrics.get('infonce_loss', 0):.3f}",
                'mag': f"{metrics.get('magnitude_loss', 0):.3f}",
            }
            # Add spatial diversity if present
            if 'spatial_diversity_loss' in metrics:
                postfix_dict['spatial_div'] = f"{metrics['spatial_diversity_loss']:.3f}"
            postfix_dict['lr'] = f"{metrics['lr']:.2e}"
            pbar.set_postfix(postfix_dict)

            # Evaluate periodically
            if self.step % self.config['eval']['eval_interval'] == 0:
                eval_metrics = self.evaluate()
                self.metrics_history.append({
                    'step': self.step,
                    'epoch': self.epoch,
                    **eval_metrics
                })

                # Print reconstruction examples with RGB latents
                # Use fixed eval set for metrics (consistent tracking)
                self._print_reconstruction_examples(
                    eval_metrics['original_texts'],
                    eval_metrics['predicted_texts'],
                    eval_metrics['eval_latents'],
                    num_examples=5
                )

                # Additionally, show training examples with their latents
                self._show_training_examples(num_examples=5)

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
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
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
