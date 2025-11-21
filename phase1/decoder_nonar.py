"""
Phase 1: Non-Autoregressive Text Decoder
Reconstructs text from visual latent ONLY - no access to input tokens!
"""

import torch
import torch.nn as nn
import math


class NonAutoregressiveTextDecoder(nn.Module):
    """
    Non-autoregressive decoder that generates all tokens in parallel from visual latent.

    Key difference from autoregressive decoder:
    - NO access to input_ids (no teacher forcing, no information leak!)
    - Generates purely from visual latent
    - All positions decoded in parallel

    Architecture:
        Visual Latent [B, H, W, C] → Flatten → [B, H*W*C]
        → Project to sequence → [B, max_seq_len, hidden_size]
        → Add positional embeddings
        → Transformer layers (self-attention only)
        → Output projection → [B, max_seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 64,
        hidden_size: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_size: int = 1536,
        dropout: float = 0.1,
        use_rope: bool = True,  # Kept for compatibility, but not used
        num_visual_channels: int = 3,
        latent_size: int = 32  # Assuming 32x32 latent
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_visual_channels = num_visual_channels
        self.latent_size = latent_size

        # Calculate flattened latent dimension
        latent_dim = latent_size * latent_size * num_visual_channels

        # Two-layer projection with intermediate bottleneck for memory efficiency
        # Instead of: latent_dim → max_seq_len * hidden_size (huge!)
        # Use: latent_dim → intermediate → max_seq_len * hidden_size
        intermediate_dim = min(2048, latent_dim)  # Bottleneck dimension
        self.latent_project = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, max_seq_len * hidden_size)
        )

        # Positional embeddings (learned)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer layers (self-attention only, no cross-attention)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        # Initialize projection layers
        for module in self.latent_project:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_latent: torch.Tensor,
        input_ids: torch.Tensor = None,  # Kept for API compatibility but IGNORED!
        attention_mask: torch.Tensor = None
    ):
        """
        Generate text tokens from visual latent only.

        Args:
            visual_latent: [B, H, W, C] visual latent from encoder
            input_ids: IGNORED! Kept for API compatibility only.
            attention_mask: Optional [B, L] mask for padding positions

        Returns:
            logits: [B, max_seq_len, vocab_size] output logits for all positions
        """
        B, H, W, C = visual_latent.shape

        # 1. Flatten visual latent: [B, H, W, C] → [B, H*W*C]
        visual_flat = visual_latent.reshape(B, H * W * C)

        # 2. Project to sequence: [B, H*W*C] → [B, max_seq_len * hidden_size]
        # Uses intermediate bottleneck for memory efficiency
        seq_flat = self.latent_project(visual_flat)

        # 3. Reshape to sequence: [B, max_seq_len * hidden_size] → [B, max_seq_len, hidden_size]
        x = seq_flat.reshape(B, self.max_seq_len, self.hidden_size)

        # 4. Add positional embeddings
        positions = torch.arange(self.max_seq_len, device=visual_latent.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)

        # 5. Apply transformer layers
        # Create attention mask if provided (for padding positions)
        if attention_mask is not None:
            # attention_mask: [B, L] with 1 for real tokens, 0 for padding
            # TransformerEncoderLayer expects: [B, L] with True for positions to MASK
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        # 6. Output projection
        x = self.output_norm(x)
        logits = self.output_projection(x)  # [B, max_seq_len, vocab_size]

        return logits


def create_decoder(config):
    """Create non-autoregressive decoder from config dict"""
    return NonAutoregressiveTextDecoder(
        vocab_size=config.get('vocab_size', 50257),
        max_seq_len=config.get('max_seq_len', 64),
        hidden_size=config.get('hidden_size', 384),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        ffn_size=config.get('ffn_size', 1536),
        dropout=config.get('dropout', 0.1),
        use_rope=config.get('use_rope', True),  # Kept for compatibility
        num_visual_channels=config.get('num_visual_channels', 3),
        latent_size=config.get('latent_size', 32)
    )


if __name__ == "__main__":
    # Test the non-autoregressive decoder
    print("Testing NonAutoregressiveTextDecoder...")

    decoder = NonAutoregressiveTextDecoder(
        vocab_size=50257,
        max_seq_len=64,
        hidden_size=384,
        num_layers=4,
        num_heads=8,
        ffn_size=1536,
        dropout=0.1,
        num_visual_channels=3,
        latent_size=32
    )

    # Test forward pass
    batch_size = 2
    visual_latent = torch.randn(batch_size, 32, 32, 3)

    # Note: input_ids is ignored!
    logits = decoder(visual_latent)

    print(f"Input latent shape: {visual_latent.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: [batch={batch_size}, seq_len=64, vocab=50257]")

    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    print("\n✓ Non-autoregressive decoder test passed!")
    print("Key feature: NO access to input tokens - generates purely from visual latent!")
