"""
Phase 1: Text Decoder
Reconstructs text from 32×32×6 visual latent using cross-attention.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """RoPE positional embeddings"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary embeddings to x"""
    # x: [B, H, L, D]
    # cos, sin: [L, D]
    cos = cos[None, None, :, :]  # [1, 1, L, D]
    sin = sin[None, None, :, :]

    # Split x into first and second half
    x1, x2 = x.chunk(2, dim=-1)

    # Apply rotation
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)


class DecoderLayer(nn.Module):
    """Transformer decoder layer with causal self-attention and cross-attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        # Cross-attention to visual latent
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        visual_memory: torch.Tensor,
        causal_mask: torch.Tensor,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None
    ):
        """
        Args:
            x: [B, L, D] decoder input
            visual_memory: [B, M, D] visual latent memory (M=1024)
            causal_mask: [L, L] causal attention mask
            rope_cos, rope_sin: RoPE embeddings

        Returns:
            [B, L, D] output
        """
        # Self-attention (causal)
        if self.use_rope and rope_cos is not None:
            # Apply RoPE to Q and K
            # Note: nn.MultiheadAttention doesn't expose Q,K,V directly,
            # so we'll skip RoPE for now in decoder (can add custom MHA later)
            attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        else:
            attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)

        x = self.self_attn_norm(x + attn_out)

        # Cross-attention to visual memory
        cross_out, _ = self.cross_attn(x, visual_memory, visual_memory)
        x = self.cross_attn_norm(x + cross_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return x


class TextDecoder(nn.Module):
    """
    Transformer decoder that reconstructs text from visual latent.

    Architecture:
        Visual Latent [B, C, H, W] → Reshape → [B, H*W, C]
        → Project → [B, H*W, hidden_size] (visual memory)

        Decoder: Token Embeddings → Self-Attention (causal)
        → Cross-Attention to visual memory → FFN → Output
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
        use_rope: bool = True,
        num_visual_channels: int = 3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_rope = use_rope

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Positional embeddings (learned, as backup to RoPE)
        if not use_rope:
            self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Visual latent projection (C channels → hidden_size)
        # Support RGB (3), grayscale (1), or legacy (6) channels
        self.num_visual_channels = num_visual_channels
        self.visual_projection = nn.Linear(self.num_visual_channels, hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, ffn_size, dropout, use_rope)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionalEmbedding(hidden_size // num_heads, max_seq_len)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if not self.use_rope:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)

    def forward(
        self,
        visual_latent: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        Args:
            visual_latent: [B, H, W, C] visual latent from encoder (e.g., [B, 32, 32, 6])
            input_ids: [B, L] decoder input token IDs
            attention_mask: [B, L] attention mask (1 for real tokens, 0 for padding)

        Returns:
            logits: [B, L, vocab_size] output logits
        """
        # Handle both [B, H, W, C] and [B, C, H, W] formats
        # Heuristic: if dim1 < dim2, assume [B, C, H, W], else [B, H, W, C]
        if visual_latent.shape[1] < visual_latent.shape[2]:
            # [B, C, H, W] format → convert to [B, H, W, C]
            B, C, H, W = visual_latent.shape
            visual_flat = visual_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            # [B, H, W, C] format (from encoder)
            B, H, W, C = visual_latent.shape
            visual_flat = visual_latent.reshape(B, H * W, C)

        _, L = input_ids.shape

        # 1. Process visual latent into memory
        # visual_flat is now [B, H*W, C]

        # Project to hidden_size: [B, H*W, C] → [B, H*W, hidden_size]
        visual_memory = self.visual_projection(visual_flat)  # [B, 1024, hidden_size]

        # 2. Token embeddings
        x = self.token_embedding(input_ids)  # [B, L, hidden_size]

        # Add positional embeddings (if not using RoPE)
        if not self.use_rope:
            positions = torch.arange(L, device=input_ids.device)
            x = x + self.pos_embedding(positions).unsqueeze(0)

        # 3. Create causal mask
        # Mask shape: [L, L], True means "do not attend"
        causal_mask = torch.triu(torch.ones(L, L, device=input_ids.device), diagonal=1).bool()
        # Convert to additive mask for nn.MultiheadAttention
        causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))

        # 4. RoPE embeddings (if using)
        if self.use_rope:
            rope_cos, rope_sin = self.rope(L, input_ids.device)
        else:
            rope_cos, rope_sin = None, None

        # 5. Decoder layers
        for layer in self.layers:
            x = layer(x, visual_memory, causal_mask, rope_cos, rope_sin)

        # 6. Output projection
        x = self.output_norm(x)
        logits = self.output_projection(x)  # [B, L, vocab_size]

        return logits


def create_decoder(config):
    """Create decoder from config dict"""
    return TextDecoder(
        vocab_size=config.get('vocab_size', 50257),
        max_seq_len=config.get('max_seq_len', 64),
        hidden_size=config.get('hidden_size', 384),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        ffn_size=config.get('ffn_size', 1536),
        dropout=config.get('dropout', 0.1),
        use_rope=config.get('use_rope', True),
        num_visual_channels=config.get('num_visual_channels', 3)
    )


if __name__ == "__main__":
    # Test decoder
    print("Testing TextDecoder...")

    decoder = TextDecoder(
        vocab_size=50257,
        max_seq_len=64,
        hidden_size=384,
        num_layers=4,
        num_heads=8,
        ffn_size=1536,
        dropout=0.1,
        use_rope=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {num_params / 1e6:.2f}M")

    # Test forward pass
    batch_size = 4
    seq_len = 32

    # Visual latent from encoder ([B, H, W, C] format)
    visual_latent = torch.randn(batch_size, 32, 32, 6)

    # Decoder input (teacher forcing)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    with torch.no_grad():
        logits = decoder(visual_latent, input_ids, attention_mask)

    print(f"Input: visual_latent {visual_latent.shape}, input_ids {input_ids.shape}")
    print(f"Output: logits {logits.shape}")
    print(f"\n✓ Decoder test passed!")
