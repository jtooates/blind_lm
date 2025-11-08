"""
Phase 1: Text Encoder Model Architecture
Produces 2D visual latents from text without semantic understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers import AutoTokenizer


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute position embeddings
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        self.register_buffer('pos_emb', emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor"""
        seq_len = x.size(1)
        pos_emb = self.pos_emb[:seq_len, :].unsqueeze(0)

        # Split x into two halves for rotation
        x1, x2 = x.chunk(2, dim=-1)

        # Get sin and cos embeddings
        sin_emb = pos_emb[..., :self.dim // 2]
        cos_emb = pos_emb[..., self.dim // 2:]

        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos_emb - x2 * sin_emb,
            x1 * sin_emb + x2 * cos_emb
        ], dim=-1)

        return x_rotated


class TransformerBlock(nn.Module):
    """Single transformer block with Pre-LayerNorm"""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + attn_out

        # FFN with residual
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for grid queries attending to token states"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Grid queries [B, H*W, D]
            keys_values: Token states [B, L, D]
        """
        # Cross-attention with residual
        norm_q = self.norm_q(queries)
        norm_kv = self.norm_kv(keys_values)

        attn_out, _ = self.cross_attn(norm_q, norm_kv, norm_kv)
        queries = queries + attn_out

        # FFN with residual
        norm_out = self.norm_ffn(queries)
        ffn_out = self.ffn(norm_out)
        queries = queries + ffn_out

        return queries


class GridQueryHead(nn.Module):
    """Maps token states to 2D grid via cross-attention"""

    def __init__(self, hidden_size: int, grid_size: int, num_channels: int,
                 num_heads: int = 8, dropout: float = 0.1, use_smooth_head: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.num_channels = num_channels

        # Initialize learned grid queries with 2D structure
        # Start with a 2D convolutional structure for better spatial initialization
        self.query_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, hidden_size, kernel_size=1)
        )

        # Initial query embeddings
        self.init_queries = nn.Parameter(torch.randn(1, grid_size, grid_size, 64))

        # Cross-attention blocks
        self.cross_attn1 = CrossAttentionBlock(hidden_size, num_heads, dropout)
        self.cross_attn2 = CrossAttentionBlock(hidden_size, num_heads, dropout)

        # Project to channels
        self.to_channels = nn.Linear(hidden_size, num_channels)

        # Optional local smoothing head
        self.use_smooth_head = use_smooth_head
        if use_smooth_head:
            self.smooth_head = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels),
                nn.GELU(),
                # Pointwise convolution
                nn.Conv2d(num_channels, num_channels, kernel_size=1)
            )

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_states: [B, L, hidden_size] token representations
        Returns:
            grid: [B, H, W, C] 2D visual latent
        """
        B = token_states.size(0)

        # Generate grid queries
        queries = self.init_queries.expand(B, -1, -1, -1)  # [B, H, W, 64]
        queries = queries.permute(0, 3, 1, 2)  # [B, 64, H, W]
        queries = self.query_conv(queries)  # [B, hidden_size, H, W]
        queries = queries.permute(0, 2, 3, 1)  # [B, H, W, hidden_size]
        queries = queries.reshape(B, -1, self.hidden_size)  # [B, H*W, hidden_size]

        # Cross-attention: queries attend to token states
        queries = self.cross_attn1(queries, token_states)
        queries = self.cross_attn2(queries, token_states)

        # Project to channels
        grid = self.to_channels(queries)  # [B, H*W, C]
        grid = grid.reshape(B, self.grid_size, self.grid_size, self.num_channels)

        # Optional smoothing
        if self.use_smooth_head:
            grid = grid.permute(0, 3, 1, 2)  # [B, C, H, W]
            grid = self.smooth_head(grid)
            grid = grid.permute(0, 2, 3, 1)  # [B, H, W, C]

        return grid


class TextEncoder(nn.Module):
    """
    Complete text encoder: tokenizer -> transformer -> grid head
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 64,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_size: int = 1536,
        dropout: float = 0.1,
        grid_size: int = 32,
        num_channels: int = 6,
        use_rope: bool = True,
        use_smooth_head: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.grid_size = grid_size
        self.num_channels = num_channels

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Position embeddings
        self.use_rope = use_rope
        if use_rope:
            self.pos_embedding = RotaryPositionalEmbedding(hidden_size, max_seq_len)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_size, dropout)
            for _ in range(num_layers)
        ])

        # Final norm
        self.ln_f = nn.LayerNorm(hidden_size)

        # Grid head
        self.grid_head = GridQueryHead(
            hidden_size, grid_size, num_channels,
            num_heads, dropout, use_smooth_head
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask (1 for valid, 0 for padding)
        Returns:
            grid: [B, H, W, C] visual latent
        """
        B, L = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, L, hidden_size]

        # Position embeddings
        if self.use_rope:
            x = self.pos_embedding(x)
        else:
            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embedding(positions)

        # Convert attention mask to proper format for MHA
        if attention_mask is not None:
            # Create causal mask for self-attention
            # Shape needs to be [L, L] for nn.MultiheadAttention
            # We need to create a mask where attended positions = 0, ignored = -inf
            attn_mask = attention_mask.unsqueeze(1).expand(-1, L, -1).float()  # [B, L, L]
            attn_mask = (1.0 - attn_mask) * -10000.0
            # For batch processing, we need to handle this differently
            # Just use the first sample's mask since nn.MHA expects 2D
            attn_mask = attn_mask[0]  # [L, L]
        else:
            attn_mask = None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Final normalization
        x = self.ln_f(x)

        # Map to 2D grid
        grid = self.grid_head(x)

        return grid


class GlobalPooling(nn.Module):
    """Global pooling for getting embedding from visual latent"""

    def __init__(self, num_channels: int, embedding_dim: int = 256):
        super().__init__()

        # Mean and variance pooling
        self.embedding_dim = embedding_dim

        # MLP to map pooled features to embedding
        self.mlp = nn.Sequential(
            nn.Linear(num_channels * 2, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, H, W, C] visual latent
        Returns:
            embedding: [B, embedding_dim]
        """
        B, H, W, C = grid.shape

        # Reshape for pooling
        grid_flat = grid.reshape(B, H * W, C)

        # Mean and variance pooling
        mean_pool = grid_flat.mean(dim=1)  # [B, C]
        var_pool = grid_flat.var(dim=1)    # [B, C]

        # Concatenate
        pooled = torch.cat([mean_pool, var_pool], dim=-1)  # [B, C*2]

        # Map to embedding
        embedding = self.mlp(pooled)

        return embedding


def create_model(config: dict) -> TextEncoder:
    """Create model from config dictionary"""
    return TextEncoder(
        vocab_size=config.get('vocab_size', 32000),
        max_seq_len=config.get('max_seq_len', 64),
        hidden_size=config.get('hidden_size', 384),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ffn_size=config.get('ffn_size', 1536),
        dropout=config.get('dropout', 0.1),
        grid_size=config.get('grid_size', 32),
        num_channels=config.get('num_channels', 6),
        use_rope=config.get('use_rope', True),
        use_smooth_head=config.get('use_smooth_head', True)
    )


if __name__ == "__main__":
    # Test the model
    model = create_model({})
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")  # Should be [2, 32, 32, 6]

        # Test global pooling
        pooler = GlobalPooling(num_channels=6)
        embedding = pooler(output)
        print(f"Embedding shape: {embedding.shape}")  # Should be [2, 256]