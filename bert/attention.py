"""
BERT Attention Mechanisms

This module implements all attention-related components used in BERT, including:
- Multi-head self-attention
- Cross-attention (for encoder-decoder models)
- Attention output projection with residual connections

ATTENTION MECHANISM OVERVIEW:
==============================
Attention allows the model to focus on different parts of the input when processing each token.
The key insight: "Not all tokens are equally relevant for understanding the current token."

SCALED DOT-PRODUCT ATTENTION FORMULA:
======================================
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
- Q (Query): "What am I looking for?" - represents the current token
- K (Key): "What do I offer?" - represents all tokens as keys
- V (Value): "What information do I carry?" - the actual information to extract
- d_k: dimension of keys (used for scaling to prevent softmax saturation)

MULTI-HEAD ATTENTION:
=====================
Instead of computing attention once, BERT computes it multiple times in parallel
with different learned projections (called "heads"). This allows the model to:
- Attend to different aspects simultaneously
- Learn diverse attention patterns
- Capture both local and long-range dependencies

For BERT-base:
- 12 attention heads
- Each head operates on 64 dimensions (768 / 12)
- Total computation is the same as single-head attention on 768 dims

EXAMPLE ATTENTION PATTERN:
==========================
Input: "The cat sat on the mat"
Token "cat" might attend to:
- Head 1: "The" (determiner relationship)
- Head 2: "sat" (subject-verb relationship)
- Head 3: "mat" (semantic similarity)
- Head 4: "cat" (self-attention for context)
"""

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

# Import utilities
try:
    from ...processing_utils import Unpack
    from ...utils import TransformersKwargs
    from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ...cache_utils import Cache, EncoderDecoderCache
    from .config import BertConfig
except ImportError:
    print("Warning: Some imports failed. Running in limited mode.")
    TransformersKwargs = dict
    Unpack = dict
    ALL_ATTENTION_FUNCTIONS = {}
    Cache = object
    EncoderDecoderCache = object
    BertConfig = None


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    Standard scaled dot-product attention mechanism (eager implementation).

    This is the core attention computation that implements:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    The "eager" name distinguishes this from optimized implementations like Flash Attention.

    Args:
        module (nn.Module): The attention module (for accessing training mode)
        query (torch.Tensor): Query tensor, shape (batch_size, num_heads, seq_length, head_dim)
        key (torch.Tensor): Key tensor, shape (batch_size, num_heads, seq_length, head_dim)
        value (torch.Tensor): Value tensor, shape (batch_size, num_heads, seq_length, head_dim)
        attention_mask (torch.Tensor, optional): Mask to prevent attention to certain positions
        scaling (float, optional): Scale factor, typically 1/sqrt(head_dim)
        dropout (float): Dropout probability for attention weights

    Returns:
        attn_output (torch.Tensor): Attention output, shape (batch_size, seq_length, num_heads, head_dim)
        attn_weights (torch.Tensor): Attention probabilities for visualization/analysis

    Process:
        1. Compute attention scores: Q @ K^T
        2. Scale by sqrt(d_k) to prevent gradient vanishing
        3. Apply attention mask (set unwanted positions to -inf)
        4. Apply softmax to get probabilities
        5. Apply dropout for regularization
        6. Multiply by values: attention_probs @ V
    """
    # Calculate scaling factor if not provided
    if scaling is None:
        scaling = query.size(-1) ** -0.5  # 1/sqrt(d_k) for numerical stability

    # ============================================================
    # STEP 1: Compute raw attention scores
    # ============================================================
    # Matrix multiplication: Q @ K^T
    # Result shape: (batch_size, num_heads, seq_length, seq_length)
    # attn_weights[i, j] = how much token i attends to token j
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    # ============================================================
    # STEP 2: Apply attention mask (if provided)
    # ============================================================
    # Attention mask typically contains:
    # - 0 for positions that can be attended to
    # - -inf for positions that should be ignored (padding, future tokens in causal models)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        # Adding -inf makes softmax output 0 for those positions
        attn_weights = attn_weights + attention_mask

    # ============================================================
    # STEP 3: Apply softmax to get attention probabilities
    # ============================================================
    # Convert scores to probabilities (sum to 1 for each query token)
    # Shape remains: (batch_size, num_heads, seq_length, seq_length)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # ============================================================
    # STEP 4: Apply dropout to attention weights
    # ============================================================
    # Dropout on attention weights provides regularization
    # Prevents the model from relying too heavily on specific attention patterns
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # ============================================================
    # STEP 5: Apply attention weights to values
    # ============================================================
    # Weighted sum of values based on attention probabilities
    # attn_output shape: (batch_size, num_heads, seq_length, head_dim)
    attn_output = torch.matmul(attn_weights, value)

    # ============================================================
    # STEP 6: Transpose back to (batch_size, seq_length, num_heads, head_dim)
    # ============================================================
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BertSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for BERT.

    Self-attention means each token attends to all tokens in the same sequence,
    including itself. This is different from cross-attention where queries come
    from one sequence and keys/values from another.

    ARCHITECTURE:
    =============
    For BERT-base (hidden_size=768, num_heads=12):
    - Total hidden size: 768
    - Number of heads: 12
    - Size per head: 768 / 12 = 64 dimensions

    Each head learns to attend to different patterns:
    - Syntax (grammar relationships)
    - Semantics (meaning relationships)
    - Positional (nearby vs distant tokens)
    - Domain-specific patterns

    PROCESS:
    ========
    1. Linear projections: hidden_states → Q, K, V (each 768-dim)
    2. Reshape and split: 768-dim → 12 heads × 64-dim
    3. Compute attention: each head independently computes attention
    4. Concatenate: 12 × 64-dim → 768-dim
    5. Output projection: done in BertSelfOutput

    COMPUTATIONAL COMPLEXITY:
    =========================
    Time: O(n^2 * d) where n=sequence length, d=hidden dimension
    Space: O(n^2) for storing attention weights
    This is why long sequences are expensive for transformers!
    """

    def __init__(self, config, is_causal=False, layer_idx=None):
        """
        Initialize multi-head self-attention.

        Args:
            config (BertConfig): Model configuration
            is_causal (bool): If True, apply causal masking (for autoregressive models)
            layer_idx (int, optional): Layer index for caching
        """
        super().__init__()

        # Validation: hidden_size must be divisible by num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config

        # Multi-head attention configuration
        self.num_attention_heads = config.num_attention_heads  # e.g., 12 for BERT-base
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # e.g., 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # Should equal hidden_size
        self.scaling = self.attention_head_size**-0.5  # 1/sqrt(64) ≈ 0.125 for BERT-base

        # ============================================================
        # Linear projections for Query, Key, Value
        # ============================================================
        # Each projects from hidden_size to all_head_size (which equals hidden_size)
        # The split into multiple heads happens later via reshape
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout for attention probabilities
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Configuration flags
        self.is_decoder = config.is_decoder
        self.is_causal = is_causal  # For causal (autoregressive) masking
        self.layer_idx = layer_idx  # For layer-wise caching

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        """
        Forward pass for multi-head self-attention.

        Args:
            hidden_states: Input tensor, shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask, shape (batch_size, 1, 1, seq_length)
            past_key_values: Cached key/value pairs for fast generation
            cache_position: Position in cache for generation

        Returns:
            attn_output: Attention output, shape (batch_size, seq_length, hidden_size)
            attn_weights: Attention probabilities (for visualization)
        """
        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_length)
        hidden_shape = (*input_shape, -1, self.attention_head_size)  # For reshaping to multi-head

        # ============================================================
        # STEP 1: Project to Q, K, V and reshape for multi-head attention
        # ============================================================
        # Shape: (batch_size, seq_length, hidden_size) → (batch_size, seq_length, num_heads, head_dim)
        # Then transpose to: (batch_size, num_heads, seq_length, head_dim)
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        # ============================================================
        # STEP 2: Handle caching for fast generation (if applicable)
        # ============================================================
        if past_key_values is not None:
            # For decoder-only BERT or generation tasks
            current_past_key_values = past_key_values
            if isinstance(past_key_values, EncoderDecoderCache):
                current_past_key_values = past_key_values.self_attention_cache

            # Update cache with current keys and values
            key_layer, value_layer = current_past_key_values.update(
                key_layer,
                value_layer,
                self.layer_idx,
                {"cache_position": cache_position},
            )

        # ============================================================
        # STEP 3: Select attention implementation (eager vs optimized)
        # ============================================================
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            # Use optimized attention (e.g., Flash Attention, Memory Efficient Attention)
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # ============================================================
        # STEP 4: Compute attention
        # ============================================================
        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )

        # ============================================================
        # STEP 5: Reshape output back to (batch_size, seq_length, hidden_size)
        # ============================================================
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        return attn_output, attn_weights


class BertCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.

    Cross-attention is used in encoder-decoder architectures where:
    - Queries (Q) come from the decoder
    - Keys (K) and Values (V) come from the encoder

    This allows the decoder to attend to the encoder's representations.

    Example use case: Machine Translation
    - Encoder processes source language sentence
    - Decoder generates target language, attending to source via cross-attention

    Note: Standard BERT (encoder-only) doesn't use cross-attention.
    This is included for completeness and decoder variants.
    """

    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        # Query projection (from decoder hidden states)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # Key and Value projections (from encoder hidden states)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        """
        Forward pass for cross-attention.

        Args:
            hidden_states: Decoder states (for queries)
            encoder_hidden_states: Encoder states (for keys and values)
            attention_mask: Mask for encoder sequence
            past_key_values: Cached encoder key/values

        Returns:
            attn_output: Cross-attention output
            attn_weights: Attention weights
        """
        # Determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]  # Decoder (target) sequence
        src_len = encoder_hidden_states.shape[1]  # Encoder (source) sequence

        q_input_shape = (bsz, tgt_len, -1, self.attention_head_size)
        kv_input_shape = (bsz, src_len, -1, self.attention_head_size)

        # Project queries from decoder hidden states
        query_layer = self.query(hidden_states).view(*q_input_shape).transpose(1, 2)

        # Check if keys/values are cached
        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False

        if past_key_values is not None and is_updated:
            # Reuse cached keys and values (encoder doesn't change during decoding)
            key_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
            value_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].values
        else:
            # Compute keys and values from encoder hidden states
            key_layer = self.key(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_layer = self.value(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

            if past_key_values is not None:
                # Cache keys and values for reuse
                key_layer, value_layer = past_key_values.cross_attention_cache.update(
                    key_layer, value_layer, self.layer_idx
                )
                past_key_values.is_updated[self.layer_idx] = True

        # Select attention implementation
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Compute cross-attention
        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )

        # Reshape output
        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()

        return attn_output, attn_weights


class BertSelfOutput(nn.Module):
    """
    Output projection and residual connection for self-attention.

    This implements the "Add & Norm" pattern from the Transformer architecture:
    output = LayerNorm(input + Dropout(Dense(attention_output)))

    WHY RESIDUAL CONNECTIONS:
    =========================
    Residual connections (skip connections) help with:
    1. Gradient flow: Gradients can flow directly through the addition
    2. Training stability: Prevents gradient vanishing in deep networks
    3. Identity mapping: Network can learn to pass information unchanged if needed

    The pattern is:
    - Take attention output
    - Apply dense projection
    - Apply dropout
    - Add back the original input (residual)
    - Apply layer normalization
    """

    def __init__(self, config):
        super().__init__()

        # Dense projection back to hidden_size
        # Even though input is already hidden_size, this learnable projection
        # allows the model to transform the multi-head attention output
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer normalization for training stability
        # Normalizes across the hidden_size dimension
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply output projection and residual connection.

        Args:
            hidden_states: Attention output, shape (batch_size, seq_length, hidden_size)
            input_tensor: Original input before attention, shape (batch_size, seq_length, hidden_size)

        Returns:
            output: Transformed output with residual, shape (batch_size, seq_length, hidden_size)
        """
        # Project the attention output
        hidden_states = self.dense(hidden_states)

        # Apply dropout for regularization
        hidden_states = self.dropout(hidden_states)

        # Add residual connection and apply layer normalization
        # This is the "Add & Norm" step from the Transformer paper
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertAttention(nn.Module):
    """
    Complete attention module combining self/cross-attention with output projection.

    This is a wrapper that combines:
    1. BertSelfAttention or BertCrossAttention
    2. BertSelfOutput (projection + residual + normalization)

    This module represents one complete attention sub-layer in a BERT layer.
    """

    def __init__(self, config, is_causal=False, layer_idx=None, is_cross_attention=False):
        """
        Initialize complete attention module.

        Args:
            config: Model configuration
            is_causal: Whether to use causal masking
            layer_idx: Layer index for caching
            is_cross_attention: Whether this is cross-attention (vs self-attention)
        """
        super().__init__()

        self.is_cross_attention = is_cross_attention

        # Select attention type: cross-attention or self-attention
        attention_class = BertCrossAttention if is_cross_attention else BertSelfAttention
        self.self = attention_class(config, is_causal=is_causal, layer_idx=layer_idx)

        # Output projection with residual connection
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        """
        Forward pass for complete attention module.

        Args:
            hidden_states: Input tensor
            attention_mask: Self-attention mask
            encoder_hidden_states: Encoder outputs (for cross-attention)
            encoder_attention_mask: Encoder attention mask
            past_key_values: Cached key/values
            cache_position: Cache position

        Returns:
            attention_output: Final attention output with residual and normalization
            attn_weights: Attention probabilities
        """
        # Use appropriate mask based on attention type
        attention_mask = attention_mask if not self.is_cross_attention else encoder_attention_mask

        # Compute attention
        attention_output, attn_weights = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # Apply output projection and residual connection
        attention_output = self.output(attention_output, hidden_states)

        return attention_output, attn_weights


__all__ = [
    "eager_attention_forward",
    "BertSelfAttention",
    "BertCrossAttention",
    "BertSelfOutput",
    "BertAttention",
]
