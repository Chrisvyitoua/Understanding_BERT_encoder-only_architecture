"""
BERT Encoder Components

This module implements the core transformer encoder components:
- BertLayer: A single transformer layer
- BertEncoder: Stack of transformer layers

TRANSFORMER LAYER ARCHITECTURE:
================================
Each BERT layer follows the standard Transformer encoder architecture:

1. Multi-Head Self-Attention
   ↓
2. Add & Norm (Residual Connection + Layer Normalization)
   ↓
3. Feed-Forward Network
   ↓
4. Add & Norm (Residual Connection + Layer Normalization)

This pattern is repeated for each layer in the stack.

INFORMATION FLOW THROUGH LAYERS:
=================================
Layer 1: Embeddings → Attention (local patterns) → FFN → Layer1 output
Layer 2: Layer1 output → Attention (more abstract) → FFN → Layer2 output
Layer 3: Layer2 output → Attention (even more abstract) → FFN → Layer3 output
...
Layer 12: Layer11 output → Attention (high-level semantics) → FFN → Final output

Early layers learn:
- Syntactic patterns (grammar, POS tags)
- Local dependencies

Later layers learn:
- Semantic relationships
- Long-range dependencies
- Task-specific features

STACKING BENEFITS:
==================
Why stack 12 (or 24) layers?
1. Hierarchical representations: Each layer refines the representation
2. Increased capacity: More layers = more parameters = more learning capacity
3. Abstraction levels: Low-level → High-level understanding
4. Non-linearity: Multiple activation functions compound non-linear transformations

RESIDUAL CONNECTIONS:
=====================
Each sub-layer uses residual connections:
output = LayerNorm(input + Sublayer(input))

Benefits:
- Gradient flow: Gradients can flow directly through the addition
- Training stability: Prevents gradient vanishing in deep networks
- Identity preservation: Network can choose to pass information unchanged
"""

from typing import Optional, Union

import torch
from torch import nn

# Import components
try:
    from ...modeling_layers import GradientCheckpointingLayer
    from ...pytorch_utils import apply_chunking_to_forward
    from ...processing_utils import Unpack
    from ...utils import TransformersKwargs
    from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    from ...cache_utils import Cache
    from .attention import BertAttention
    from .feedforward import BertIntermediate, BertOutput
    from .config import BertConfig
except ImportError:
    print("Warning: Some imports failed. Using fallbacks.")
    GradientCheckpointingLayer = nn.Module
    TransformersKwargs = dict
    Unpack = dict
    Cache = object
    BaseModelOutputWithPastAndCrossAttentions = object

    # Fallback for apply_chunking_to_forward
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
        return forward_fn(*args)

    # Local imports as fallback
    try:
        from .attention import BertAttention
        from .feedforward import BertIntermediate, BertOutput
        from .config import BertConfig
    except ImportError:
        BertAttention = None
        BertIntermediate = None
        BertOutput = None
        BertConfig = None


class BertLayer(GradientCheckpointingLayer):
    """
    A single transformer layer in BERT.

    This implements one complete transformer encoder layer with:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    3. Residual connections and layer normalization

    ARCHITECTURE DIAGRAM:
    =====================
    Input (from previous layer or embeddings)
      ↓
    ┌─────────────────────────────────┐
    │  Multi-Head Self-Attention      │
    └─────────────────────────────────┘
      ↓
    Add & Norm (Residual Connection)
      ↓
    ┌─────────────────────────────────┐
    │  Position-wise Feed-Forward     │
    │  (Expand → Activate → Contract) │
    └─────────────────────────────────┘
      ↓
    Add & Norm (Residual Connection)
      ↓
    Output (to next layer)

    For BERT-base, this layer is stacked 12 times.
    For BERT-large, this layer is stacked 24 times.

    OPTIONAL COMPONENTS:
    ====================
    - Cross-Attention: Used in encoder-decoder models (between self-attention and FFN)
    - Gradient Checkpointing: Trade compute for memory (recompute activations during backward pass)
    """

    def __init__(self, config, layer_idx=None):
        """
        Initialize a BERT transformer layer.

        Args:
            config (BertConfig): Model configuration
            layer_idx (int, optional): Index of this layer in the stack (for caching)
        """
        super().__init__()

        # Configuration for memory optimization
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1  # Sequence length is the 2nd dimension (after batch)

        # ============================================================
        # COMPONENT 1: Multi-Head Self-Attention
        # ============================================================
        # This allows each token to attend to all other tokens in the sequence
        self.attention = BertAttention(
            config,
            is_causal=config.is_decoder,
            layer_idx=layer_idx
        )

        # ============================================================
        # OPTIONAL: Cross-Attention (for decoder mode)
        # ============================================================
        # Only used in encoder-decoder architectures
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        if self.add_cross_attention:
            # Cross-attention requires decoder mode
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            # Cross-attention: decoder attends to encoder outputs
            self.crossattention = BertAttention(
                config,
                is_causal=False,
                layer_idx=layer_idx,
                is_cross_attention=True,
            )

        # ============================================================
        # COMPONENT 2: Position-Wise Feed-Forward Network
        # ============================================================
        # Two-layer FFN with activation in between
        # hidden_size → intermediate_size → hidden_size
        # 768 → 3072 → 768 (for BERT-base)
        self.intermediate = BertIntermediate(config)  # Expansion + Activation
        self.output = BertOutput(config)  # Contraction + Residual + Norm

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
        Forward pass through one transformer layer.

        Process:
        1. Self-attention on input
        2. (Optional) Cross-attention with encoder outputs
        3. Feed-forward network
        4. Return transformed hidden states

        Args:
            hidden_states: Input tensor, shape (batch_size, seq_length, hidden_size)
            attention_mask: Mask for self-attention, shape (batch_size, 1, 1, seq_length)
            encoder_hidden_states: Encoder outputs for cross-attention (decoder only)
            encoder_attention_mask: Mask for cross-attention
            past_key_values: Cached key/value pairs for generation
            cache_position: Position in cache

        Returns:
            layer_output: Transformed tensor, shape (batch_size, seq_length, hidden_size)

        Example:
            Input: [batch=32, seq=128, hidden=768]
            → Self-Attention → [batch=32, seq=128, hidden=768]
            → FFN → [batch=32, seq=128, hidden=768]
        """
        # ============================================================
        # STEP 1: Multi-Head Self-Attention
        # ============================================================
        # Each token attends to all tokens in the sequence
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        attention_output = self_attention_output

        # ============================================================
        # STEP 2: Optional Cross-Attention (decoder mode only)
        # ============================================================
        if self.is_decoder and encoder_hidden_states is not None:
            # Validate that cross-attention is configured
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # Cross-attention: decoder attends to encoder
            cross_attention_output, _ = self.crossattention(
                self_attention_output,
                None,  # attention_mask (use encoder_attention_mask instead)
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output

        # ============================================================
        # STEP 3: Position-Wise Feed-Forward Network
        # ============================================================
        # Apply FFN with optional chunking for memory efficiency
        # Chunking processes the sequence in smaller chunks to reduce memory usage
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )

        return layer_output

    def feed_forward_chunk(self, attention_output):
        """
        Apply feed-forward network to a chunk of the sequence.

        This method is called by apply_chunking_to_forward to process
        the sequence in chunks for memory efficiency.

        Args:
            attention_output: Output from attention, shape (batch_size, chunk_size, hidden_size)

        Returns:
            layer_output: FFN output, shape (batch_size, chunk_size, hidden_size)

        Process:
            attention_output → Intermediate (expand + activate) → Output (contract + residual + norm)
        """
        # Expand: hidden_size → intermediate_size (e.g., 768 → 3072)
        intermediate_output = self.intermediate(attention_output)

        # Contract: intermediate_size → hidden_size (e.g., 3072 → 768)
        # Also applies residual connection and layer normalization
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class BertEncoder(nn.Module):
    """
    Stack of BERT transformer layers (the core encoder).

    This is the heart of BERT - multiple identical transformer layers stacked together.
    Each layer processes the sequence and passes its output to the next layer,
    building increasingly abstract and contextual representations.

    LAYER STACKING:
    ===============
    For BERT-base (12 layers):
    Embeddings → Layer 0 → Layer 1 → ... → Layer 11 → Final representations

    For BERT-large (24 layers):
    Embeddings → Layer 0 → Layer 1 → ... → Layer 23 → Final representations

    REPRESENTATION EVOLUTION:
    =========================
    As information flows through layers, representations evolve:

    Layer 0-2 (Early):
        - Basic syntax and grammar
        - Part-of-speech patterns
        - Simple word relationships

    Layer 3-8 (Middle):
        - Phrase-level understanding
        - Semantic relationships
        - Dependency parsing

    Layer 9-11 (Late):
        - High-level semantics
        - Task-specific features
        - Abstract conceptual understanding

    WHY STACKING WORKS:
    ===================
    1. Compositionality: Complex patterns = compositions of simpler patterns
    2. Hierarchical learning: Low-level features → High-level concepts
    3. Non-linearity: Each layer adds non-linear transformations
    4. Representational power: More layers = more expressive model
    """

    def __init__(self, config):
        """
        Initialize the BERT encoder with multiple layers.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()
        self.config = config

        # ============================================================
        # Create a stack of transformer layers
        # ============================================================
        # Each layer is initialized with its index for tracking and caching
        # For BERT-base: Creates 12 layers
        # For BERT-large: Creates 24 layers
        self.layer = nn.ModuleList([
            BertLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass through all encoder layers.

        The input hidden states (embeddings) are passed sequentially through
        each layer. Each layer transforms the representations, and the output
        of one layer becomes the input to the next.

        Args:
            hidden_states: Input embeddings, shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask, shape (batch_size, 1, 1, seq_length)
            encoder_hidden_states: Encoder outputs for cross-attention (decoder only)
            encoder_attention_mask: Encoder attention mask
            past_key_values: Cached key/value pairs for generation
            use_cache: Whether to return cached key/values
            cache_position: Position in cache

        Returns:
            BaseModelOutputWithPastAndCrossAttentions containing:
                - last_hidden_state: Final layer output, shape (batch_size, seq_length, hidden_size)
                - past_key_values: Updated cache (if use_cache=True)

        Example:
            Input embeddings: [batch=32, seq=128, hidden=768]
            → Layer 0 → [32, 128, 768]
            → Layer 1 → [32, 128, 768]
            ...
            → Layer 11 → [32, 128, 768] (final representations)
        """
        # ============================================================
        # Sequential processing through all layers
        # ============================================================
        for i, layer_module in enumerate(self.layer):
            # Pass hidden states through current layer
            # Each layer returns transformed hidden states
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # Positional arg for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # ============================================================
        # Return final outputs
        # ============================================================
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


__all__ = ["BertLayer", "BertEncoder"]
