"""
BERT Feed-Forward Network (FFN) Components

This module implements the position-wise feed-forward network used in BERT.
The FFN is applied to each position independently and identically.

FEED-FORWARD NETWORK ARCHITECTURE:
===================================
The FFN consists of two linear transformations with a non-linear activation in between:

FFN(x) = activation(x * W1 + b1) * W2 + b2

For BERT-base:
- Input dimension: 768 (hidden_size)
- Intermediate dimension: 3072 (intermediate_size = 4 * hidden_size)
- Output dimension: 768 (hidden_size)

WHY THE EXPANSION?
==================
The FFN expands to 4x the hidden size before contracting back. This expansion:
1. Increases model capacity - more parameters to learn complex transformations
2. Creates a bottleneck architecture - forces learning of compressed representations
3. Allows non-linear transformations in a higher-dimensional space

ACTIVATION FUNCTION - GELU:
============================
BERT uses GELU (Gaussian Error Linear Unit) instead of ReLU:

GELU(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of the standard normal distribution

GELU vs ReLU:
- ReLU: Hard cutoff at 0 (outputs 0 for negative inputs)
- GELU: Smooth, probabilistic activation
- GELU often performs better in transformer models

POSITION-WISE APPLICATION:
==========================
"Position-wise" means the same FFN is applied independently to each token position.
For sequence [token1, token2, token3]:
- FFN(token1) is computed independently
- FFN(token2) is computed independently
- FFN(token3) is computed independently
- All use the same weights W1, W2

This is different from attention, where tokens interact with each other.

RESIDUAL CONNECTION:
====================
Like attention, the FFN output is combined with the input using a residual connection:
output = LayerNorm(input + Dropout(FFN(input)))
"""

from typing import Optional

import torch
from torch import nn

# Import utilities
try:
    from ...activations import ACT2FN
    from .config import BertConfig
except ImportError:
    print("Warning: Some imports failed. Using fallbacks.")
    ACT2FN = {"gelu": nn.GELU()}
    BertConfig = None


class BertIntermediate(nn.Module):
    """
    First part of the feed-forward network (FFN).

    This layer expands the representation from hidden_size to intermediate_size.
    For BERT-base: 768 → 3072 (4x expansion)

    The expansion allows the model to:
    - Learn more complex transformations
    - Operate in a higher-dimensional space
    - Increase model capacity without adding layers

    Architecture:
        input (768) → Linear → intermediate (3072) → GELU activation → output (3072)

    The next layer (BertOutput) will contract it back to hidden_size.
    """

    def __init__(self, config):
        """
        Initialize the intermediate layer.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()

        # Expand from hidden_size to intermediate_size
        # For BERT-base: 768 → 3072
        # Weight matrix shape: (hidden_size, intermediate_size) = (768, 3072)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # Activation function (GELU for BERT)
        # GELU provides smooth, non-linear transformation
        if isinstance(config.hidden_act, str):
            # If activation is specified as a string (e.g., "gelu")
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # If activation is provided as a callable function
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: expand and activate.

        Process:
        1. Linear projection to intermediate_size
        2. Apply non-linear activation (GELU)

        Args:
            hidden_states: Input tensor, shape (batch_size, seq_length, hidden_size)
                          e.g., (32, 128, 768) for batch=32, seq_len=128

        Returns:
            output: Activated tensor, shape (batch_size, seq_length, intermediate_size)
                   e.g., (32, 128, 3072)

        Example:
            Input: [batch=2, seq=3, hidden=768]
            → Linear(768→3072): [batch=2, seq=3, inter=3072]
            → GELU: [batch=2, seq=3, inter=3072]
        """
        # STEP 1: Project to intermediate_size (expansion)
        # Shape: (batch_size, seq_length, hidden_size) → (batch_size, seq_length, intermediate_size)
        hidden_states = self.dense(hidden_states)

        # STEP 2: Apply non-linear activation (GELU)
        # This introduces non-linearity, allowing the network to learn complex patterns
        # Without activation, stacking linear layers would still be linear!
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Module):
    """
    Second part of the feed-forward network with residual connection.

    This layer contracts the representation back from intermediate_size to hidden_size.
    For BERT-base: 3072 → 768 (contraction to original size)

    This completes the FFN transformation and applies:
    - Linear projection (contraction)
    - Dropout (regularization)
    - Residual connection (gradient flow)
    - Layer normalization (training stability)

    The complete FFN with residual connection:
        output = LayerNorm(input + Dropout(Linear(GELU(Linear(input)))))

    WHY RESIDUAL CONNECTION:
    ========================
    Without residual: output = FFN(input)
    With residual: output = input + FFN(input)

    Benefits:
    1. Gradient flow: Gradients can flow directly through the addition
    2. Identity mapping: If FFN learns zero transformation, output = input
    3. Easier training: Network can learn to refine representations incrementally
    """

    def __init__(self, config):
        """
        Initialize the output layer.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()

        # Contract from intermediate_size back to hidden_size
        # For BERT-base: 3072 → 768
        # Weight matrix shape: (intermediate_size, hidden_size) = (3072, 768)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        # Layer normalization for training stability
        # Normalizes across the hidden_size dimension for each token
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout for regularization
        # Randomly zeros some elements during training to prevent overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: contract, add residual, and normalize.

        Process:
        1. Linear projection back to hidden_size (contraction)
        2. Apply dropout for regularization
        3. Add residual connection (original input)
        4. Apply layer normalization

        Args:
            hidden_states: Output from BertIntermediate, shape (batch_size, seq_length, intermediate_size)
                          e.g., (32, 128, 3072)
            input_tensor: Original input before FFN, shape (batch_size, seq_length, hidden_size)
                         e.g., (32, 128, 768)

        Returns:
            output: Final output with residual and normalization, shape (batch_size, seq_length, hidden_size)
                   e.g., (32, 128, 768)

        Example:
            hidden_states: [batch=2, seq=3, inter=3072] (from BertIntermediate)
            input_tensor: [batch=2, seq=3, hidden=768] (original input)

            → Linear(3072→768): [batch=2, seq=3, hidden=768]
            → Dropout: [batch=2, seq=3, hidden=768]
            → Add residual: [batch=2, seq=3, hidden=768]
            → LayerNorm: [batch=2, seq=3, hidden=768]
        """
        # STEP 1: Project back to hidden_size (contraction)
        # Shape: (batch_size, seq_length, intermediate_size) → (batch_size, seq_length, hidden_size)
        hidden_states = self.dense(hidden_states)

        # STEP 2: Apply dropout for regularization
        # During training, randomly zeros some elements with probability hidden_dropout_prob
        hidden_states = self.dropout(hidden_states)

        # STEP 3 & 4: Add residual connection and apply layer normalization
        # This is the "Add & Norm" step from the Transformer paper
        # Residual: output = input + transformation(input)
        # This allows gradients to flow directly through the addition
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


__all__ = ["BertIntermediate", "BertOutput"]
