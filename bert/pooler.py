"""
BERT Pooler Module

This module implements the BertPooler class, which creates a fixed-size
sequence representation from the variable-length hidden states.

POOLING STRATEGY:
=================
BERT uses a specific pooling strategy:
- Extract the hidden state of the [CLS] token (first token)
- Apply a linear transformation
- Apply Tanh activation
- Result: Single vector representing the entire sequence

THE [CLS] TOKEN:
================
[CLS] (Classification token) is a special token that BERT adds to the
beginning of every input sequence. Through self-attention across all layers,
the [CLS] token's hidden state aggregates information from the entire sequence.

Why [CLS] works:
- It attends to all tokens via self-attention
- It has no specific word meaning, so it can freely represent the sequence
- It's been trained to be useful for sequence-level tasks

ALTERNATIVE POOLING STRATEGIES:
================================
Different pooling approaches and their use cases:

1. [CLS] Token Pooling (BERT's approach):
   - Use: Sequence classification, sentence pair tasks
   - Pros: Trained specifically for this purpose
   - Cons: Relies on attention to aggregate information

2. Mean Pooling:
   - Average all token representations
   - Use: Sentence embeddings (e.g., Sentence-BERT)
   - Pros: All tokens contribute equally
   - Cons: Dilutes important information

3. Max Pooling:
   - Take maximum value across sequence for each dimension
   - Use: When dominant features are important
   - Pros: Captures salient features
   - Cons: Can be noisy

4. First/Last Token:
   - Use first or last non-special token
   - Use: Simple sequence representation
   - Cons: Limited context

BERT uses [CLS] pooling because the model is pretrained to encode
sequence-level information into this token through Next Sentence Prediction (NSP).

USAGE IN TASKS:
===============
The pooled output is used for:
- Sequence Classification: Binary/multi-class text classification
- Sentence Pair Classification: Natural Language Inference, paraphrase detection
- Next Sentence Prediction: Pretraining task
- Any task requiring a single vector per sequence
"""

import torch
from torch import nn

# Import configuration
try:
    from .config import BertConfig
except ImportError:
    try:
        from bert.config import BertConfig
    except ImportError:
        print("Warning: BertConfig not found.")
        BertConfig = None


class BertPooler(nn.Module):
    """
    Pool the model's output to a fixed-size vector for sequence-level tasks.

    This module extracts the [CLS] token's hidden state from the final layer
    and transforms it into a pooled representation of the entire sequence.

    ARCHITECTURE:
    =============
    [CLS] hidden state (768-dim for BERT-base)
        ↓
    Linear transformation (768 → 768)
        ↓
    Tanh activation
        ↓
    Pooled output (768-dim)

    WHY TANH ACTIVATION:
    ====================
    - Tanh squashes values to range [-1, 1]
    - Provides bounded output (prevents extreme values)
    - Creates a more stable representation for downstream tasks
    - Historically used in BERT's original implementation

    MATHEMATICAL FORMULA:
    =====================
    pooled_output = tanh(W * hidden_states[0] + b)

    Where:
    - hidden_states[0] is the [CLS] token's final hidden state
    - W is the weight matrix (hidden_size × hidden_size)
    - b is the bias vector
    - tanh is the hyperbolic tangent activation

    Args:
        config (BertConfig): Model configuration

    Inputs (in forward):
        hidden_states (torch.Tensor): Output from final encoder layer,
                                      shape (batch_size, sequence_length, hidden_size)

    Outputs:
        pooled_output (torch.Tensor): Pooled sequence representation,
                                      shape (batch_size, hidden_size)

    Example:
        >>> hidden_states = torch.randn(32, 128, 768)  # batch=32, seq=128, hidden=768
        >>> pooler = BertPooler(config)
        >>> pooled = pooler(hidden_states)
        >>> print(pooled.shape)
        torch.Size([32, 768])  # One vector per sequence
    """

    def __init__(self, config):
        """
        Initialize the pooler.

        Args:
            config (BertConfig): Model configuration with hidden_size parameter
        """
        super().__init__()

        # ============================================================
        # Linear transformation
        # ============================================================
        # Projects the [CLS] hidden state to a new representation space
        # Input dimension: hidden_size (e.g., 768)
        # Output dimension: hidden_size (e.g., 768)
        # This is a learnable transformation that's optimized during fine-tuning
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # ============================================================
        # Activation function
        # ============================================================
        # Tanh activation: outputs in range [-1, 1]
        # Formula: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        # Provides smooth, bounded non-linearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool the sequence to a single vector.

        Process:
        1. Extract [CLS] token (first token, index 0)
        2. Apply linear transformation
        3. Apply Tanh activation
        4. Return pooled vector

        Args:
            hidden_states: Final encoder output, shape (batch_size, sequence_length, hidden_size)
                          Example: (32, 128, 768)
                          Where:
                          - 32 is the batch size
                          - 128 is the sequence length
                          - 768 is the hidden size (BERT-base)

        Returns:
            pooled_output: Pooled representation, shape (batch_size, hidden_size)
                          Example: (32, 768)
                          One vector per sequence in the batch

        Note:
            The [CLS] token is always at position 0 because:
            - The tokenizer adds [CLS] at the start: "[CLS] Hello world [SEP]"
            - Position 0 = [CLS] token
            - Position 1 = First actual token ("Hello")
            - Position 2 = Second actual token ("world")
            - Position 3 = [SEP] token
        """
        # ============================================================
        # STEP 1: Extract the [CLS] token's hidden state
        # ============================================================
        # hidden_states shape: (batch_size, sequence_length, hidden_size)
        # We want only the first token for each sequence in the batch
        # Result shape: (batch_size, hidden_size)
        first_token_tensor = hidden_states[:, 0]

        # Explanation of indexing:
        # [:, 0] means:
        # - : → take all batches
        # - 0 → take only the first position (the [CLS] token)
        #
        # Example:
        # If hidden_states.shape = [32, 128, 768]
        # Then first_token_tensor.shape = [32, 768]

        # ============================================================
        # STEP 2: Apply linear transformation
        # ============================================================
        # Transform the [CLS] representation
        # Shape: (batch_size, hidden_size) → (batch_size, hidden_size)
        pooled_output = self.dense(first_token_tensor)

        # ============================================================
        # STEP 3: Apply Tanh activation
        # ============================================================
        # Apply non-linear activation to bound the output
        # Values will be in range [-1, 1]
        pooled_output = self.activation(pooled_output)

        return pooled_output


__all__ = ["BertPooler"]
