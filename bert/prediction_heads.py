"""
BERT Prediction Heads for Pretraining Tasks

This module implements the prediction heads used for BERT's pretraining objectives:
1. Masked Language Modeling (MLM) - predict masked tokens
2. Next Sentence Prediction (NSP) - predict if two sentences are consecutive

BERT PRETRAINING:
=================
BERT is pretrained on two unsupervised tasks before fine-tuning:

TASK 1: Masked Language Modeling (MLM)
---------------------------------------
- Randomly mask 15% of input tokens
- Model predicts the original tokens
- Teaches bidirectional context understanding

Example:
Input:  "The cat [MASK] on the mat"
Target: "sat"

Masking strategy:
- 80% of time: Replace with [MASK]
- 10% of time: Replace with random token
- 10% of time: Keep original (to handle non-masked tokens)

TASK 2: Next Sentence Prediction (NSP)
---------------------------------------
- Given two sentences A and B
- Predict if B actually follows A in the original text
- Teaches sentence-level relationships

Example:
Input:  Sentence A: "The cat sat on the mat."
        Sentence B: "It was very comfortable."
Label:  IsNext (positive)

Input:  Sentence A: "The cat sat on the mat."
        Sentence B: "Paris is the capital of France."
Label:  NotNext (negative)

WHY THESE TASKS:
================
- MLM: Learns word meanings, syntax, and context
- NSP: Learns sentence relationships, discourse, and coherence
- Together: Create rich, bidirectional representations useful for downstream tasks

PREDICTION HEAD ARCHITECTURE:
==============================
Both heads take BERT's output and add task-specific layers:

MLM Head:
hidden_states → Transform → Linear → vocab_size logits

NSP Head:
pooled_output ([CLS] token) → Linear → 2 class logits (IsNext/NotNext)
"""

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


class BertPredictionHeadTransform(nn.Module):
    """
    Transform layer applied before final prediction in MLM head.

    This transformation prepares the hidden states before predicting tokens.
    It helps the model learn better representations for masked token prediction.

    Architecture:
        hidden_states → Linear → Activation (GELU) → LayerNorm → output

    This is similar to a small feed-forward network that refines the
    representation before the final vocabulary projection.
    """

    def __init__(self, config):
        """
        Initialize the prediction head transform.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()

        # Linear transformation (hidden_size → hidden_size)
        # Allows the model to learn a task-specific representation
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Activation function (typically GELU for BERT)
        # Provides non-linearity for better learning
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        # Layer normalization for stability
        # Normalizes the activations before final prediction
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to hidden states.

        Args:
            hidden_states: Input tensor, shape (batch_size, seq_length, hidden_size)

        Returns:
            transformed: Output tensor, shape (batch_size, seq_length, hidden_size)

        Process:
            hidden_states → Dense → Activation → LayerNorm → output
        """
        # Linear transformation
        hidden_states = self.dense(hidden_states)

        # Non-linear activation
        hidden_states = self.transform_act_fn(hidden_states)

        # Layer normalization
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class BertLMPredictionHead(nn.Module):
    """
    Language Modeling prediction head for masked token prediction.

    This head predicts the original token for each masked position by:
    1. Transforming hidden states
    2. Projecting to vocabulary size
    3. Computing logits for each token in the vocabulary

    The output logits are used to compute cross-entropy loss against the
    true masked tokens during pretraining.

    WEIGHT TYING:
    =============
    Note: In the original BERT implementation, the output decoder weights
    are often tied to the input embedding weights. This reduces parameters
    and enforces consistency between input and output representations.

    Output shape: (batch_size, seq_length, vocab_size)
    - For each token position, we get a probability distribution over the vocabulary
    - Only masked positions contribute to the loss during training
    """

    def __init__(self, config):
        """
        Initialize the language modeling prediction head.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()

        # Transform hidden states before prediction
        self.transform = BertPredictionHeadTransform(config)

        # Output projection to vocabulary size
        # Projects from hidden_size (768) to vocab_size (30522 for BERT)
        # This produces logits for every token in the vocabulary
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Additional bias parameter for output (legacy from original BERT)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        Predict tokens for masked positions.

        Args:
            hidden_states: Encoder output, shape (batch_size, seq_length, hidden_size)

        Returns:
            prediction_scores: Logits for each token in vocab,
                              shape (batch_size, seq_length, vocab_size)

        Example:
            Input: [batch=32, seq=128, hidden=768]
            → Transform: [batch=32, seq=128, hidden=768]
            → Decoder: [batch=32, seq=128, vocab=30522]

            For each masked position, we get 30522 scores (one per vocab token)
        """
        # Apply transformation (Linear → Activation → LayerNorm)
        hidden_states = self.transform(hidden_states)

        # Project to vocabulary size to get prediction logits
        hidden_states = self.decoder(hidden_states)

        return hidden_states


class BertOnlyMLMHead(nn.Module):
    """
    Head for Masked Language Modeling only (without NSP).

    Used for models that only perform masked language modeling,
    such as during fine-tuning for text generation or fill-in-the-blank tasks.

    This is a simple wrapper around BertLMPredictionHead for consistency
    with the model architecture.
    """

    def __init__(self, config):
        """
        Initialize the MLM-only head.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()
        # Use the language modeling prediction head
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        Predict masked tokens.

        Args:
            sequence_output: Encoder output, shape (batch_size, seq_length, hidden_size)

        Returns:
            prediction_scores: Token logits, shape (batch_size, seq_length, vocab_size)
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    """
    Head for Next Sentence Prediction only (without MLM).

    This head classifies whether two sentences are consecutive in the original text.

    Output:
    - 2 logits: [logit_for_NotNext, logit_for_IsNext]
    - Apply softmax to get probabilities
    - Class 0: NotNext (sentences are not consecutive)
    - Class 1: IsNext (sentences are consecutive)

    During pretraining:
    - 50% of examples are positive (IsNext)
    - 50% of examples are negative (NotNext - random sentence pair)
    """

    def __init__(self, config):
        """
        Initialize the NSP-only head.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()

        # Binary classification layer
        # Input: pooled_output from [CLS] token (hidden_size)
        # Output: 2 logits (NotNext, IsNext)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        """
        Predict sentence relationship.

        Args:
            pooled_output: Pooled [CLS] representation, shape (batch_size, hidden_size)

        Returns:
            seq_relationship_score: Binary classification logits, shape (batch_size, 2)
                                   [:, 0] = logit for NotNext
                                   [:, 1] = logit for IsNext
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    """
    Combined heads for BERT pretraining (MLM + NSP).

    This module combines both pretraining objectives:
    1. Masked Language Modeling (token-level predictions)
    2. Next Sentence Prediction (sequence-level prediction)

    Used during the pretraining phase of BERT to learn both:
    - Token-level understanding (through MLM)
    - Sentence-level understanding (through NSP)

    Inputs:
    - sequence_output: All token representations from encoder
    - pooled_output: [CLS] token representation

    Outputs:
    - prediction_scores: MLM logits (batch_size, seq_length, vocab_size)
    - seq_relationship_score: NSP logits (batch_size, 2)

    Both outputs are used to compute their respective losses:
    - MLM loss: CrossEntropy between predictions and true masked tokens
    - NSP loss: CrossEntropy between predictions and true sentence relationship
    - Total loss: MLM loss + NSP loss
    """

    def __init__(self, config):
        """
        Initialize combined pretraining heads.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__()

        # Masked Language Modeling head
        # Predicts masked tokens using all token representations
        self.predictions = BertLMPredictionHead(config)

        # Next Sentence Prediction head
        # Predicts sentence relationship using [CLS] token
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        """
        Compute predictions for both pretraining tasks.

        Args:
            sequence_output: All token representations, shape (batch_size, seq_length, hidden_size)
            pooled_output: [CLS] representation, shape (batch_size, hidden_size)

        Returns:
            prediction_scores: MLM logits, shape (batch_size, seq_length, vocab_size)
            seq_relationship_score: NSP logits, shape (batch_size, 2)

        Example:
            sequence_output: [batch=32, seq=128, hidden=768]
            pooled_output: [batch=32, hidden=768]

            Returns:
            - prediction_scores: [batch=32, seq=128, vocab=30522]
            - seq_relationship_score: [batch=32, 2]
        """
        # MLM: Predict masked tokens
        prediction_scores = self.predictions(sequence_output)

        # NSP: Predict sentence relationship
        seq_relationship_score = self.seq_relationship(pooled_output)

        return prediction_scores, seq_relationship_score


__all__ = [
    "BertPredictionHeadTransform",
    "BertLMPredictionHead",
    "BertOnlyMLMHead",
    "BertOnlyNSPHead",
    "BertPreTrainingHeads",
]
