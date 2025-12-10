"""
BERT Embeddings Module

This module implements the BertEmbeddings class, which creates the initial
embeddings for BERT by combining three types of learned embeddings.

BERT EMBEDDING COMPOSITION:
============================
Unlike vanilla Transformers that use sinusoidal position encodings,
BERT uses LEARNED embeddings for all three components:

1. Word Embeddings (Token Embeddings):
   - Maps each token ID to a dense vector
   - Shape: (vocab_size, hidden_size)
   - Example: (30522, 768) for BERT-base

2. Position Embeddings:
   - Encodes the position of each token in the sequence
   - Shape: (max_position_embeddings, hidden_size)
   - Example: (512, 768) for standard BERT
   - Position 0 gets embedding[0], position 1 gets embedding[1], etc.

3. Token Type Embeddings (Segment Embeddings):
   - Distinguishes between different segments/sentences in the input
   - Shape: (type_vocab_size, hidden_size)
   - Example: (2, 768) - one for sentence A (id=0), one for sentence B (id=1)
   - Used for tasks like Next Sentence Prediction (NSP)

FINAL EMBEDDING FORMULA:
========================
For each token at position i:
    embedding[i] = word_emb[token_id[i]] + position_emb[i] + token_type_emb[segment_id[i]]
    embedding[i] = LayerNorm(embedding[i])
    embedding[i] = Dropout(embedding[i])

EXAMPLE:
========
Input: "[CLS] Hello world [SEP]"
Token IDs: [2, 7592, 2088, 3]
Position IDs: [0, 1, 2, 3]
Token Type IDs: [0, 0, 0, 0]  (all from sentence A)

For token "Hello" at position 1:
- Word embedding: word_embeddings[7592] → vector of size 768
- Position embedding: position_embeddings[1] → vector of size 768
- Token type embedding: token_type_embeddings[0] → vector of size 768
- Final: LayerNorm(Dropout(sum of all three))
"""

from typing import Optional

import torch
from torch import nn

# Import configuration
try:
    from .config import BertConfig
except ImportError:
    # Fallback for different import structures
    try:
        from bert.config import BertConfig
    except ImportError:
        print("Warning: BertConfig not found. Using standalone mode.")
        BertConfig = None


class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.

    This is the first layer in the BERT model, converting input token IDs into
    dense vector representations that can be processed by the transformer layers.

    BERT uses three types of embeddings that are summed together:
    1. Word embeddings: Learned representations for each token in the vocabulary
    2. Position embeddings: Learned representations for each position (0 to max_position_embeddings-1)
    3. Token type embeddings: Distinguish between sentence A and sentence B in sentence pairs

    The final embedding is: LayerNorm(word_emb + position_emb + token_type_emb)

    Args:
        config (BertConfig): Model configuration with architecture parameters

    Inputs (in forward):
        input_ids (torch.LongTensor): Token IDs, shape (batch_size, seq_length)
        token_type_ids (torch.LongTensor, optional): Segment IDs, shape (batch_size, seq_length)
        position_ids (torch.LongTensor, optional): Position IDs, shape (batch_size, seq_length)
        inputs_embeds (torch.FloatTensor, optional): Pre-computed embeddings instead of input_ids
        past_key_values_length (int): Length of past key values (for generation)

    Outputs:
        embeddings (torch.FloatTensor): Combined embeddings, shape (batch_size, seq_length, hidden_size)
    """

    def __init__(self, config):
        super().__init__()

        # ============================================================
        # EMBEDDING LAYERS
        # ============================================================

        # Word embeddings: Maps token IDs to dense vectors
        # Shape: (vocab_size, hidden_size), e.g., (30522, 768) for BERT-base
        # padding_idx: The padding token ID will have a zero embedding (not learned)
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

        # Position embeddings: Encodes the position of each token in the sequence
        # Shape: (max_position_embeddings, hidden_size), e.g., (512, 768)
        # Unlike Transformers' sinusoidal encodings, BERT uses LEARNED position embeddings
        # This allows the model to learn position-dependent patterns specific to the task
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        # Token type embeddings: Differentiates between sentence A (id=0) and sentence B (id=1)
        # Shape: (type_vocab_size, hidden_size), typically (2, 768)
        # Used for tasks like question answering where we have two input sequences
        # Example: [CLS] question [SEP] context [SEP]
        #          └── type 0 ──┘      └─ type 1 ─┘
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )

        # ============================================================
        # NORMALIZATION AND REGULARIZATION
        # ============================================================

        # Layer normalization for stability and faster training
        # Normalizes across the hidden_size dimension
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ============================================================
        # POSITION IDS BUFFER
        # ============================================================

        # Pre-computed position IDs buffer (0, 1, 2, ..., max_position_embeddings-1)
        # Registered as a buffer so it's moved to the correct device but not trained
        # Shape: (1, max_position_embeddings)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )

        # Default token_type_ids buffer (all zeros, meaning all tokens are type 0)
        # This is used when token_type_ids is not provided
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Compute BERT embeddings from input token IDs.

        The process:
        1. Get word embeddings from token IDs
        2. Get position embeddings for each position
        3. Get token type embeddings for segment distinction
        4. Sum all three embeddings element-wise
        5. Apply layer normalization
        6. Apply dropout

        Args:
            input_ids: Token IDs from vocabulary, shape (batch_size, seq_length)
            token_type_ids: Segment IDs (0 or 1), shape (batch_size, seq_length)
            position_ids: Position indices, shape (batch_size, seq_length)
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            past_key_values_length: Offset for position IDs (used in generation)

        Returns:
            embeddings: Combined and normalized embeddings, shape (batch_size, seq_length, hidden_size)
        """
        # Determine input shape from either input_ids or inputs_embeds
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        # ============================================================
        # STEP 1: Prepare position IDs
        # ============================================================
        if position_ids is None:
            # Use the buffered position IDs (0, 1, 2, ..., seq_length-1)
            # Add offset for generation (when we have past_key_values)
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # ============================================================
        # STEP 2: Prepare token type IDs (segment IDs)
        # ============================================================
        if token_type_ids is None:
            # Use the buffered token_type_ids (all zeros) if available
            if hasattr(self, "token_type_ids"):
                # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(
                    buffered_token_type_ids, dim=1, index=position_ids
                )
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                # Create all-zero token_type_ids (all tokens are type 0)
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        # ============================================================
        # STEP 3: Get word embeddings
        # ============================================================
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # ============================================================
        # STEP 4: Get token type embeddings and add to word embeddings
        # ============================================================
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        # ============================================================
        # STEP 5: Get position embeddings and add to current embeddings
        # ============================================================
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        # ============================================================
        # STEP 6: Apply layer normalization for stability
        # ============================================================
        embeddings = self.LayerNorm(embeddings)

        # ============================================================
        # STEP 7: Apply dropout for regularization
        # ============================================================
        embeddings = self.dropout(embeddings)

        return embeddings


__all__ = ["BertEmbeddings"]
