"""
BERT Base Model Components

This module contains the core BERT model implementation:
- BertPreTrainedModel: Base class with weight initialization and utilities
- BertModel: The main BERT model architecture

BERT MODEL ARCHITECTURE:
========================
The complete BERT model consists of three main components:

1. Embeddings Layer:
   - Converts input token IDs to dense vectors
   - Combines word, position, and token type embeddings

2. Encoder Stack:
   - Multiple transformer layers (12 for base, 24 for large)
   - Each layer has self-attention and feed-forward sub-layers

3. Pooler (optional):
   - Pools the [CLS] token representation
   - Used for sequence-level tasks

BERT VARIANTS:
==============
The BertModel can be configured for different use cases:
- Encoder-only (standard BERT): For classification, NER, etc.
- With pooling: For sentence classification
- Without pooling: For token classification, QA
- Decoder mode: For generation tasks (less common)

MODEL INITIALIZATION:
=====================
The weights are initialized using a truncated normal distribution
with a small standard deviation (typically 0.02). This initialization:
- Prevents saturation of activation functions
- Ensures gradients are well-behaved initially
- Follows best practices from BERT paper

EDUCATIONAL NOTE:
=================
This is a simplified, educational version focusing on clarity.
A production version would include additional features like:
- Gradient checkpointing for memory efficiency
- Different attention implementations (Flash, SDPA)
- Extended output formats
- Caching mechanisms for generation
"""

from typing import Optional

import torch
from torch import nn

# Import model components
try:
    from ...modeling_utils import PreTrainedModel
    from ...utils import logging
    from .config import BertConfig
    from .embeddings import BertEmbeddings
    from .encoder import BertEncoder
    from .pooler import BertPooler
except ImportError:
    print("Warning: Some imports failed. Using standalone mode.")
    PreTrainedModel = nn.Module
    logging = None
    try:
        from .config import BertConfig
        from .embeddings import BertEmbeddings
        from .encoder import BertEncoder
        from .pooler import BertPooler
    except ImportError:
        BertConfig = None
        BertEmbeddings = None
        BertEncoder = None
        BertPooler = None


logger = logging.get_logger(__name__) if logging else None


class BertPreTrainedModel(PreTrainedModel):
    """
    Base class for all BERT models.

    This class provides:
    - Weight initialization functionality
    - Common configuration
    - Utility methods for model management

    All BERT models inherit from this class to ensure consistent
    initialization and behavior across different task-specific models.

    WEIGHT INITIALIZATION:
    ======================
    Proper weight initialization is crucial for training deep networks.
    BERT uses these initialization strategies:

    1. Linear layers and Embeddings:
       - Normal distribution with mean=0, std=0.02
       - Small std prevents saturation of activations
       - Truncated to avoid extreme values

    2. Layer Normalization:
       - Bias: initialized to 0
       - Weight: initialized to 1
       - These are the standard LayerNorm initialization values

    3. Biases in Linear layers:
       - Initialized to 0
       - Allows the model to start with no bias
    """

    # Configuration class for this model type
    config_class = BertConfig

    # Prefix for model weights (used in loading/saving)
    base_model_prefix = "bert"

    def __init__(self, config):
        """
        Initialize the pre-trained model.

        Args:
            config (BertConfig): Model configuration
        """
        super().__init__(config)
        self.config = config

    def _init_weights(self, module):
        """
        Initialize the weights of a module.

        This method is called for each module in the model to set up
        initial weights. Proper initialization is critical for:
        - Training stability
        - Convergence speed
        - Final model performance

        Args:
            module (nn.Module): The module to initialize

        Weight initialization strategies:
        - Linear: Normal(0, 0.02)
        - Embedding: Normal(0, 0.02)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights
            # Normal distribution with small standard deviation
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # Initialize bias to zero if present
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights
            # Same strategy as linear layers
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # If padding_idx is specified, zero out that embedding
            # This ensures padding tokens have no contribution
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization
            # Bias to zero, weight to one (standard practice)
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BertModel(BertPreTrainedModel):
    """
    The main BERT model (encoder-only, bidirectional transformer).

    This is the core BERT architecture that can be used for various NLP tasks
    by adding task-specific heads on top.

    ARCHITECTURE:
    =============
    Input IDs → Embeddings → Encoder Stack → Output Hidden States
                                          ↓
                                     Pooler (optional) → Pooled Output

    Components:
    1. BertEmbeddings: Convert tokens to embeddings
    2. BertEncoder: Stack of transformer layers
    3. BertPooler: Pool [CLS] token (optional)

    INPUTS:
    =======
    - input_ids: Token IDs from vocabulary, shape (batch_size, seq_length)
    - attention_mask: Mask for padding tokens, shape (batch_size, seq_length)
    - token_type_ids: Segment IDs for sentence pairs, shape (batch_size, seq_length)

    OUTPUTS:
    ========
    - last_hidden_state: Final layer outputs, shape (batch_size, seq_length, hidden_size)
    - pooler_output: Pooled [CLS] representation, shape (batch_size, hidden_size)

    USAGE EXAMPLES:
    ===============
    1. Sequence Classification:
       - Use pooler_output
       - Add linear classifier on top

    2. Token Classification (NER, POS tagging):
       - Use last_hidden_state
       - Add linear classifier for each token

    3. Question Answering:
       - Use last_hidden_state
       - Add span prediction heads

    4. Masked Language Modeling:
       - Use last_hidden_state
       - Add MLM prediction head
    """

    def __init__(self, config, add_pooling_layer=True):
        """
        Initialize the BERT model.

        Args:
            config (BertConfig): Model configuration
            add_pooling_layer (bool): Whether to add the pooling layer.
                                     Set to False for token-level tasks that don't need pooling.
        """
        super().__init__(config)
        self.config = config

        # ============================================================
        # COMPONENT 1: Embeddings
        # ============================================================
        # Converts input token IDs to dense embedding vectors
        # Combines word + position + token type embeddings
        self.embeddings = BertEmbeddings(config)

        # ============================================================
        # COMPONENT 2: Encoder Stack
        # ============================================================
        # Stack of transformer layers (12 for base, 24 for large)
        # Each layer refines the representations
        self.encoder = BertEncoder(config)

        # ============================================================
        # COMPONENT 3: Pooler (optional)
        # ============================================================
        # Pools the [CLS] token for sequence-level tasks
        # Set to None if not needed (e.g., for token classification)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights using the initialization strategy
        self.post_init()

    def post_init(self):
        """
        Post-initialization hook.

        Called after model initialization to:
        - Initialize all weights
        - Tie weights if needed (e.g., input/output embeddings)
        - Perform any other setup
        """
        # Apply weight initialization to all modules
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        """
        Get the input embedding layer.

        Returns:
            nn.Embedding: The word embedding layer

        Used for:
        - Inspecting embeddings
        - Modifying vocabulary
        - Weight tying
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Set the input embedding layer.

        Args:
            value (nn.Embedding): New embedding layer

        Used for:
        - Replacing embeddings
        - Weight tying
        """
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass through the BERT model.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
                           1 for real tokens, 0 for padding
            token_type_ids: Segment IDs, shape (batch_size, seq_length)
                           0 for sentence A, 1 for sentence B
            position_ids: Position indices, shape (batch_size, seq_length)
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            return_dict: Whether to return a dictionary (vs tuple)

        Returns:
            If return_dict=True:
                Dictionary with keys:
                - last_hidden_state: shape (batch_size, seq_length, hidden_size)
                - pooler_output: shape (batch_size, hidden_size) or None

            If return_dict=False:
                Tuple: (last_hidden_state, pooler_output)

        Example:
            >>> input_ids = torch.tensor([[101, 2023, 2003, 102]])  # [CLS] this is [SEP]
            >>> attention_mask = torch.tensor([[1, 1, 1, 1]])
            >>> outputs = model(input_ids, attention_mask)
            >>> last_hidden_state = outputs["last_hidden_state"]  # (1, 4, 768)
            >>> pooler_output = outputs["pooler_output"]  # (1, 768)
        """
        # ============================================================
        # STEP 1: Validate inputs
        # ============================================================
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # ============================================================
        # STEP 2: Prepare attention mask
        # ============================================================
        if attention_mask is None:
            # If no mask provided, assume all tokens are real (no padding)
            attention_mask = torch.ones(input_shape, device=device)

        # Expand attention mask to shape (batch_size, 1, 1, seq_length)
        # The extra dimensions are for broadcasting across attention heads
        extended_attention_mask = attention_mask[:, None, None, :]

        # Convert mask to attention scores:
        # - 0.0 for real tokens (can attend)
        # - -10000.0 for padding tokens (cannot attend)
        # After softmax, -10000 becomes ~0, effectively masking those positions
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # ============================================================
        # STEP 3: Get embeddings
        # ============================================================
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # ============================================================
        # STEP 4: Pass through encoder stack
        # ============================================================
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )

        # Get the final hidden states from the last layer
        sequence_output = encoder_outputs.last_hidden_state

        # ============================================================
        # STEP 5: Pool the [CLS] token (if pooler exists)
        # ============================================================
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # ============================================================
        # STEP 6: Return outputs
        # ============================================================
        if not return_dict:
            return (sequence_output, pooled_output)

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
        }


__all__ = ["BertPreTrainedModel", "BertModel"]
