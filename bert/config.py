"""
BERT Model Configuration

This module defines the BertConfig class which stores all hyperparameters that define
the BERT model architecture. The configuration separates model architecture definition
from the model code itself, enabling:

1. Easy model variants: Create BERT-base, BERT-large, or custom sizes by changing config
2. Model serialization: Save/load model architecture specifications
3. Reproducibility: Ensure consistent model structure across different environments
4. Flexibility: Modify architecture without changing model implementation code

COMMON BERT CONFIGURATIONS:
============================
BERT-base (default):
- hidden_size: 768
- num_hidden_layers: 12
- num_attention_heads: 12
- intermediate_size: 3072
- Total parameters: ~110M

BERT-large:
- hidden_size: 1024
- num_hidden_layers: 24
- num_attention_heads: 16
- intermediate_size: 4096
- Total parameters: ~340M

KEY ARCHITECTURE PRINCIPLES:
============================
1. hidden_size must be divisible by num_attention_heads
   - Each attention head operates on hidden_size / num_attention_heads dimensions
   - For BERT-base: 768 / 12 = 64 dimensions per head

2. intermediate_size is typically 4x hidden_size
   - The feed-forward network expands then contracts
   - This expansion provides capacity for learning complex transformations

3. max_position_embeddings defines the maximum sequence length
   - Standard BERT supports up to 512 tokens
   - Longer sequences require more memory and computation
"""

# Note: In a standalone version, you would replace these with actual implementations
try:
    from ...configuration_utils import PreTrainedConfig
    from ...utils import logging
except ImportError:
    # Fallback for standalone usage
    print("Warning: Running in standalone mode. Using basic config.")
    PreTrainedConfig = object
    logging = None


logger = logging.get_logger(__name__) if logging else None


class BertConfig(PreTrainedConfig):
    """
    Configuration class to store the configuration of a BERT model.

    This class defines all the hyperparameters needed to instantiate a BERT model,
    from the vocabulary size to the number of layers and attention heads.

    Args:
        vocab_size (int, optional): Vocabulary size of the BERT model. Defines the number of
            different tokens that can be represented by the inputs_ids. Defaults to 30522.

        hidden_size (int, optional): Dimensionality of the encoder layers and the pooler layer.
            This is the core dimension used throughout the model. Defaults to 768.

        num_hidden_layers (int, optional): Number of hidden layers in the Transformer encoder.
            More layers = more model capacity but slower training. Defaults to 12.

        num_attention_heads (int, optional): Number of attention heads for each attention layer.
            Must divide hidden_size evenly. More heads = more diverse attention patterns. Defaults to 12.

        intermediate_size (int, optional): Dimensionality of the "intermediate" (feed-forward)
            layer in the Transformer encoder. Typically 4x hidden_size. Defaults to 3072.

        hidden_act (str or Callable, optional): The non-linear activation function in the
            encoder and pooler. Supports "gelu", "relu", "silu", "gelu_new". Defaults to "gelu".

        hidden_dropout_prob (float, optional): The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler. Defaults to 0.1.

        attention_probs_dropout_prob (float, optional): The dropout ratio for the attention
            probabilities. Prevents attention overfitting. Defaults to 0.1.

        max_position_embeddings (int, optional): The maximum sequence length that this model
            might ever be used with. Standard BERT uses 512. Defaults to 512.

        type_vocab_size (int, optional): The vocabulary size of the token_type_ids. Used to
            distinguish between sentence A and sentence B in sentence pairs. Defaults to 2.

        initializer_range (float, optional): The standard deviation of the truncated_normal_initializer
            for initializing all weight matrices. Defaults to 0.02.

        layer_norm_eps (float, optional): The epsilon used by the layer normalization layers.
            Small value for numerical stability. Defaults to 1e-12.

        pad_token_id (int, optional): The ID of the padding token in the vocabulary. Defaults to 0.

        use_cache (bool, optional): Whether or not the model should return the last key/values
            attentions for faster autoregressive generation. Defaults to True.

        classifier_dropout (float, optional): The dropout ratio for the classification head.
            If None, defaults to hidden_dropout_prob. Defaults to None.

    Example:
        >>> from bert.config import BertConfig
        >>> from bert.base_model import BertModel
        >>>
        >>> # Create a BERT-base configuration
        >>> config = BertConfig()
        >>>
        >>> # Create a BERT-large configuration
        >>> config_large = BertConfig(
        ...     hidden_size=1024,
        ...     num_hidden_layers=24,
        ...     num_attention_heads=16,
        ...     intermediate_size=4096
        ... )
        >>>
        >>> # Create a custom small BERT for experimentation
        >>> config_small = BertConfig(
        ...     hidden_size=256,
        ...     num_hidden_layers=6,
        ...     num_attention_heads=8,
        ...     intermediate_size=1024
        ... )
    """

    # Model type identifier used by HuggingFace to recognize BERT models
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,              # Size of vocabulary (30522 for BERT's WordPiece vocab)
        hidden_size=768,                # Dimensionality of embeddings and hidden states (768 for base, 1024 for large)
        num_hidden_layers=12,           # Number of transformer layers stacked (12 for base, 24 for large)
        num_attention_heads=12,         # Number of attention heads per layer (12 for base, 16 for large)
        intermediate_size=3072,         # Size of feed-forward layer (typically 4x hidden_size)
        hidden_act="gelu",              # Activation function (GELU is smoother than ReLU, helps training)
        hidden_dropout_prob=0.1,        # Dropout rate for embeddings and hidden layers (10% for regularization)
        attention_probs_dropout_prob=0.1,  # Dropout rate for attention weights (prevents attention overfitting)
        max_position_embeddings=512,    # Maximum sequence length the model can handle
        type_vocab_size=2,              # Number of token types (2 for sentence A/B in sentence pairs)
        initializer_range=0.02,         # Standard deviation for weight initialization (small values for stability)
        layer_norm_eps=1e-12,           # Small constant for numerical stability in layer normalization
        pad_token_id=0,                 # ID of padding token in vocabulary
        use_cache=True,                 # Whether to cache key/value pairs for faster generation
        classifier_dropout=None,        # Dropout for classification head (defaults to hidden_dropout_prob)
        **kwargs,                       # Additional arguments passed to parent class
    ):
        """
        Initialize BERT configuration with architecture hyperparameters.

        The configuration defines the structure of the BERT model but doesn't
        create the actual model weights. It's used to instantiate a BertModel.
        """
        # Initialize parent PreTrainedConfig class
        # This sets up common configuration attributes and methods
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # ============================================================
        # ARCHITECTURE PARAMETERS
        # These define the model's structure and capacity
        # ============================================================

        # Vocabulary and embedding dimensions
        self.vocab_size = vocab_size                    # Number of tokens in vocabulary
        self.hidden_size = hidden_size                  # Core dimension throughout the model

        # Transformer architecture dimensions
        self.num_hidden_layers = num_hidden_layers      # Depth of the model (more layers = more capacity)
        self.num_attention_heads = num_attention_heads  # Width of attention (more heads = more diverse patterns)

        # IMPORTANT: hidden_size must be divisible by num_attention_heads
        # Each head operates on hidden_size / num_attention_heads dimensions
        # For BERT-base: 768 / 12 = 64 dimensions per head
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"

        # Feed-forward network size
        self.intermediate_size = intermediate_size      # FFN expands then contracts (3072 = 4 * 768 for base)
        self.hidden_act = hidden_act                   # Non-linearity in FFN (GELU is default)

        # ============================================================
        # REGULARIZATION PARAMETERS
        # These prevent overfitting during training
        # ============================================================
        self.hidden_dropout_prob = hidden_dropout_prob                  # Dropout after embeddings and FFN
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # Dropout on attention weights
        self.classifier_dropout = classifier_dropout                    # Dropout before classification layer

        # ============================================================
        # POSITION AND TOKEN TYPE PARAMETERS
        # ============================================================
        self.max_position_embeddings = max_position_embeddings  # Maximum input length (512 for standard BERT)
        self.type_vocab_size = type_vocab_size                  # Sentence A vs B distinction (for NSP task)

        # ============================================================
        # INITIALIZATION AND NUMERICAL STABILITY
        # ============================================================
        self.initializer_range = initializer_range  # Controls initial weight magnitudes
        self.layer_norm_eps = layer_norm_eps        # Prevents division by zero in normalization

        # ============================================================
        # CACHING (for generation tasks)
        # ============================================================
        self.use_cache = use_cache  # Store past key/values for faster autoregressive generation


__all__ = ["BertConfig"]
