# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT model configuration

CONFIGURATION PURPOSE:
======================
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
"""

# Base class for all model configurations in HuggingFace Transformers
from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class BertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

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
        # Initialize parent PreTrainedConfig class
        # This sets up common configuration attributes and methods
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # ARCHITECTURE PARAMETERS
        # These define the model's structure and capacity

        # Vocabulary and embedding dimensions
        self.vocab_size = vocab_size                    # Number of tokens in vocabulary
        self.hidden_size = hidden_size                  # Core dimension throughout the model

        # Transformer architecture dimensions
        self.num_hidden_layers = num_hidden_layers      # Depth of the model (more layers = more capacity)
        self.num_attention_heads = num_attention_heads  # Width of attention (more heads = more diverse attention patterns)

        # Note: hidden_size must be divisible by num_attention_heads
        # Each head operates on hidden_size / num_attention_heads dimensions
        # For BERT-base: 768 / 12 = 64 dimensions per head

        # Feed-forward network size
        self.intermediate_size = intermediate_size      # FFN expands then contracts (3072 = 4 * 768 for base)
        self.hidden_act = hidden_act                   # Non-linearity in FFN (GELU is default)

        # REGULARIZATION PARAMETERS
        # These prevent overfitting during training
        self.hidden_dropout_prob = hidden_dropout_prob                  # Dropout after embeddings and FFN
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # Dropout on attention weights
        self.classifier_dropout = classifier_dropout                    # Dropout before classification layer

        # POSITION AND TOKEN TYPE PARAMETERS
        self.max_position_embeddings = max_position_embeddings  # Maximum input length (512 for standard BERT)
        self.type_vocab_size = type_vocab_size                  # Sentence A vs B distinction (for NSP task)

        # INITIALIZATION AND NUMERICAL STABILITY
        self.initializer_range = initializer_range  # Controls initial weight magnitudes
        self.layer_norm_eps = layer_norm_eps        # Prevents division by zero in normalization

        # CACHING (for generation tasks)
        self.use_cache = use_cache  # Store past key/values for faster autoregressive generation


__all__ = ["BertConfig"]
