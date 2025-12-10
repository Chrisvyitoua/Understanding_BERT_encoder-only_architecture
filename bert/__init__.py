"""
BERT Module - Complete BERT Implementation

This module contains all components needed to understand and use BERT:

CORE COMPONENTS:
================
- config.py: Model configuration (BertConfig)
- embeddings.py: Input embeddings (word + position + token type)
- attention.py: Multi-head self-attention mechanisms
- feedforward.py: Position-wise feed-forward networks
- encoder.py: Transformer encoder layers
- pooler.py: Sequence pooling (for classification)
- prediction_heads.py: MLM and NSP heads
- base_model.py: Core BERT model
- task_models.py: Task-specific models

EDUCATIONAL STRUCTURE:
======================
This module is organized for educational purposes:
1. Start with config.py to understand hyperparameters
2. Study embeddings.py to see how inputs are processed
3. Learn attention.py to understand the attention mechanism
4. Explore feedforward.py for the FFN component
5. See encoder.py for how layers are stacked
6. Check pooler.py for sequence-level representation
7. Review prediction_heads.py for pretraining objectives
8. Understand base_model.py for the complete architecture
9. Study task_models.py for downstream applications

QUICK USAGE:
============
from bert import BertConfig, BertModel, BertForSequenceClassification

# Create a config
config = BertConfig()

# Create a base model
model = BertModel(config)

# Or create a task-specific model
classifier = BertForSequenceClassification(config)
"""

# Configuration
from .config import BertConfig

# Core architectural components
from .embeddings import BertEmbeddings
from .attention import (
    BertSelfAttention,
    BertCrossAttention,
    BertSelfOutput,
    BertAttention,
    eager_attention_forward,
)
from .feedforward import BertIntermediate, BertOutput
from .encoder import BertLayer, BertEncoder
from .pooler import BertPooler

# Prediction heads for pretraining
from .prediction_heads import (
    BertPredictionHeadTransform,
    BertLMPredictionHead,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPreTrainingHeads,
)

# Base models
from .base_model import BertPreTrainedModel, BertModel

# Task-specific models
from .task_models import (
    BertForPreTraining,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

__all__ = [
    # Configuration
    "BertConfig",
    # Embeddings
    "BertEmbeddings",
    # Attention
    "BertSelfAttention",
    "BertCrossAttention",
    "BertSelfOutput",
    "BertAttention",
    "eager_attention_forward",
    # Feed-forward
    "BertIntermediate",
    "BertOutput",
    # Encoder
    "BertLayer",
    "BertEncoder",
    # Pooler
    "BertPooler",
    # Prediction heads
    "BertPredictionHeadTransform",
    "BertLMPredictionHead",
    "BertOnlyMLMHead",
    "BertOnlyNSPHead",
    "BertPreTrainingHeads",
    # Base models
    "BertPreTrainedModel",
    "BertModel",
    # Task models
    "BertForPreTraining",
    "BertForMaskedLM",
    "BertForSequenceClassification",
    "BertForQuestionAnswering",
]
