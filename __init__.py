"""
Understanding BERT: Educational Implementation

This is an educational restructuring of BERT (Bidirectional Encoder Representations from Transformers)
organized into clear, well-documented modules for learning purposes.

PROJECT STRUCTURE:
==================
tokenization/          - Tokenization components
    vocab_utils.py     - Vocabulary loading utilities
    bert_tokenizer.py  - BERT WordPiece tokenizer

bert/                  - BERT model components
    config.py          - Model configuration
    embeddings.py      - Input embeddings
    attention.py       - Attention mechanisms
    feedforward.py     - Feed-forward networks
    encoder.py         - Transformer layers
    pooler.py          - Sequence pooling
    prediction_heads.py- Pretraining heads (MLM, NSP)
    base_model.py      - Core BERT model
    task_models.py     - Task-specific models

LEARNING PATH:
==============
1. Configuration (bert.config):
   Start here to understand BERT's hyperparameters and architecture choices.

2. Tokenization (tokenization):
   Learn how text is converted to tokens using WordPiece algorithm.

3. Embeddings (bert.embeddings):
   Understand how tokens become dense vectors (word + position + segment).

4. Attention (bert.attention):
   Study the multi-head self-attention mechanism - the heart of transformers.

5. Feed-Forward (bert.feedforward):
   Explore the position-wise FFN that processes each token independently.

6. Encoder (bert.encoder):
   See how attention and FFN combine into transformer layers and stack together.

7. Pooler (bert.pooler):
   Learn how [CLS] token is used for sequence-level tasks.

8. Prediction Heads (bert.prediction_heads):
   Understand pretraining objectives: MLM and NSP.

9. Base Model (bert.base_model):
   See how all components combine into the complete BERT model.

10. Task Models (bert.task_models):
    Learn how BERT adapts to specific tasks through fine-tuning.

QUICK START:
============
# Import configuration and models
from bert import BertConfig, BertModel, BertForSequenceClassification

# Import tokenizer
from tokenization import BertTokenizer

# Create a BERT-base configuration
config = BertConfig()  # Defaults to BERT-base (12 layers, 768 hidden, 12 heads)

# Create a base BERT model
model = BertModel(config)

# Or create a task-specific model (e.g., for sentiment classification)
classifier = BertForSequenceClassification(config)

# Create a tokenizer
tokenizer = BertTokenizer(do_lower_case=True)

EDUCATIONAL NOTES:
==================
- All modules have extensive comments explaining concepts
- Code is organized for clarity over optimization
- Each file focuses on a single conceptual component
- Comments explain both "what" and "why"

ORIGINAL AUTHORS:
=================
This code is derived from the HuggingFace Transformers library and
Google's original BERT implementation, reorganized for educational purposes.
"""

# ============================================================
# TOKENIZATION COMPONENTS
# ============================================================
from tokenization import (
    BertTokenizer,
    BertTokenizerFast,
    load_vocab,
)

# ============================================================
# BERT MODEL COMPONENTS
# ============================================================

# Configuration
from bert import BertConfig

# Core architectural components
from bert import (
    BertEmbeddings,
    BertSelfAttention,
    BertCrossAttention,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayer,
    BertEncoder,
    BertPooler,
)

# Prediction heads for pretraining
from bert import (
    BertPreTrainingHeads,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
)

# Base models
from bert import (
    BertPreTrainedModel,
    BertModel,
)

# Task-specific models
from bert import (
    BertForPreTraining,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

# ============================================================
# DEFINE EXPORTS
# ============================================================
__all__ = [
    # Tokenization
    "BertTokenizer",
    "BertTokenizerFast",
    "load_vocab",
    # Configuration
    "BertConfig",
    # Core components
    "BertEmbeddings",
    "BertSelfAttention",
    "BertCrossAttention",
    "BertAttention",
    "BertIntermediate",
    "BertOutput",
    "BertLayer",
    "BertEncoder",
    "BertPooler",
    # Prediction heads
    "BertPreTrainingHeads",
    "BertOnlyMLMHead",
    "BertOnlyNSPHead",
    # Base models
    "BertPreTrainedModel",
    "BertModel",
    # Task models
    "BertForPreTraining",
    "BertForMaskedLM",
    "BertForSequenceClassification",
    "BertForQuestionAnswering",
]

# Version information
__version__ = "1.0.0-educational"

# Educational attribution
__educational_note__ = """
This is an educational restructuring of BERT for learning purposes.
Original implementations by Google AI Language Team and HuggingFace.
Reorganized with extensive comments for understanding transformer architecture.
"""
