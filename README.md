# Understanding BERT architecture through its source code

This repository reorganizes the BERT implementation into well-separated, commented components, each isolated in its own file to help understand the architecture. 

```
Understanding_BERT_encoder-only_architecture/
│
├── tokenization/              # Tokenization Components
│   ├── __init__.py
│   ├── vocab_utils.py        # Vocabulary loading utilities
│   └── bert_tokenizer.py     # BERT WordPiece tokenizer
│
├── bert/                      # BERT Model Components
│   ├── __init__.py
│   ├── config.py             # Model configuration & hyperparameters
│   ├── embeddings.py         # Input embeddings 
│   ├── attention.py          # Multi-head attention mechanisms
│   ├── feedforward.py        # Position-wise feed-forward networks
│   ├── encoder.py            # Transformer encoder layers
│   ├── pooler.py             # Sequence pooling for classification
│   ├── prediction_heads.py   # Pretraining heads (MLM & NSP)
│   ├── base_model.py         # Core BERT model
│   └── task_models.py        # Task-specific models
│
├── __init__.py               
└── README.md                 # This file
```
## Additional Resources

- [Original BERT Paper](https://arxiv.org/abs/1810.04805) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- [Illustrated BERT](http://jalammar.github.io/illustrated-bert/) - Visual explanations by Jay Alammar
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Production implementation
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper

