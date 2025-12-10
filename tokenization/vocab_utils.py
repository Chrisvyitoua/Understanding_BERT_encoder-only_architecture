"""
Vocabulary Loading Utilities for BERT Tokenization

This module provides utilities for loading and managing BERT vocabularies.
The vocabulary maps tokens (words and subwords) to unique integer IDs.

VOCABULARY FORMAT:
==================
BERT uses a vocabulary file (typically vocab.txt) where each line contains one token.
The line number (0-indexed) becomes the token's ID.

Example vocab.txt:
    [PAD]       # ID: 0
    [UNK]       # ID: 1
    [CLS]       # ID: 2
    [SEP]       # ID: 3
    [MASK]      # ID: 4
    the         # ID: 5
    a           # ID: 6
    ...

WHY ORDERED DICTIONARY:
=======================
We use OrderedDict to preserve the exact order of tokens as they appear in the file.
This ensures consistent token IDs across different program runs.
"""

import collections
from typing import Dict


def load_vocab(vocab_file: str) -> Dict[str, int]:
    """
    Loads a vocabulary file into a dictionary mapping tokens to IDs.

    This function reads a vocabulary file where each line contains one token.
    The position (line number) of each token becomes its unique ID.

    Args:
        vocab_file (str): Path to the vocabulary file (e.g., "vocab.txt")

    Returns:
        Dict[str, int]: Ordered dictionary mapping token strings to integer IDs

    Example:
        >>> vocab = load_vocab("vocab.txt")
        >>> vocab["[CLS]"]
        2
        >>> vocab["the"]
        5

    Note:
        The order of tokens is preserved using OrderedDict, which is critical
        for maintaining consistent token IDs across different environments.
    """
    # Initialize an ordered dictionary to preserve the order of tokens
    # This is important because token position determines its ID
    vocab = collections.OrderedDict()

    # Read the vocabulary file (typically vocab.txt)
    # Each line contains one token
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()

    # Build the vocabulary dictionary
    # Key: token string, Value: token ID (index in file)
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # Remove newline character
        vocab[token] = index

    return vocab


__all__ = ["load_vocab"]
