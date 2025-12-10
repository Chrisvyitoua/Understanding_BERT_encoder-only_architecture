# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for Bert."""

# Standard library imports
import collections  # Used for OrderedDict to maintain vocabulary order
from typing import Optional

# HuggingFace tokenizers library components
# - Tokenizer: Main tokenizer class
# - decoders: Converts token IDs back to text
# - normalizers: Text preprocessing (lowercasing, accent removal, etc.)
# - pre_tokenizers: Initial text splitting before WordPiece algorithm
# - processors: Post-processing to add special tokens like [CLS] and [SEP]
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece  # WordPiece algorithm used by BERT

# Internal HuggingFace imports
from ...tokenization_utils_tokenizers import TokenizersBackend  # Base class for fast tokenizers
from ...utils import logging


logger = logging.get_logger(__name__)

# Dictionary mapping vocabulary file types to their default filenames
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
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


class BertTokenizer(TokenizersBackend):
    r"""
    Construct a BERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    TOKENIZATION PIPELINE OVERVIEW:
    ================================
    The BERT tokenization process consists of several stages executed in sequence:

    1. Normalization: Clean and preprocess text (lowercasing, accent removal, etc.)
    2. Pre-tokenization: Split text into words (by whitespace and punctuation)
    3. WordPiece: Split words into subword tokens using the WordPiece algorithm
    4. Post-processing: Add special tokens ([CLS], [SEP]) and create token_type_ids

    Example tokenization flow:
    Input text: "Hello, world!"
    → After normalization: "hello, world!" (if do_lower_case=True)
    → After pre-tokenization: ["hello", ",", "world", "!"]
    → After WordPiece: ["hello", ",", "world", "!"]
    → After post-processing: ["[CLS]", "hello", ",", "world", "!", "[SEP]"]

    Args:
        vocab_file (`str`, *optional*):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.
    """

    # Class-level attributes
    vocab_files_names = VOCAB_FILES_NAMES  # Default names for vocabulary files
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]  # Expected model inputs
    slow_tokenizer_class = None  # No slow tokenizer fallback for this implementation

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        do_lower_case: bool = False,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        vocab: Optional[dict] = None,
        **kwargs,
    ):
        # Store tokenization configuration parameters
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

        # Step 1: Initialize the vocabulary
        # If a custom vocabulary is provided, use it
        if vocab is not None:
            # Handle both list and dict formats for vocabulary
            self._vocab = (
                {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
            )
        else:
            # Create a minimal default vocabulary with only special tokens
            # Note: In practice, a full vocabulary file would be loaded with thousands of tokens
            self._vocab = {
                str(pad_token): 0,    # [PAD] - for padding sequences to same length
                str(unk_token): 1,    # [UNK] - for unknown/out-of-vocabulary tokens
                str(cls_token): 2,    # [CLS] - classification token at start of sequence
                str(sep_token): 3,    # [SEP] - separator between sequences
                str(mask_token): 4,   # [MASK] - for masked language modeling
            }

        # Step 2: Create the main tokenizer with WordPiece algorithm
        # WordPiece splits words into subword units (e.g., "playing" -> "play" + "##ing")
        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))

        # Step 3: Configure the normalizer (text preprocessing)
        # This runs FIRST in the tokenization pipeline
        # Performs: text cleaning, lowercasing, accent removal, Chinese character handling
        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,                                # Remove control characters and normalize whitespace
            handle_chinese_chars=tokenize_chinese_chars,    # Add spaces around Chinese characters
            strip_accents=strip_accents,                    # Remove accents (é -> e)
            lowercase=do_lower_case,                        # Convert to lowercase
        )

        # Step 4: Configure the pre-tokenizer (initial splitting)
        # This runs SECOND, after normalization
        # Splits on whitespace and punctuation (e.g., "Hello, world!" -> ["Hello", ",", "world", "!"])
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # Step 5: Configure the decoder (for converting tokens back to text)
        # The "##" prefix indicates subword continuation in WordPiece
        # Decoder removes these prefixes when converting back to text
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")

        tokenizer_object = self._tokenizer

        # Step 6: Initialize the parent class (TokenizersBackend)
        # This provides common tokenizer functionality and integrates with HuggingFace ecosystem
        super().__init__(
            tokenizer_object=tokenizer_object,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # Step 7: Configure the post-processor (adds special tokens)
        # This runs LAST in the tokenization pipeline, after WordPiece tokenization
        # Get the token IDs for [CLS] and [SEP] tokens
        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 2
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 3

        # Template processing defines how to format tokenized sequences
        # The number after ":" is the token_type_id (0 for first sequence, 1 for second sequence)
        self._tokenizer.post_processor = processors.TemplateProcessing(
            # For single sequence: [CLS] sequence [SEP]
            # Example: "Hello world" -> [CLS] Hello world [SEP]
            single=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0",

            # For sequence pairs: [CLS] sequence_A [SEP] sequence_B [SEP]
            # Example: ("Hello", "world") -> [CLS] Hello [SEP] world [SEP]
            # Note: sequence_A has token_type_id=0, sequence_B has token_type_id=1
            pair=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0 $B:1 {str(self.sep_token)}:1",

            # Register the special tokens and their IDs
            special_tokens=[
                (str(self.cls_token), cls_token_id),
                (str(self.sep_token), sep_token_id),
            ],
        )


# Module exports - defines what gets imported when using "from transformers import *"
__all__ = ["BertTokenizer"]

# Alias for backward compatibility
# BertTokenizerFast is an alias to BertTokenizer since this implementation is already fast
# (it uses the Rust-backed HuggingFace tokenizers library)
BertTokenizerFast = BertTokenizer
