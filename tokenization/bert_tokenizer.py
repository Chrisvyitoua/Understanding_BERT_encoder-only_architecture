"""
BERT Tokenizer Implementation

This module implements the BertTokenizer class which handles the complete
tokenization pipeline for BERT models using the WordPiece algorithm.

TOKENIZATION PIPELINE:
======================
The BERT tokenization process consists of several stages executed in sequence:

1. Normalization: Clean and preprocess text (lowercasing, accent removal, etc.)
2. Pre-tokenization: Split text into words (by whitespace and punctuation)
3. WordPiece: Split words into subword tokens using the WordPiece algorithm
4. Post-processing: Add special tokens ([CLS], [SEP]) and create token_type_ids

EXAMPLE FLOW:
=============
Input text: "Hello, world!"
→ After normalization: "hello, world!" (if do_lower_case=True)
→ After pre-tokenization: ["hello", ",", "world", "!"]
→ After WordPiece: ["hello", ",", "world", "!"]
→ After post-processing: ["[CLS]", "hello", ",", "world", "!", "[SEP]"]

SPECIAL TOKENS:
===============
- [PAD]: Padding token to make sequences the same length in a batch
- [UNK]: Unknown token for out-of-vocabulary words
- [CLS]: Classification token placed at the start of every sequence
- [SEP]: Separator token used to separate sequences in sequence pairs
- [MASK]: Mask token used in masked language modeling pretraining

WORDPIECE ALGORITHM:
====================
WordPiece is a subword tokenization algorithm that breaks words into smaller units.
This helps handle rare words and reduces vocabulary size.

Example: "playing" → ["play", "##ing"]
The "##" prefix indicates a subword continuation (not a word start).
"""

from typing import Optional

# HuggingFace tokenizers library components
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

# Note: In a standalone version, you would replace these with actual imports
# For educational purposes, we assume these are available from the transformers library
try:
    from ...tokenization_utils_tokenizers import TokenizersBackend
    from ...utils import logging
except ImportError:
    # Fallback for standalone usage
    print("Warning: Running in standalone mode. Some features may not be available.")
    TokenizersBackend = object
    logging = None


# Initialize logger if available
logger = logging.get_logger(__name__) if logging else None

# Dictionary mapping vocabulary file types to their default filenames
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "tokenizer_file": "tokenizer.json"
}


class BertTokenizer(TokenizersBackend):
    """
    BERT Tokenizer - Converts text to token IDs using WordPiece algorithm.

    This tokenizer implements the full BERT tokenization pipeline:
    - Text normalization (lowercasing, accent removal)
    - Pre-tokenization (splitting on whitespace and punctuation)
    - WordPiece subword tokenization
    - Post-processing (adding [CLS], [SEP] tokens)

    Args:
        vocab_file (str, optional): Path to vocabulary file
        do_lower_case (bool, optional): Whether to lowercase the input. Defaults to False.
        unk_token (str, optional): The unknown token. Defaults to "[UNK]".
        sep_token (str, optional): The separator token. Defaults to "[SEP]".
        pad_token (str, optional): The padding token. Defaults to "[PAD]".
        cls_token (str, optional): The classifier token. Defaults to "[CLS]".
        mask_token (str, optional): The mask token. Defaults to "[MASK]".
        tokenize_chinese_chars (bool, optional): Whether to tokenize Chinese characters. Defaults to True.
        strip_accents (bool, optional): Whether to strip accents. Defaults to None (auto-detect).
        vocab (dict, optional): Custom vocabulary dictionary. Defaults to None.

    Example:
        >>> tokenizer = BertTokenizer(do_lower_case=True)
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(tokens)
        [2, 7592, 1010, 2088, 999, 3]  # [CLS] hello , world ! [SEP]
    """

    # Class-level attributes
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    slow_tokenizer_class = None

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
        """
        Initialize the BERT tokenizer with configuration parameters.

        The initialization process:
        1. Store configuration (lowercasing, Chinese chars, accents)
        2. Load or create vocabulary
        3. Create WordPiece tokenizer
        4. Configure normalizer (text preprocessing)
        5. Configure pre-tokenizer (initial word splitting)
        6. Configure decoder (token-to-text conversion)
        7. Configure post-processor (add special tokens)
        """
        # Store tokenization configuration parameters
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

        # STEP 1: Initialize the vocabulary
        # The vocabulary maps tokens (strings) to IDs (integers)
        if vocab is not None:
            # Handle both list and dict formats for vocabulary
            self._vocab = (
                {token: idx for idx, (token, _score) in enumerate(vocab)}
                if isinstance(vocab, list)
                else vocab
            )
        else:
            # Create a minimal default vocabulary with only special tokens
            # Note: In practice, a full vocabulary file would be loaded with ~30,000 tokens
            self._vocab = {
                str(pad_token): 0,    # [PAD] - for padding sequences to same length
                str(unk_token): 1,    # [UNK] - for unknown/out-of-vocabulary tokens
                str(cls_token): 2,    # [CLS] - classification token at start of sequence
                str(sep_token): 3,    # [SEP] - separator between sequences
                str(mask_token): 4,   # [MASK] - for masked language modeling
            }

        # STEP 2: Create the main tokenizer with WordPiece algorithm
        # WordPiece splits words into subword units (e.g., "playing" → "play" + "##ing")
        self._tokenizer = Tokenizer(
            WordPiece(self._vocab, unk_token=str(unk_token))
        )

        # STEP 3: Configure the normalizer (text preprocessing)
        # This runs FIRST in the tokenization pipeline
        # Performs: text cleaning, lowercasing, accent removal, Chinese character handling
        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,                                # Remove control characters, normalize whitespace
            handle_chinese_chars=tokenize_chinese_chars,    # Add spaces around Chinese characters
            strip_accents=strip_accents,                    # Remove accents (é → e)
            lowercase=do_lower_case,                        # Convert to lowercase
        )

        # STEP 4: Configure the pre-tokenizer (initial splitting)
        # This runs SECOND, after normalization
        # Splits on whitespace and punctuation (e.g., "Hello, world!" → ["Hello", ",", "world", "!"])
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # STEP 5: Configure the decoder (for converting tokens back to text)
        # The "##" prefix indicates subword continuation in WordPiece
        # Decoder removes these prefixes when converting back to text
        # Example: ["play", "##ing"] → "playing"
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")

        tokenizer_object = self._tokenizer

        # STEP 6: Initialize the parent class (TokenizersBackend)
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

        # STEP 7: Configure the post-processor (adds special tokens)
        # This runs LAST in the tokenization pipeline, after WordPiece tokenization
        # Get the token IDs for [CLS] and [SEP] tokens
        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 2
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 3

        # Template processing defines how to format tokenized sequences
        # The number after ":" is the token_type_id (0 for first sequence, 1 for second)
        self._tokenizer.post_processor = processors.TemplateProcessing(
            # For single sequence: [CLS] sequence [SEP]
            # Example: "Hello world" → [CLS] Hello world [SEP]
            # All tokens have token_type_id = 0
            single=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0",

            # For sequence pairs: [CLS] sequence_A [SEP] sequence_B [SEP]
            # Example: ("Hello", "world") → [CLS] Hello [SEP] world [SEP]
            # sequence_A tokens have token_type_id=0, sequence_B tokens have token_type_id=1
            # This distinction is important for Next Sentence Prediction task
            pair=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0 $B:1 {str(self.sep_token)}:1",

            # Register the special tokens and their IDs
            special_tokens=[
                (str(self.cls_token), cls_token_id),
                (str(self.sep_token), sep_token_id),
            ],
        )


# Module exports
__all__ = ["BertTokenizer", "BertTokenizerFast"]

# Alias for backward compatibility
# BertTokenizerFast is an alias to BertTokenizer since this implementation
# is already fast (uses the Rust-backed HuggingFace tokenizers library)
BertTokenizerFast = BertTokenizer
