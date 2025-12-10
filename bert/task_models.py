"""
BERT Task-Specific Models

This module contains BERT models adapted for specific downstream tasks.
Each model adds task-specific layers on top of the base BERT model.

TASK MODELS INCLUDED:
=====================
1. BertForPreTraining: Combined MLM + NSP pretraining
2. BertForMaskedLM: Masked language modeling only
3. BertForSequenceClassification: Text classification, sentiment analysis, etc.
4. BertForQuestionAnswering: Extractive question answering (SQuAD-style)

GENERAL PATTERN:
================
All task models follow this pattern:
1. Initialize base BERT model
2. Add task-specific head(s)
3. Define forward pass with task-specific logic
4. Compute and return loss (during training)

TRANSFER LEARNING WORKFLOW:
============================
1. Pretrain: Use BertForPreTraining on large unlabeled corpus
2. Fine-tune: Use task-specific model (e.g., BertForSequenceClassification)
               on labeled data for your task
3. Inference: Use the fine-tuned model for predictions

EDUCATIONAL NOTE:
=================
This is a simplified version focusing on the most common tasks.
Each model is heavily commented to explain the architecture and design choices.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# Import base components
try:
    from .base_model import BertModel, BertPreTrainedModel
    from .prediction_heads import BertPreTrainingHeads, BertOnlyMLMHead
    from .config import BertConfig
except ImportError:
    print("Warning: Could not import required components.")
    BertModel = None
    BertPreTrainedModel = None
    BertPreTrainingHeads = None
    BertOnlyMLMHead = None
    BertConfig = None


class BertForPreTraining(BertPreTrainedModel):
    """
    BERT model for pretraining with both MLM and NSP objectives.

    This model is used for the initial pretraining phase where BERT learns
    general language understanding from large amounts of unlabeled text.

    PRETRAINING OBJECTIVES:
    =======================
    1. Masked Language Modeling (MLM):
       - Randomly mask 15% of tokens
       - Predict the original tokens
       - Loss: Cross-entropy on masked tokens only

    2. Next Sentence Prediction (NSP):
       - Given two sentences A and B
       - Predict if B follows A in original text
       - Loss: Binary cross-entropy

    Total Loss = MLM Loss + NSP Loss

    PRETRAINING DATA:
    =================
    Example input for pretraining:
    - Sentence A: "The cat sat on the mat."
    - Sentence B: "It was very comfortable." (50% chance) OR
                  "Paris is the capital of France." (50% chance - random)
    - Masked: "The [MASK] sat on the [MASK]."
    - Labels: ["cat", "mat"], IsNext or NotNext

    After pretraining, the model has learned:
    - Word meanings and relationships (from MLM)
    - Sentence-level coherence (from NSP)
    - General language patterns
    """

    def __init__(self, config):
        super().__init__(config)

        # Base BERT model (embeddings + encoder + pooler)
        self.bert = BertModel(config)

        # Pretraining heads (MLM + NSP)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for pretraining.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask
            token_type_ids: Segment IDs (0 for A, 1 for B)
            labels: True tokens for masked positions, shape (batch_size, seq_length)
                    -100 for non-masked positions (ignored in loss)
            next_sentence_label: NSP labels, shape (batch_size,)
                                0 = NotNext, 1 = IsNext

        Returns:
            Dictionary with:
            - loss: Combined MLM + NSP loss (if labels provided)
            - prediction_logits: MLM predictions
            - seq_relationship_logits: NSP predictions
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]

        # Get prediction scores
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()

            # MLM loss (only on masked tokens)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

            # NSP loss
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )

            total_loss = masked_lm_loss + next_sentence_loss

        return {
            "loss": total_loss,
            "prediction_logits": prediction_scores,
            "seq_relationship_logits": seq_relationship_score,
        }


class BertForMaskedLM(BertPreTrainedModel):
    """
    BERT model for Masked Language Modeling only (no NSP).

    This model is used for:
    1. Continue pretraining on domain-specific data
    2. Fill-in-the-blank tasks
    3. Text generation (though BERT is not ideal for this)

    USAGE:
    ======
    Example:
    Input:  "The capital of France is [MASK]."
    Output: Probability distribution over vocabulary for [MASK] position
    Prediction: "Paris" (highest probability)

    This is useful for:
    - Domain adaptation: Pretrain on medical/legal/scientific text
    - Masked token prediction tasks
    - Understanding model's language knowledge
    """

    def __init__(self, config):
        super().__init__(config)

        # Base BERT model
        self.bert = BertModel(config)

        # MLM head only (no NSP)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs with [MASK] tokens
            attention_mask: Attention mask
            token_type_ids: Segment IDs
            labels: True tokens for masked positions
                    -100 for non-masked (ignored in loss)

        Returns:
            Dictionary with:
            - loss: MLM loss (if labels provided)
            - logits: Predictions for all positions
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs["last_hidden_state"]

        # Get prediction scores
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        return {
            "loss": masked_lm_loss,
            "logits": prediction_scores,
        }


class BertForSequenceClassification(BertPreTrainedModel):
    """
    BERT model for sequence classification tasks.

    This is one of the most common BERT use cases, applicable to:
    - Sentiment analysis: Positive, Negative, Neutral
    - Topic classification: Sports, Politics, Technology, etc.
    - Natural Language Inference: Entailment, Contradiction, Neutral
    - Spam detection: Spam, Not Spam
    - Any task where you need to classify entire sequences

    ARCHITECTURE:
    =============
    Input → BERT → [CLS] representation → Dropout → Linear → Class logits

    The [CLS] token's representation contains information about the entire
    sequence (learned through self-attention), making it suitable for
    sequence-level classification.

    NUMBER OF LABELS:
    =================
    - Binary classification: num_labels = 2 (e.g., positive/negative)
    - Multi-class: num_labels = N (e.g., 5-star rating → 5 classes)
    - Multi-label: num_labels = N (e.g., topic tags, multiple can be true)

    USAGE EXAMPLE:
    ==============
    Sentiment Analysis:
    Input:  "This movie was amazing! I loved it."
    Output: [0.05, 0.05, 0.90]  # Probabilities for [Negative, Neutral, Positive]
    Prediction: Positive (class 2, 90% confidence)
    """

    def __init__(self, config):
        super().__init__(config)

        # Number of output classes
        self.num_labels = config.num_labels

        # Base BERT model with pooling (need [CLS] representation)
        self.bert = BertModel(config)

        # Dropout for regularization
        # Higher dropout than pretraining to prevent overfitting on smaller datasets
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Classification head: Linear layer
        # Maps from hidden_size (768) to num_labels (e.g., 3 for sentiment)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for sequence classification.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask
            token_type_ids: Segment IDs
            labels: True class labels, shape (batch_size,)
                    Values in range [0, num_labels)

        Returns:
            Dictionary with:
            - loss: Classification loss (if labels provided)
            - logits: Class scores, shape (batch_size, num_labels)

        Process:
        1. Pass through BERT to get [CLS] representation
        2. Apply dropout
        3. Pass through linear classifier
        4. Compute cross-entropy loss (if labels provided)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract pooled output ([CLS] token representation)
        # Shape: (batch_size, hidden_size)
        pooled_output = outputs["pooler_output"]

        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)

        # Apply classification head
        # Shape: (batch_size, num_labels)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Determine loss function based on problem type
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification task
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }


class BertForQuestionAnswering(BertPreTrainedModel):
    """
    BERT model for extractive question answering (e.g., SQuAD).

    In extractive QA:
    - Question: "What is the capital of France?"
    - Context: "Paris is the capital of France. It is a beautiful city."
    - Answer: "Paris" (extracted from context)

    The model predicts the START and END positions of the answer span in the context.

    ARCHITECTURE:
    =============
    Input: [CLS] question [SEP] context [SEP]
    Output: For each token, predict probability of being:
           - Start of answer span
           - End of answer span

    PREDICTION:
    ===========
    For each token position i:
    - start_logits[i] = score for "answer starts at position i"
    - end_logits[i] = score for "answer ends at position i"

    The predicted answer is tokens[start:end+1] where:
    - start = argmax(start_logits)
    - end = argmax(end_logits) with end >= start

    USAGE EXAMPLE:
    ==============
    Input tokens: [CLS] what is the capital of france [SEP] paris is the capital [SEP]
    Positions:     0     1   2  3   4       5  6      7      8    9  10   11      12

    Predictions:
    - start_logits: [..., 0.1, 0.1, 0.1, 0.9, 0.2, ...] → start = 8 ("paris")
    - end_logits:   [..., 0.1, 0.1, 0.1, 0.9, 0.1, ...] → end = 8 ("paris")
    - Answer: "paris"
    """

    def __init__(self, config):
        super().__init__(config)

        # Number of labels is always 2 for QA (start and end)
        self.num_labels = 2

        # Base BERT model (no pooling needed - we use all token representations)
        self.bert = BertModel(config, add_pooling_layer=False)

        # QA head: Linear layer that outputs 2 scores per token
        # Maps from hidden_size (768) to 2 (start and end scores)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for question answering.

        Args:
            input_ids: Token IDs [CLS] question [SEP] context [SEP]
            attention_mask: Attention mask
            token_type_ids: 0 for question, 1 for context
            start_positions: True start positions, shape (batch_size,)
            end_positions: True end positions, shape (batch_size,)

        Returns:
            Dictionary with:
            - loss: QA loss (if positions provided)
            - start_logits: Start position scores, shape (batch_size, seq_length)
            - end_logits: End position scores, shape (batch_size, seq_length)

        Process:
        1. Pass through BERT to get all token representations
        2. Apply QA head to get start/end scores for each token
        3. Compute cross-entropy loss on start and end positions
        """
        # Get BERT outputs (all token representations)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get sequence output (all token representations)
        # Shape: (batch_size, seq_length, hidden_size)
        sequence_output = outputs["last_hidden_state"]

        # Apply QA head to get start and end logits
        # Shape: (batch_size, seq_length, 2)
        logits = self.qa_outputs(sequence_output)

        # Split into start and end logits
        # Each has shape: (batch_size, seq_length)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Compute loss for start and end positions
            loss_fct = CrossEntropyLoss()

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }


__all__ = [
    "BertForPreTraining",
    "BertForMaskedLM",
    "BertForSequenceClassification",
    "BertForQuestionAnswering",
]
