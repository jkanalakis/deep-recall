"""
Summarization module for handling long conversation contexts.

This module provides functionality to summarize long conversation histories
to fit within token limits for LLM input.
"""

import logging
from typing import Any, Dict, List, Optional

from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class Summarizer:
    """
    A class to handle text summarization for long conversation contexts.
    """

    def __init__(
        self, model_name: str = "facebook/bart-large-cnn", device: str = "cuda"
    ):
        """
        Initialize the summarizer with a specific model.

        Args:
            model_name: The name of the model to use for summarization
            device: The device to run the model on (cuda or cpu)
        """
        try:
            self.summarizer = pipeline("summarization", model=model_name, device=device)
            logger.info(f"Initialized summarizer with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {str(e)}")
            # Fallback to CPU if CUDA is not available
            if device == "cuda":
                logger.warning("Falling back to CPU for summarization")
                self.summarizer = pipeline(
                    "summarization", model=model_name, device="cpu"
                )
            else:
                raise

    def summarize_text(
        self, text: str, max_length: int = 150, min_length: int = 40
    ) -> str:
        """
        Summarize a text to fit within token limits.

        Args:
            text: The text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary

        Returns:
            The summarized text
        """
        if not text or len(text.split()) < min_length:
            return text

        try:
            # Split text if it's too long for the model
            if len(text.split()) > 1024:
                chunks = self._split_text(text, max_chunk_size=1024)
                summaries = []
                for chunk in chunks:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                    )
                    summaries.append(summary[0]["summary_text"])
                return " ".join(summaries)
            else:
                summary = self.summarizer(
                    text, max_length=max_length, min_length=min_length, do_sample=False
                )
                return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            # Return original text if summarization fails
            return text

    def _split_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        """
        Split text into chunks of approximately equal size.

        Args:
            text: The text to split
            max_chunk_size: Maximum size of each chunk in words

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_chunk_size):
            chunk = " ".join(words[i : i + max_chunk_size])
            chunks.append(chunk)

        return chunks


class ContextManager:
    """
    A class to manage conversation context and prepare it for LLM input.
    """

    def __init__(self, summarizer: Optional[Summarizer] = None):
        """
        Initialize the context manager.

        Args:
            summarizer: An optional summarizer instance
        """
        self.summarizer = summarizer or Summarizer()

    def prepare_context_for_llm(
        self, conversation_history: List[Dict[str, Any]], token_limit: int = 1024
    ) -> str:
        """
        Prepare conversation history for LLM input, summarizing if necessary.

        Args:
            conversation_history: List of conversation turns
            token_limit: Maximum number of tokens for the context

        Returns:
            Formatted context ready for LLM input
        """
        # Extract text from conversation history
        context_text = ""
        for turn in conversation_history:
            if "user" in turn:
                context_text += f"User: {turn['user']}\n"
            if "assistant" in turn:
                context_text += f"Assistant: {turn['assistant']}\n"

        # Rough estimate of tokens (words + some overhead)
        estimated_tokens = len(context_text.split()) * 1.3

        if estimated_tokens > token_limit:
            logger.info(
                f"Context exceeds token limit ({estimated_tokens} > {token_limit}), summarizing"
            )
            summarized_context = self.summarizer.summarize_text(context_text)
            return summarized_context

        return context_text
