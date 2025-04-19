"""
Embedding model implementation using sentence-transformers library.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from memory.embeddings.base import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """Embedding model that uses sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """
        Initialize the sentence transformer embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            batch_size: Batch size for processing multiple texts
            normalize_embeddings: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Configure device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Cache embedding dimension
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to vector embeddings.

        Args:
            text: Single text or list of texts to embed

        Returns:
            numpy.ndarray: The embedding vectors with shape (batch_size, embedding_dim)
        """
        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(
            text,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Ensure 2D array even for single input
        if isinstance(text, str):
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def embed_text_async(self, text: Union[str, List[str]]) -> asyncio.Future:
        """
        Asynchronously convert text to vector embeddings.

        Args:
            text: Single text or list of texts to embed

        Returns:
            asyncio.Future: Future object that resolves to numpy.ndarray
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._executor, self.embed_text, text)

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embeddings generated by this model.

        Returns:
            int: Dimension of the embedding vectors
        """
        return self._embedding_dim
