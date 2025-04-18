"""
Embeddings module for deep-recall.
This module provides functionality for converting text to vector embeddings.
"""

from memory.embeddings.base import EmbeddingModel, EmbeddingModelFactory
from memory.embeddings.transformer import TransformerEmbeddingModel
from memory.embeddings.sentence_transformer import SentenceTransformerModel

__all__ = [
    'EmbeddingModel',
    'EmbeddingModelFactory',
    'TransformerEmbeddingModel',
    'SentenceTransformerModel',
] 