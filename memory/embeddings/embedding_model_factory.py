"""
Embedding Model Factory

This module provides a factory for creating different types of embedding models.
"""

from typing import Optional, Dict, Any
from .base import EmbeddingModel
from .sentence_transformer import SentenceTransformerEmbedding

class EmbeddingModelFactory:
    """Factory for creating embedding models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> EmbeddingModel:
        """
        Create an embedding model based on model_type
        
        Args:
            model_type: Type of model to create (SentenceTransformer, etc.)
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An initialized embedding model
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "SentenceTransformer":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return SentenceTransformerEmbedding(model_name=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 