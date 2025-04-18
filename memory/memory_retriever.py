# memory/memory_retriever.py

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
import asyncio

from memory.embeddings import EmbeddingModelFactory, EmbeddingModel

class MemoryRetriever:
    def __init__(
        self, 
        memory_store,
        model_type: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        **model_kwargs
    ):
        """
        Initialize the memory retriever with a specified embedding model.
        
        Args:
            memory_store: Vector store for embeddings and text data
            model_type: Type of embedding model to use ("transformer" or "sentence_transformer")
            model_name: Name of the model to use (from HuggingFace or sentence-transformers)
            **model_kwargs: Additional arguments to pass to the embedding model
        """
        # Create embedding model through factory
        self.model_config = {
            "model_name": model_name,
            **model_kwargs
        }
        self.embedding_model = EmbeddingModelFactory.create_model(
            model_type,
            **self.model_config
        )
        
        self.memory_store = memory_store
        
        # Default similarity threshold
        self.similarity_threshold = 0.7

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to a vector embedding.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        return self.embedding_model.embed_text(text)

    async def embed_text_async(self, text: str) -> np.ndarray:
        """
        Asynchronously convert text to a vector embedding.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        return await self.embedding_model.embed_text_async(text)

    def add_to_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add text and its metadata to memory.
        
        Args:
            text: The text to add to memory
            metadata: Additional information like user_id, timestamp, session_id, and tags
            
        Returns:
            int: The ID of the stored memory
        """
        emb = self.embed_text(text)
        memory_id = self.memory_store.add_text(text, emb, metadata)
        return memory_id
        
    async def add_to_memory_async(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Asynchronously add text and its metadata to memory.
        
        Args:
            text: The text to add to memory
            metadata: Additional information like user_id, timestamp, session_id, and tags
            
        Returns:
            int: The ID of the stored memory
        """
        emb = await self.embed_text_async(text)
        memory_id = self.memory_store.add_text(text, emb, metadata)
        return memory_id

    def get_relevant_memory(self, 
                           query: str, 
                           k: int = 3, 
                           threshold: Optional[float] = None,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Return top-k most relevant memories for the query.
        
        Args:
            query: The text query to find relevant memories for
            k: Number of results to return
            threshold: Optional similarity threshold (0-1), overrides default
            filter_metadata: Optional filter to apply on metadata fields
            
        Returns:
            List of dictionaries containing relevant memories with text, score, and metadata
        """
        query_emb = self.embed_text(query)
        sim_threshold = threshold if threshold is not None else self.similarity_threshold
        results = self.memory_store.search(
            query_emb, 
            k=k,
            threshold=sim_threshold,
            filter_metadata=filter_metadata
        )
        return results
        
    async def get_relevant_memory_async(self, 
                                       query: str, 
                                       k: int = 3, 
                                       threshold: Optional[float] = None,
                                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Asynchronously return top-k most relevant memories for the query.
        
        Args:
            query: The text query to find relevant memories for
            k: Number of results to return
            threshold: Optional similarity threshold (0-1), overrides default
            filter_metadata: Optional filter to apply on metadata fields
            
        Returns:
            List of dictionaries containing relevant memories with text, score, and metadata
        """
        query_emb = await self.embed_text_async(query)
        sim_threshold = threshold if threshold is not None else self.similarity_threshold
        results = self.memory_store.search(
            query_emb, 
            k=k,
            threshold=sim_threshold,
            filter_metadata=filter_metadata
        )
        return results
        
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the default similarity threshold for memory retrieval.
        
        Args:
            threshold: Similarity threshold between 0 and 1
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
        else:
            raise ValueError("Similarity threshold must be between 0 and 1")
