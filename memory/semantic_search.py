#!/usr/bin/env python3
"""
Semantic search capabilities for Deep Recall.

This module provides semantic similarity search functionality for retrieving memories.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.models import Memory

class SemanticSearch:
    def __init__(
        self,
        vector_store,
        embedding_model=None,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        """
        Initialize the semantic search with the specified parameters.

        Args:
            vector_store: Vector store instance for searching embeddings
            embedding_model: Optional pre-configured embedding model
            model_name: Name of the sentence transformer model to use
            dimension: Embedding dimension
        """
        self.vector_store = vector_store
        self.dimension = dimension
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            self.embedding_model = SentenceTransformer(model_name)
        else:
            self.embedding_model = embedding_model

    def index_memory(self, memory: Memory) -> bool:
        """
        Index a memory for semantic search.
        
        Args:
            memory: Memory object to index
            
        Returns:
            Success status
        """
        # Generate embedding for the memory text
        embedding = self.embed_text(memory.text)
        
        # Store in vector store
        return self.vector_store.store_embedding(memory.id, embedding, memory.metadata)
    
    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        threshold: float = 0.6,
    ) -> List[Memory]:
        """
        Perform semantic search based on query text.

        Args:
            user_id: User ID to filter results
            query: The query text to search for
            limit: Maximum number of results to return
            threshold: Similarity threshold (0-1)

        Returns:
            List of Memory objects with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Search in vector store
        filter_expr = {"user_id": user_id}
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit, 
            threshold=threshold,
            filter_expr=filter_expr
        )
        
        # Format results as Memory objects
        memories = []
        for result in results:
            # Create a memory object with similarity score
            memory_id = result["id"]
            # We don't have the original memory data in the vector store results
            # This is a simplified implementation
            memory = Memory(
                id=memory_id,
                text="", # Placeholder
                user_id=user_id,
                created_at=datetime.now().isoformat(),
            )
            
            # Add similarity score as an attribute
            memory.similarity = result["score"]
            memories.append(memory)
            
        return memories
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text, normalize_embeddings=True)
