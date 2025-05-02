#!/usr/bin/env python3
"""
Semantic search for Deep Recall.

This module provides semantic search capabilities for finding relevant memories.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

from memory.models import Memory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SemanticSearch:
    """Semantic search for finding relevant memories."""

    def __init__(self, embedding_model, memory_store):
        """
        Initialize the semantic search with embedding model and memory store.

        Args:
            embedding_model: Model for generating text embeddings
            memory_store: Memory store for storing and retrieving memories
        """
        self.embedding_model = embedding_model
        self.memory_store = memory_store
        logger.info("Initialized SemanticSearch")

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Memory]:
        """
        Search for relevant memories based on semantic similarity.

        Args:
            user_id: ID of the user to search memories for
            query: Query text to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            filter_metadata: Optional metadata filters to apply

        Returns:
            List of Memory objects with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search using the memory store's optimized search function
        results = self.memory_store.search_memories(
            user_id=user_id,
            query_vector=query_embedding,
            limit=limit,
            threshold=threshold
        )
        
        # Convert results to Memory objects
        memories = []
        for result in results:
            memory = Memory(
                id=result["id"],
                text=result["text"],
                user_id=result["user_id"],
                created_at=result["created_at"],
                metadata=result["metadata"],
            )
            
            # Set similarity score
            memory.similarity = result.get("similarity", 0.0)
            
            memories.append(memory)
            
        logger.info(f"Found {len(memories)} memories for query: {query[:50]}...")
        return memories

    def index_memory(self, memory: Memory) -> bool:
        """
        Index a memory for future semantic search.

        Args:
            memory: Memory object to index

        Returns:
            True if indexing was successful, False otherwise
        """
        # Generate embedding if not already present
        if not hasattr(memory, "embedding") or memory.embedding is None:
            try:
                memory.embedding = self.embedding_model.embed_text(memory.text)
            except Exception as e:
                logger.error(f"Error generating embedding for memory {memory.id}: {e}")
                return False
        
        # Store the memory with embedding in the memory store
        try:
            self.memory_store.add_memory(memory)
            logger.info(f"Indexed memory {memory.id} for user {memory.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error indexing memory {memory.id}: {e}")
            return False
