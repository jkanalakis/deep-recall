#!/usr/bin/env python3
"""
Memory Service for Deep Recall.

This module provides a service layer for managing memories in the Deep Recall framework.
It integrates with the vector store and embedding components.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from memory.semantic_search import SemanticSearch
from memory.memory_store import MemoryStore
from memory.vector_db.vector_store import VectorStore
from memory.models import Memory, MemoryImportance

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing memories in Deep Recall."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the memory service.
        
        Args:
            vector_store: Vector store for storing and retrieving embeddings
        """
        self.memory_store = MemoryStore()
        self.semantic_search = SemanticSearch(vector_store=vector_store)
        self.vector_store = vector_store
        logger.info(f"Initialized MemoryService with {vector_store.__class__.__name__}")

    def store_memory(self, memory: Memory) -> str:
        """
        Store a memory.
        
        Args:
            memory: Memory object to store
            
        Returns:
            ID of the stored memory
        """
        # Store the memory in the local store
        self.memory_store.add_memory(memory)
        
        # Add to vector store for semantic search
        self.semantic_search.index_memory(memory)
        
        logger.info(f"Stored memory for user {memory.user_id}: {memory.text[:50]}...")
        return memory.id

    def retrieve_memories(
        self, user_id: str, query: str, limit: int = 5, threshold: float = 0.6
    ) -> List[Memory]:
        """
        Retrieve memories for a user based on semantic similarity to a query.
        
        Args:
            user_id: ID of the user
            query: Query to match against memories
            limit: Maximum number of memories to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory objects with similarity scores
        """
        # Search for relevant memories
        relevant_memories = self.semantic_search.search(
            user_id=user_id, 
            query=query,
            limit=limit,
            threshold=threshold
        )
        
        logger.info(f"Retrieved {len(relevant_memories)} memories for user {user_id} matching query: {query[:50]}...")
        return relevant_memories

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Memory object if found, None otherwise
        """
        return self.memory_store.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            True if deleted, False otherwise
        """
        memory = self.memory_store.get_memory(memory_id)
        if not memory:
            logger.warning(f"Attempt to delete non-existent memory: {memory_id}")
            return False
            
        # Remove from memory store
        self.memory_store.delete_memory(memory_id)
        
        # Remove from vector store
        # Note: This assumes vector_store supports deletion by memory_id
        if hasattr(self.vector_store, 'delete'):
            self.vector_store.delete(memory_id)
            
        logger.info(f"Deleted memory {memory_id}")
        return True

    def get_user_memories(self, user_id: str) -> List[Memory]:
        """
        Get all memories for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of memory objects
        """
        return self.memory_store.get_memories_for_user(user_id) 