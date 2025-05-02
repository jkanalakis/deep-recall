#!/usr/bin/env python3
"""
Memory Service for Deep Recall.

This module provides a service layer for managing memories in the Deep Recall framework.
It integrates with the memory store and semantic search components.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from memory.semantic_search import SemanticSearch
from memory.memory_store import MemoryStore
from memory.embeddings.embedding_model import EmbeddingModel
from memory.models import Memory, MemoryImportance

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing memories in Deep Recall."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        memory_store: Optional[MemoryStore] = None,
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
    ):
        """
        Initialize the memory service.
        
        Args:
            embedding_model: Model for generating text embeddings
            memory_store: Optional pre-configured memory store
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        # Initialize memory store if not provided
        if memory_store is None:
            self.memory_store = MemoryStore(
                embedding_dim=embedding_model.dimension,
                db_host=db_host,
                db_port=db_port,
                db_name=db_name,
                db_user=db_user,
                db_password=db_password,
            )
        else:
            self.memory_store = memory_store
            
        # Initialize embedding model and semantic search
        self.embedding_model = embedding_model
        self.semantic_search = SemanticSearch(
            embedding_model=embedding_model,
            memory_store=self.memory_store
        )
        
        logger.info(f"Initialized MemoryService with {embedding_model.__class__.__name__}")

    def store_memory(self, memory: Memory) -> str:
        """
        Store a memory.
        
        Args:
            memory: Memory object to store
            
        Returns:
            ID of the stored memory
        """
        # Generate embedding for the memory if not already present
        if not hasattr(memory, 'embedding') or memory.embedding is None:
            memory.embedding = self.embedding_model.embed_text(memory.text)
        
        # Store the memory
        memory_id = self.memory_store.add_memory(memory)
        
        logger.info(f"Stored memory for user {memory.user_id}: {memory.text[:50]}...")
        return memory_id

    def retrieve_memories(
        self, user_id: str, query: str, limit: int = 5, threshold: float = 0.7
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
        memory_data = self.memory_store.get_memory(memory_id)
        if memory_data:
            return Memory.from_dict(memory_data)
        return None

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            True if deleted, False otherwise
        """
        return self.memory_store.delete_memory(memory_id)

    def get_user_memories(self, user_id: str) -> List[Memory]:
        """
        Get all memories for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of memory objects
        """
        memory_data_list = self.memory_store.get_memories_for_user(user_id)
        return [Memory.from_dict(memory_data) for memory_data in memory_data_list] 