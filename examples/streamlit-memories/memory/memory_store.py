#!/usr/bin/env python3
"""
Memory store for Deep Recall.

This module provides a memory store for managing text data, embeddings, and metadata.
"""

import os
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor

from app.db_utils import (
    get_db_connection, 
    store_embedding, 
    store_memory, 
    search_memories, 
    get_memory, 
    delete_memory, 
    get_all_memories_for_user
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MemoryStore:
    def __init__(
        self,
        embedding_dim: int = 384,
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
    ):
        """
        Initialize a memory store.

        Args:
            embedding_dim: Dimension of the embedding vectors
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.embedding_dim = embedding_dim
        
        # Initialize database connection
        self.conn = None
        self._init_db_connection()
        
        # Initialize empty fallback data structures (used if DB connection fails)
        self.text_data = {}
        self.metadata = {}
        self.next_id = 0

    def _init_db_connection(self):
        """Initialize the database connection."""
        try:
            self.conn = get_db_connection()
            logger.info("Connected to database: recall_memories_db")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self.conn = None

    def _ensure_connection(self):
        """Ensure database connection is active, reconnect if needed."""
        if self.conn is None:
            self._init_db_connection()
        else:
            try:
                # Check if connection is alive
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception:
                # Reconnect if connection is lost
                logger.info("Database connection lost, reconnecting...")
                self._init_db_connection()

    def add_memory(self, memory):
        """
        Add a memory to the store.

        Args:
            memory: Memory object to store

        Returns:
            ID of the stored memory
        """
        # First try to store in database
        stored_in_db = self._store_memory_in_db(memory)
        
        # Also maintain in-memory cache for fast access
        numeric_id = hash(memory.id) % (2**31)
        self.text_data[numeric_id] = memory.text

        # Create metadata from memory object
        metadata = memory.metadata.copy() if memory.metadata else {}
        metadata["user_id"] = memory.user_id
        metadata["created_at"] = memory.created_at
        metadata["importance"] = memory.importance.value if hasattr(memory.importance, "value") else memory.importance

        self.metadata[numeric_id] = metadata
        
        logger.info(f"Memory {memory.id} stored{'in database' if stored_in_db else ' (in-memory only)'}")
        return memory.id
    
    def _store_memory_in_db(self, memory) -> bool:
        """
        Store a memory in the PostgreSQL database.
        
        Args:
            memory: Memory object to store
            
        Returns:
            True if successfully stored in database, False otherwise
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot store memory in database: No connection")
            return False
            
        try:
            # First check if the memory has an embedding
            if not hasattr(memory, 'embedding') or memory.embedding is None:
                logger.warning(f"Memory {memory.id} has no embedding, cannot store in database")
                return False
            
            # Format the embedding vector for pgvector
            vector_str = "[" + ",".join(str(x) for x in memory.embedding) + "]"
            
            # Store the embedding
            embedding_id = store_embedding(self.conn, vector_str)
            if not embedding_id:
                logger.error("Failed to store embedding")
                return False
            
            # Store the memory with the embedding ID
            memory_id = store_memory(
                self.conn, 
                memory.user_id, 
                memory.text, 
                memory.metadata or {}, 
                embedding_id
            )
            
            if not memory_id:
                logger.error("Failed to store memory")
                return False
                
            return True
                
        except Exception as e:
            logger.error(f"Error storing memory in database: {e}")
            return False

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            Dict with memory data or None if not found
        """
        # First try to get from database
        db_memory = self._get_memory_from_db(memory_id)
        if db_memory:
            return db_memory
            
        # Fall back to in-memory data
        numeric_id = hash(memory_id) % (2**31)
        if numeric_id not in self.text_data:
            return None

        metadata = self.metadata.get(numeric_id, {})
        
        # Skip deleted items
        if metadata.get("deleted", False):
            return None

        return {
            "id": memory_id,
            "text": self.text_data[numeric_id],
            "metadata": metadata
        }
    
    def _get_memory_from_db(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory directly from the database.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Memory dict if found, None otherwise
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot retrieve memory from database: No connection")
            return None
            
        try:
            memory = get_memory(self.conn, int(memory_id))
            
            if memory:
                # Update local cache for future fast access
                numeric_id = hash(memory_id) % (2**31)
                self.text_data[numeric_id] = memory["text"]
                self.metadata[numeric_id] = memory["metadata"]
                
                return memory
                
            return None
                
        except Exception as e:
            logger.error(f"Error retrieving memory from database: {e}")
            return None

    def get_memories_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of memory dictionaries
        """
        # Try to get from database first
        db_memories = self._get_user_memories_from_db(user_id)
        if db_memories is not None:
            return db_memories
            
        # Fall back to in-memory data
        results = []
        for numeric_id, text in self.text_data.items():
            metadata = self.metadata.get(numeric_id, {})
            
            # Skip deleted items
            if metadata.get("deleted", False):
                continue
                
            # Only include memories for this user
            if metadata.get("user_id") != user_id:
                continue
                
            results.append({
                "id": str(numeric_id),  # Convert back to string
                "text": text,
                "metadata": metadata
            })
            
        return results
    
    def _get_user_memories_from_db(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve all memories for a user from the database.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of memory dictionaries or None if retrieval failed
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot retrieve user memories from database: No connection")
            return None
            
        try:
            memories = get_all_memories_for_user(self.conn, user_id)
            
            # Update local cache
            for memory in memories:
                numeric_id = hash(str(memory["id"])) % (2**31)
                self.text_data[numeric_id] = memory["text"]
                self.metadata[numeric_id] = memory["metadata"]
                
            return memories
                
        except Exception as e:
            logger.error(f"Error retrieving user memories from database: {e}")
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            bool: True if deleted successfully
        """
        # Try to delete in database
        db_deleted = self._delete_memory_in_db(memory_id)
        
        # Also update local cache
        numeric_id = hash(memory_id) % (2**31)
        if numeric_id not in self.text_data:
            return db_deleted or False  # Return db result if available

        # Mark as deleted in metadata
        if numeric_id in self.metadata:
            self.metadata[numeric_id]["deleted"] = True
            self.metadata[numeric_id]["deletion_timestamp"] = datetime.now().isoformat()

        return True
    
    def _delete_memory_in_db(self, memory_id: str) -> bool:
        """
        Delete a memory from the database.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            True if deleted, False otherwise
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot delete memory from database: No connection")
            return False
            
        try:
            return delete_memory(self.conn, int(memory_id))
                
        except Exception as e:
            logger.error(f"Error deleting memory from database: {e}")
            return False
