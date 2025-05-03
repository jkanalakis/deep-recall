#!/usr/bin/env python3
"""
Semantic search for Deep Recall.

This module provides semantic search capabilities for the memory system.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.db_utils import search_memories
from memory.vector_db.vector_store import VectorStore
from memory.models import Memory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SemanticSearch:
    """Semantic search for finding relevant memories."""

    def __init__(
        self, 
        vector_store: VectorStore,
        model_name: str = "BAAI/bge-base-en-v1.5"
    ):
        """
        Initialize the semantic search.
        
        Args:
            vector_store: Vector store for storing and retrieving embeddings
            model_name: Name of the sentence transformer model to use
        """
        self.vector_store = vector_store
        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to a vector embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Vector embedding
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
            
        # Generate embedding
        return self.model.encode(text)

    def index_memory(self, memory: Memory) -> bool:
        """
        Index a memory for search.
        
        Args:
            memory: Memory object to index
            
        Returns:
            True if indexed successfully, False otherwise
        """
        try:
            # Generate embedding if not already present
            if not hasattr(memory, "embedding") or memory.embedding is None:
                memory.embedding = self.encode(memory.text)
                
            # Add to vector store
            self.vector_store.add_item(
                item_id=memory.id,
                vector=memory.embedding,
                metadata={
                    "user_id": memory.user_id,
                    "text": memory.text,
                    "created_at": memory.created_at
                }
            )
            
            logger.info(f"Indexed memory {memory.id} for user {memory.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error indexing memory: {e}")
            return False

    def search(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5, 
        threshold: float = 0.2
    ) -> List[Memory]:
        """
        Search for relevant memories.
        
        Args:
            user_id: User ID to search memories for
            query: Query text
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory objects
        """
        try:
            # Generate query embedding
            query_embedding = self.encode(query)
            
            # First try database search using pgvector
            db_results = self._search_db(query_embedding, user_id, limit, threshold)
            if db_results:
                return db_results
                
            # Fall back to vector store
            results = self.vector_store.search(
                query_vector=query_embedding,
                limit=limit,
                threshold=threshold,
                filter_metadata={"user_id": user_id}
            )
            
            # Convert to Memory objects
            memories = []
            for result in results:
                memory = Memory(
                    id=result.get("id", str(uuid.uuid4())),
                    user_id=result.get("metadata", {}).get("user_id", user_id),
                    text=result.get("metadata", {}).get("text", ""),
                    created_at=result.get("metadata", {}).get("created_at", ""),
                    metadata=result.get("metadata", {})
                )
                # Add similarity score
                memory.similarity = result.get("similarity", 0.0)
                memories.append(memory)
                
            logger.info(f"Found {len(memories)} memories for user {user_id} matching query")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
            
    def _search_db(
        self, 
        query_embedding: np.ndarray, 
        user_id: str, 
        limit: int, 
        threshold: float
    ) -> Optional[List[Memory]]:
        """
        Search for memories in the PostgreSQL database.
        
        Args:
            query_embedding: Query embedding vector
            user_id: User ID to filter memories
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory objects or None if search failed
        """
        # Connect to database through vector store
        conn = getattr(self.vector_store, "conn", None)
        if not conn:
            logger.warning("No database connection available for search")
            return None
            
        try:
            # Format vector for pgvector
            vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # Search using the database function
            results = search_memories(conn, vector_str, user_id, limit, threshold)
            
            # Convert to Memory objects
            memories = []
            for result in results:
                memory = Memory(
                    id=str(result["id"]),
                    user_id=result["user_id"],
                    text=result["text"],
                    created_at=result.get("created_at", ""),
                    metadata=result.get("metadata", {})
                )
                # Add similarity score
                memory.similarity = result.get("similarity", 0.0)
                memories.append(memory)
                
            logger.info(f"Found {len(memories)} memories in database for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories in database: {e}")
            return None
