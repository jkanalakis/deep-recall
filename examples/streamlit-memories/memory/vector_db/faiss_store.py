#!/usr/bin/env python3
"""
FAISS vector store implementation for Deep Recall.

This module provides a FAISS-based vector store that integrates with PostgreSQL.
"""

import os
import logging
import faiss
import numpy as np
from typing import Dict, List, Optional, Any, Union
import psycopg2

from app.db_utils import get_db_connection
from memory.vector_db.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""

    def __init__(
        self, 
        dimension: int = 384,
        metric: str = "cosine",
        index_type: str = "flat",
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of the vectors
            metric: Similarity metric ("cosine", "euclidean", "dot")
            index_type: Type of FAISS index ("flat", "hnsw", "ivf")
            db_host: Database host
            db_port: Database port 
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        
        # Initialize the FAISS index
        self._init_index()
        
        # Initialize database connection
        self.conn = None
        self._init_db_connection()
        
        # Local cache of vectors for faster retrieval
        self.vector_cache = {}
        self.metadata_cache = {}
        
        logger.info(f"Initialized FAISSVectorStore with dimension {dimension} and metric {metric}")
        
    def _init_index(self):
        """Initialize the FAISS index."""
        if self.metric == "cosine":
            # For cosine similarity, we need to normalize vectors
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "euclidean":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "dot":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
            
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
                
    def add_item(self, item_id: str, vector: np.ndarray, metadata: Dict = None) -> bool:
        """
        Add an item to the vector store.
        
        Args:
            item_id: ID of the item
            vector: Vector embedding
            metadata: Optional metadata
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Ensure vector is the right shape
            if len(vector.shape) == 1:
                vector = vector.reshape(1, -1)
                
            # Ensure vectors are normalized for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(vector)
                
            # Add to the index
            self.index.add(vector)
            
            # Store in local cache
            numeric_id = hash(item_id) % (2**31)
            self.vector_cache[numeric_id] = vector[0]
            self.metadata_cache[numeric_id] = metadata or {}
            
            return True
        except Exception as e:
            logger.error(f"Error adding item to vector store: {e}")
            return False
            
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        threshold: float = 0.6,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Search for similar items.
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            # Ensure vector is the right shape
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
                
            # Ensure vectors are normalized for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(query_vector)
                
            # Perform search
            similarities, indices = self.index.search(query_vector, limit)
            
            # Process results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                similarity = similarities[0][i]
                if similarity < threshold:
                    continue
                    
                # Get metadata
                numeric_id = idx
                metadata = self.metadata_cache.get(numeric_id, {})
                
                # Apply metadata filter if provided
                if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                    continue
                    
                results.append({
                    "id": metadata.get("item_id", str(numeric_id)),
                    "similarity": float(similarity),
                    "metadata": metadata
                })
                
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
            
    def _matches_filter(self, metadata: Dict, filter_metadata: Dict) -> bool:
        """
        Check if metadata matches the filter criteria.
        
        Args:
            metadata: Item metadata
            filter_metadata: Filter criteria
            
        Returns:
            True if matches, False otherwise
        """
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
                
            if metadata[key] != value:
                return False
                
        return True
        
    def delete(self, item_id: str) -> bool:
        """
        Delete an item from the vector store.
        
        Args:
            item_id: ID of the item
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            # FAISS doesn't support direct deletion, so we would need to rebuild the index
            # For simplicity, we'll just remove from our local cache
            numeric_id = hash(item_id) % (2**31)
            if numeric_id in self.vector_cache:
                del self.vector_cache[numeric_id]
            if numeric_id in self.metadata_cache:
                del self.metadata_cache[numeric_id]
                
            return True
        except Exception as e:
            logger.error(f"Error deleting item from vector store: {e}")
            return False
            
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the index
            faiss.write_index(self.index, path)
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
            
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load the index from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Vector store file does not exist: {path}")
                return False
                
            # Load the index
            self.index = faiss.read_index(path)
            
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False 