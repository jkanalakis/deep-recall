"""
FAISS vector store implementation.
This module provides a FAISS-based implementation of the vector store interface.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import os

from memory.vector_db.vector_store import VectorStore
from memory.vector_db.faiss_db import FaissVectorDB


class FAISSVectorStore(VectorStore):
    """FAISS-based implementation of the vector store interface."""

    def __init__(self, dimension: int = 384, index_type: str = "flat", metric: str = "ip"):
        """
        Initialize a FAISS vector store.
        
        Args:
            dimension: Dimension of the vectors to store
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('ip' for inner product/cosine, 'l2' for Euclidean)
        """
        # Initialize FAISS vector database
        faiss_db = FaissVectorDB(
            dimension=dimension,
            index_type=index_type,
            metric=metric
        )
        
        # Initialize base VectorStore with FAISS database
        super().__init__(vector_db=faiss_db, dimension=dimension)
        
        # Additional FAISS-specific properties
        self.index_type = index_type
        self.metric = metric
    
    def optimize(self) -> bool:
        """
        Optimize the FAISS index for better performance.
        
        Returns:
            True if optimization succeeded, False otherwise
        """
        return self.vector_db.optimize_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats (count, dimension, etc)
        """
        return {
            "vector_count": self.vector_db.get_vector_count(),
            "dimension": self.vector_db.get_dimension(),
            "index_type": self.index_type,
            "metric": self.metric
        } 