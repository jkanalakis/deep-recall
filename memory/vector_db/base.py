"""
Base classes for vector database interfaces.
This module defines the abstract base classes that all vector database implementations must follow.
"""

from abc import ABC, abstractmethod
import numpy as np
import os
from typing import List, Dict, Optional, Any, Tuple, Union


class VectorDB(ABC):
    """Abstract base class for all vector database implementations."""
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> List[int]:
        """
        Add vectors to the database.
        
        Args:
            vectors: Matrix of vectors to add with shape (n_vectors, dim)
            ids: Optional list of IDs to assign to the vectors. If None, IDs will be auto-assigned.
            
        Returns:
            List of IDs assigned to the vectors
        """
        pass
    
    @abstractmethod
    def search(self, 
              query_vectors: np.ndarray, 
              k: int = 5,
              filter_expressions: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_vectors: Matrix of query vectors with shape (n_queries, dim)
            k: Number of results to return per query
            filter_expressions: Optional filters to apply to the search
            
        Returns:
            Tuple containing:
                - similarities: Matrix of similarity scores with shape (n_queries, k)
                - indices: Matrix of vector indices with shape (n_queries, k)
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[int]) -> bool:
        """
        Delete vectors from the database.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save the vector database to disk.
        
        Args:
            path: Directory path where to save the database
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load the vector database from disk.
        
        Args:
            path: Directory path from where to load the database
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the database.
        
        Returns:
            Count of vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of vectors in the database.
        
        Returns:
            Vector dimension
        """
        pass

    @abstractmethod
    def optimize_index(self) -> bool:
        """
        Optimize the index for faster queries.
        
        Returns:
            True if successful, False otherwise
        """
        pass


class VectorDBFactory:
    """Factory class to create vector database instances based on configuration."""
    
    @staticmethod
    def create_db(db_type: str, dimension: int, **kwargs) -> VectorDB:
        """
        Create and return a vector database instance based on the specified type.
        
        Args:
            db_type: Type of vector database to create ('faiss', 'qdrant', 'milvus', 'chroma')
            dimension: Dimension of vectors to be stored
            **kwargs: Additional configuration parameters for the database
            
        Returns:
            VectorDB: An instance of the specified vector database
            
        Raises:
            ValueError: If the specified database type is not supported
        """
        from memory.vector_db.faiss_db import FaissVectorDB
        from memory.vector_db.qdrant_db import QdrantVectorDB
        from memory.vector_db.milvus_db import MilvusVectorDB
        from memory.vector_db.chroma_db import ChromaVectorDB
        
        db_type = db_type.lower()
        
        if db_type == 'faiss':
            return FaissVectorDB(dimension, **kwargs)
        elif db_type == 'qdrant':
            return QdrantVectorDB(dimension, **kwargs)
        elif db_type == 'milvus':
            return MilvusVectorDB(dimension, **kwargs)
        elif db_type == 'chroma':
            return ChromaVectorDB(dimension, **kwargs)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}") 