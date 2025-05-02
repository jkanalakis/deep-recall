"""
Vector store interface for memory storage.
This module provides an abstraction over different vector database implementations.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np

from memory.vector_db.base import VectorDB


class VectorStore:
    """Interface for memory vector storage."""

    def __init__(self, vector_db: VectorDB = None, dimension: int = 384):
        """
        Initialize a vector store.
        
        Args:
            vector_db: A VectorDB instance or None
            dimension: Vector dimension if no vector_db is provided
        """
        self.vector_db = vector_db
        
        # If no vector_db provided, create a FAISS vector store as default
        if self.vector_db is None:
            from memory.vector_db.faiss_db import FaissVectorDB
            self.vector_db = FaissVectorDB(dimension=dimension)
    
    def store_embedding(self, 
                        id: str, 
                        embedding: np.ndarray, 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store an embedding vector with optional metadata.
        
        Args:
            id: Unique identifier for the embedding
            embedding: The embedding vector
            metadata: Optional metadata to associate with the embedding
            
        Returns:
            True if successfully stored, False otherwise
        """
        # Reshape if needed to make it a 2D array
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
            
        # Add to vector database
        id_int = hash(id) % (2**31)  # Convert string ID to integer
        self.vector_db.add(embedding, [id_int])
        return True
    
    def search(self, 
               query_embedding: np.ndarray, 
               limit: int = 5, 
               threshold: float = 0.0,
               filter_expr: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filter_expr: Optional filter expression
            
        Returns:
            List of match results
        """
        # Reshape if needed to make it a 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search in vector database
        scores, indices = self.vector_db.search(query_embedding, limit, filter_expr)
        
        # Prepare results
        results = []
        for i in range(indices.shape[1]):
            if indices[0, i] != -1 and scores[0, i] >= threshold:
                results.append({
                    "id": str(indices[0, i]),
                    "score": float(scores[0, i]),
                })
        
        return results
    
    def delete(self, id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            id: Unique identifier for the embedding
            
        Returns:
            True if successfully deleted, False otherwise
        """
        id_int = hash(id) % (2**31)  # Convert string ID to integer
        return self.vector_db.delete([id_int])
    
    def save(self, path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store
            
        Returns:
            True if successfully saved, False otherwise
        """
        return self.vector_db.save(path)
    
    def load(self, path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load the vector store from
            
        Returns:
            True if successfully loaded, False otherwise
        """
        return self.vector_db.load(path) 