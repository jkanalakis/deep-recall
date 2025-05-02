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
        try:
            # Reshape if needed to make it a 2D array
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
                
            # Add to vector database
            id_int = int(hash(str(id)) % (2**31))  # Convert ID to string first, then hash to ensure consistency
            added = self.vector_db.add(embedding, [id_int])
            
            # Store the id mapping for retrieval
            # TODO: In a production system, this would be stored in a database
            if not hasattr(self, '_id_mapping'):
                self._id_mapping = {}
            self._id_mapping[id_int] = id
            
            return True
        except Exception as e:
            import logging
            logging.error(f"Error storing embedding: {e}")
            return False
    
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
        try:
            # Check if database is empty first
            if self.vector_db.get_vector_count() == 0:
                import logging
                logging.warning("Vector store is empty, returning empty results")
                return []
                
            # Reshape if needed to make it a 2D array
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Search in vector database
            scores, indices = self.vector_db.search(query_embedding, limit, filter_expr)
            
            # Prepare results
            results = []
            for i in range(indices.shape[1]):
                if indices[0, i] != -1 and scores[0, i] >= threshold:
                    # Convert integer ID back to original string ID (unhash if possible)
                    # But since we can't reverse the hash, we need to store a mapping
                    # For now, we'll just ensure the ID is properly formatted
                    id_val = str(indices[0, i])
                    
                    # Try to retrieve the original string ID from our mapping
                    if hasattr(self, '_id_mapping') and indices[0, i] in self._id_mapping:
                        id_val = self._id_mapping[indices[0, i]]
                    
                    # Calculate a proper similarity score that's normalized between 0 and 1
                    # Inner product scores (cosine similarity) should be between -1 and 1
                    # Scale to 0-1 range for consistency
                    similarity = 0.0
                    if hasattr(self.vector_db, 'metric') and self.vector_db.metric == "ip":
                        # For inner product, higher is better, normalize properly
                        # Cosine similarity ranges from -1 to 1, rescale to 0-1
                        raw_score = float(scores[0, i])
                        # Adjust the normalization to avoid always returning 1.0
                        # Assuming vectors are normalized, cosine similarity should be in [-1, 1]
                        # Scale to [0, 1] range
                        similarity = (raw_score + 1) / 2
                    else:
                        # For L2 distance, lower is better, so convert to similarity
                        distance = float(scores[0, i])
                        # Using an exponential decay to get more meaningful distinctions
                        # This gives a more gradual falloff than 1/(1+d)
                        similarity = float(np.exp(-distance))
                    
                    results.append({
                        "id": id_val,
                        "score": similarity,
                    })
            
            return results
        except Exception as e:
            import logging
            logging.error(f"Error searching vector store: {e}")
            return []
    
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