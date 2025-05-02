#!/usr/bin/env python3
"""
Vector store base class for Deep Recall.

This module defines the base interface for vector stores used by the memory system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import numpy as np

class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_item(self, item_id: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add an item to the vector store.
        
        Args:
            item_id: ID of the item
            vector: Vector embedding
            metadata: Optional metadata
            
        Returns:
            True if added successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        threshold: float = 0.6,
        filter_metadata: Optional[Dict] = None
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
        pass
        
    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """
        Delete an item from the vector store.
        
        Args:
            item_id: ID of the item
            
        Returns:
            True if deleted, False otherwise
        """
        pass
        
    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            True if saved successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        pass 