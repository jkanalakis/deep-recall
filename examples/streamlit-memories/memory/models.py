#!/usr/bin/env python3
"""
Memory models for Deep Recall.

This module defines the memory models used by the memory system.
"""

import enum
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np

class MemoryImportance(enum.Enum):
    """Importance levels for memories."""
    
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"

class Memory:
    """Memory object for storing and retrieving information."""
    
    def __init__(
        self,
        id: str = None,
        user_id: str = None,
        text: str = None,
        embedding: np.ndarray = None,
        importance: Union[MemoryImportance, str] = MemoryImportance.NORMAL,
        created_at: str = None,
        metadata: Dict = None,
    ):
        """
        Initialize a memory object.
        
        Args:
            id: Unique identifier for the memory
            user_id: User identifier
            text: Memory text content
            embedding: Vector embedding of the text
            importance: Importance level of the memory
            created_at: Creation timestamp
            metadata: Additional metadata
        """
        self.id = id or str(uuid.uuid4())
        self.user_id = user_id
        self.text = text
        self.embedding = embedding
        
        # Handle importance as either enum or string
        if isinstance(importance, str):
            try:
                self.importance = MemoryImportance[importance]
            except KeyError:
                self.importance = MemoryImportance.NORMAL
        else:
            self.importance = importance
            
        self.created_at = created_at or datetime.now().isoformat()
        self.metadata = metadata or {}
        
        # This will be set when retrieved from search
        self.similarity = None
        
    def to_dict(self) -> Dict:
        """
        Convert the memory to a dictionary.
        
        Returns:
            Dictionary representation of the memory
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "importance": self.importance.value if isinstance(self.importance, MemoryImportance) else self.importance,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "similarity": self.similarity
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Memory':
        """
        Create a memory from a dictionary.
        
        Args:
            data: Dictionary containing memory data
            
        Returns:
            Memory object
        """
        return cls(
            id=data.get("id"),
            user_id=data.get("user_id"),
            text=data.get("text"),
            importance=data.get("importance", MemoryImportance.NORMAL),
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {})
        ) 