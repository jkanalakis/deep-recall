#!/usr/bin/env python3
"""
Data models for Deep Recall.

This module defines the data models used in the Deep Recall framework.
"""

import enum
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

class MemoryImportance(enum.Enum):
    """Importance level of a memory."""
    LOW = 0
    NORMAL = 1
    HIGH = 2

class Memory:
    """Represents a memory in the Deep Recall system."""

    def __init__(
        self, 
        id: str = None, 
        text: str = "", 
        user_id: str = "",
        created_at: str = None,
        importance: MemoryImportance = MemoryImportance.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ):
        """
        Initialize a memory.
        
        Args:
            id: Unique identifier for the memory
            text: Text content of the memory
            user_id: ID of the user who owns this memory
            created_at: Timestamp when the memory was created
            importance: Importance level of the memory
            metadata: Additional metadata for the memory
            embedding: Vector embedding of the memory text
        """
        self.id = id or str(uuid.uuid4())
        self.text = text
        self.user_id = user_id
        self.created_at = created_at or datetime.now().isoformat()
        self.importance = importance
        self.metadata = metadata or {}
        self.embedding = embedding
        
        # Additional attributes that may be added later
        self.similarity = None  # For search results
        self.embedding_id = None  # ID of the embedding in the database

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "importance": self.importance.value if isinstance(self.importance, MemoryImportance) else self.importance,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "similarity": self.similarity,
            "embedding_id": self.embedding_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create Memory object from dictionary."""
        # Convert importance string or value to enum if needed
        importance = data.get('importance', MemoryImportance.NORMAL)
        if isinstance(importance, str):
            try:
                importance = MemoryImportance[importance]
            except KeyError:
                importance = MemoryImportance.NORMAL
        elif isinstance(importance, int):
            try:
                importance = MemoryImportance(importance)
            except ValueError:
                importance = MemoryImportance.NORMAL
        
        # Convert embedding list to numpy array if needed
        embedding = data.get('embedding')
        if embedding is not None and not isinstance(embedding, np.ndarray) and isinstance(embedding, (list, tuple)):
            embedding = np.array(embedding)
        
        mem = cls(
            id=data.get('id'),
            text=data.get('text', ''),
            user_id=data.get('user_id', ''),
            created_at=data.get('created_at'),
            importance=importance,
            metadata=data.get('metadata', {}),
            embedding=embedding
        )
        
        # Add additional fields
        mem.similarity = data.get('similarity')
        mem.embedding_id = data.get('embedding_id')
        
        return mem 