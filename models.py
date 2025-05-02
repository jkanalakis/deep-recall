#!/usr/bin/env python3
"""
Data models for Deep Recall.

This module defines the data models used in the Deep Recall framework.
"""

import enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

class MemoryImportance(enum.Enum):
    """Importance level of a memory."""
    LOW = 0
    NORMAL = 1
    HIGH = 2

@dataclass
class Memory:
    """Represents a memory in the Deep Recall system."""

    def __init__(
        self, 
        id: str, 
        text: str = "", 
        user_id: str = "",
        created_at: str = "",
        importance: MemoryImportance = MemoryImportance.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
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
        """
        self.id = id
        self.text = text
        self.user_id = user_id
        self.created_at = created_at
        self.importance = importance
        self.metadata = metadata or {}
        
        # Additional attributes that may be added later
        self.similarity = None  # For search results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "importance": self.importance.value if isinstance(self.importance, MemoryImportance) else self.importance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create Memory object from dictionary."""
        # Convert importance string to enum if needed
        importance = data.get("importance", MemoryImportance.NORMAL)
        if isinstance(importance, str):
            try:
                importance = MemoryImportance[importance]
            except KeyError:
                importance = MemoryImportance.NORMAL
        
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            user_id=data.get("user_id", ""),
            created_at=data.get("created_at", ""),
            importance=importance,
            metadata=data.get("metadata", {})
        ) 