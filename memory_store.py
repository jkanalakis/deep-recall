#!/usr/bin/env python3
"""
Memory store for Deep Recall.

This module provides a memory store for managing text data, embeddings, and metadata.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

class MemoryStore:
    def __init__(
        self,
        embedding_dim: int = 384,
        metadata_path: str = "memory_metadata.json",
    ):
        """
        Initialize a memory store.

        Args:
            embedding_dim: Dimension of the embedding vectors
            metadata_path: Path to save/load the metadata JSON
        """
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.last_backup_time = None
        self.backup_interval = 3600  # Default to hourly backups (in seconds)

        # Initialize empty data structures
        self.text_data = {}
        self.metadata = {}
        self.next_id = 0

        # Load existing data if available
        self._load_existing_data()

    def _load_existing_data(self):
        """Load metadata if it exists."""
        # Load metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                    self.text_data = data.get("text_data", {})
                    # Convert string keys back to integers
                    self.text_data = {int(k): v for k, v in self.text_data.items()}
                    self.metadata = data.get("metadata", {})
                    # Convert string keys back to integers
                    self.metadata = {int(k): v for k, v in self.metadata.items()}
                    self.next_id = data.get("next_id", 0)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading metadata: {e}")
                # Initialize empty if loading fails
                self.text_data = {}
                self.metadata = {}
                self.next_id = 0

    def add_memory(self, memory, text=None):
        """
        Add a memory to the store.

        Args:
            memory: Memory object to store
            text: Optional text to use if memory.text is not set

        Returns:
            ID of the stored memory
        """
        # Handle case where memory might not have text attribute or text is empty
        if not hasattr(memory, 'text') or memory.text is None or memory.text == "":
            if text:
                memory.text = text
            else:
                memory.text = ""  # Ensure text is at least empty string, not None
                
        # Store text and metadata
        numeric_id = hash(memory.id) % (2**31)
        self.text_data[numeric_id] = memory.text

        # Create metadata from memory object
        metadata = memory.metadata.copy() if memory.metadata else {}
        metadata["user_id"] = memory.user_id
        metadata["created_at"] = memory.created_at
        metadata["importance"] = memory.importance.value if hasattr(memory.importance, "value") else memory.importance

        self.metadata[numeric_id] = metadata

        # Consider auto-backup
        self._maybe_backup()

        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            Dict with memory data or None if not found
        """
        numeric_id = hash(memory_id) % (2**31)
        if numeric_id not in self.text_data:
            return None

        metadata = self.metadata.get(numeric_id, {})
        
        # Skip deleted items
        if metadata.get("deleted", False):
            return None

        return {
            "id": memory_id,
            "text": self.text_data[numeric_id],
            "metadata": metadata
        }

    def get_memories_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of memory dictionaries
        """
        results = []
        for numeric_id, text in self.text_data.items():
            metadata = self.metadata.get(numeric_id, {})
            
            # Skip deleted items
            if metadata.get("deleted", False):
                continue
                
            # Only include memories for this user
            if metadata.get("user_id") != user_id:
                continue
                
            results.append({
                "id": str(numeric_id),  # Convert back to string
                "text": text,
                "metadata": metadata
            })
            
        return results

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            bool: True if deleted successfully
        """
        numeric_id = hash(memory_id) % (2**31)
        if numeric_id not in self.text_data:
            return False

        # Mark as deleted in metadata
        if numeric_id in self.metadata:
            self.metadata[numeric_id]["deleted"] = True
            self.metadata[numeric_id]["deletion_timestamp"] = datetime.now().isoformat()

        # Save changes
        self._save_metadata()
        return True

    def _save_metadata(self):
        """Save metadata to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.metadata_path)), exist_ok=True)

        # Convert int keys to strings for JSON serialization
        serializable_text_data = {str(k): v for k, v in self.text_data.items()}
        serializable_metadata = {str(k): v for k, v in self.metadata.items()}

        data = {
            "text_data": serializable_text_data,
            "metadata": serializable_metadata,
            "next_id": self.next_id,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(data, f)

    def _maybe_backup(self):
        """Backup data if enough time has passed since the last backup."""
        current_time = time.time()

        if (
            self.last_backup_time is None
            or (current_time - self.last_backup_time) > self.backup_interval
        ):
            self._save_metadata()
            self.last_backup_time = current_time

    def set_backup_interval(self, interval_seconds: int):
        """Set the automatic backup interval."""
        self.backup_interval = interval_seconds

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            "active_memories": sum(
                1 for m in self.metadata.values() if not m.get("deleted", False)
            ),
            "deleted_memories": sum(
                1 for m in self.metadata.values() if m.get("deleted", False)
            ),
            "embedding_dimension": self.embedding_dim,
            "last_backup": self.last_backup_time,
        } 