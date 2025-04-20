# memory/memory_store.py

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import faiss

from memory.vector_db import VectorDBFactory


class MemoryStore:
    def __init__(
        self,
        embedding_dim: int,
        db_type: str = "faiss",
        db_path: str = "vector_db",
        metadata_path: str = "memory_metadata.json",
        **db_kwargs,
    ):
        """
        Initialize a memory store with vector database integration.

        Args:
            embedding_dim: Dimension of the embedding vectors
            db_type: Type of vector database to use ('faiss', 'qdrant', 'milvus', 'chroma')
            db_path: Path to save/load the vector database
            metadata_path: Path to save/load the metadata JSON
            **db_kwargs: Additional configuration parameters for the vector database
        """
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.last_backup_time = None
        self.backup_interval = 3600  # Default to hourly backups (in seconds)
        self.db_type = db_type

        # Create vector database
        self.vector_db = VectorDBFactory.create_db(
            db_type=db_type, dimension=embedding_dim, **db_kwargs
        )

        # Load the vector database if it exists
        if os.path.exists(self.db_path):
            self.vector_db.load(self.db_path)

        # Load metadata if exists
        self.text_data = {}
        self.metadata = {}
        self.next_id = 0
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from disk if available."""
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

    def add_text(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add text + embedding to the vector database and store text and metadata.

        Args:
            text: The text to store
            embedding: Vector representation of the text
            metadata: Optional dictionary with metadata (user_id, timestamp, etc.)

        Returns:
            int: ID of the stored memory
        """
        # Prepare embedding for vector database
        if len(embedding.shape) == 1:
            # Reshape to match expected format
            vectors = embedding.reshape(1, -1)
        else:
            vectors = embedding

        # Assign ID
        current_id = self.next_id
        self.next_id += 1

        # Add to vector database
        ids = self.vector_db.add(vectors, [current_id])

        # Store text and metadata
        self.text_data[current_id] = text

        # Create or update metadata
        if metadata is None:
            metadata = {}

        # Always add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        self.metadata[current_id] = metadata

        # Consider auto-backup
        self._maybe_backup()

        return current_id

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Return the top-k most similar texts and scores.

        Args:
            query_embedding: Vector representation of query
            k: Number of results to return
            threshold: Similarity threshold (0-1), higher is more similar
            filter_metadata: Optional filter to apply on metadata fields

        Returns:
            List of dictionaries containing results with text, similarity, and metadata
        """
        # Prepare embedding for vector database
        if len(query_embedding.shape) == 1:
            # Reshape to match expected format
            query_vectors = query_embedding.reshape(1, -1)
        else:
            query_vectors = query_embedding

        # Ensure query vector is normalized for cosine similarity
        if self.vector_db.metric == "ip":
            faiss.normalize_L2(query_vectors)

        # Perform search in vector database
        similarities, indices = self.vector_db.search(query_vectors, k=k)

        # Flatten results (we only have one query)
        similarities = similarities[0]
        indices = indices[0]

        results = []
        for similarity, idx in zip(similarities, indices):
            if idx == -1:
                continue

            # Skip results below threshold
            if similarity < threshold:
                continue

            # Apply metadata filtering if provided
            if filter_metadata and not self._matches_filter(idx, filter_metadata):
                continue

            idx = int(idx)  # Ensure index is an integer
            if idx in self.text_data:
                text = self.text_data[idx]
                result_metadata = self.metadata.get(idx, {})

                # Skip deleted items
                if result_metadata.get("deleted", False):
                    continue

                results.append(
                    {
                        "id": idx,
                        "text": text,
                        "similarity": float(similarity),
                        "metadata": result_metadata,
                    }
                )

        return results

    def _matches_filter(self, idx: int, filter_metadata: Dict[str, Any]) -> bool:
        """Check if a memory's metadata matches the filter criteria."""
        item_metadata = self.metadata.get(idx, {})

        for key, value in filter_metadata.items():
            # Special case for timestamp range
            if key == "timestamp_range" and isinstance(value, list) and len(value) == 2:
                if "timestamp" not in item_metadata:
                    return False

                start, end = value
                timestamp = item_metadata["timestamp"]

                if (start and timestamp < start) or (end and timestamp > end):
                    return False
                continue

            # Normal exact matching
            if key not in item_metadata or item_metadata[key] != value:
                return False

        return True

    def save_index(self):
        """Persist the vector database and metadata to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        # Save the vector database
        self.vector_db.save(self.db_path)

        # Save metadata
        self._save_metadata()

        self.last_backup_time = time.time()

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
            self.save_index()

    def set_backup_interval(self, interval_seconds: int):
        """Set the automatic backup interval."""
        self.backup_interval = interval_seconds

    def optimize_index(self):
        """Optimize the vector database index for better performance."""
        return self.vector_db.optimize_index()

    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a specific memory by ID.
        This marks as deleted in metadata but doesn't remove from vector database
        as some vector databases don't support direct removal.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            bool: True if deleted successfully
        """
        if memory_id not in self.text_data:
            return False

        # Mark as deleted in metadata
        if memory_id in self.metadata:
            self.metadata[memory_id]["deleted"] = True
            self.metadata[memory_id]["deletion_timestamp"] = datetime.now().isoformat()

        # Save changes
        self._save_metadata()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            "total_vectors": self.vector_db.get_vector_count(),
            "active_memories": sum(
                1 for m in self.metadata.values() if not m.get("deleted", False)
            ),
            "deleted_memories": sum(
                1 for m in self.metadata.values() if m.get("deleted", False)
            ),
            "embedding_dimension": self.embedding_dim,
            "vector_db_type": self.db_type,
            "last_backup": self.last_backup_time,
        }
