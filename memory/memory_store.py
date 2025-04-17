# memory/memory_store.py

import faiss
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Union

class MemoryStore:
    def __init__(self, 
                embedding_dim: int, 
                index_path: str = "vector_index.faiss",
                metadata_path: str = "memory_metadata.json"):
        """
        Initialize a memory store with FAISS vector database integration.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            index_path: Path to save/load the FAISS index
            metadata_path: Path to save/load the metadata JSON
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.last_backup_time = None
        self.backup_interval = 3600  # Default to hourly backups (in seconds)

        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Use IndexFlatIP for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Load metadata if exists
        self.text_data = {}
        self.metadata = {}
        self.next_id = 0
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from disk if available."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.text_data = data.get('text_data', {})
                    # Convert string keys back to integers
                    self.text_data = {int(k): v for k, v in self.text_data.items()}
                    self.metadata = data.get('metadata', {})
                    # Convert string keys back to integers
                    self.metadata = {int(k): v for k, v in self.metadata.items()}
                    self.next_id = data.get('next_id', 0)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading metadata: {e}")
                # Initialize empty if loading fails
                self.text_data = {}
                self.metadata = {}
                self.next_id = 0

    def add_text(self, 
                text: str, 
                embedding: np.ndarray, 
                metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add text + embedding to the index and store text and metadata.
        
        Args:
            text: The text to store
            embedding: Vector representation of the text
            metadata: Optional dictionary with metadata (user_id, timestamp, etc.)
            
        Returns:
            int: ID of the stored memory
        """
        # Normalize vector for cosine similarity
        vec = embedding.astype(np.float32)
        faiss.normalize_L2(vec)
        
        self.index.add(vec)
        current_id = self.next_id
        self.text_data[current_id] = text
        
        # Create or update metadata
        if metadata is None:
            metadata = {}
        
        # Always add timestamp if not provided
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
            
        self.metadata[current_id] = metadata
        self.next_id += 1
        
        # Consider auto-backup
        self._maybe_backup()
        
        return current_id

    def search(self, 
              query_embedding: np.ndarray, 
              k: int = 3,
              threshold: float = 0.0,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
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
        # Normalize vector for cosine similarity
        query_vec = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        # Initial search with more results than needed to allow for filtering
        k_search = max(k * 3, 10) if filter_metadata else k
        
        similarities, indices = self.index.search(query_vec, k_search)
        
        results = []
        for sim_list, idx_list in zip(similarities, indices):
            for similarity, idx in zip(sim_list, idx_list):
                if idx == -1:
                    continue
                
                # Skip results below threshold
                if similarity < threshold:
                    continue
                    
                # Apply metadata filtering if provided
                if filter_metadata and not self._matches_filter(idx, filter_metadata):
                    continue
                
                text = self.text_data[idx]
                result_metadata = self.metadata.get(idx, {})
                
                results.append({
                    "id": int(idx),
                    "text": text,
                    "similarity": float(similarity),
                    "metadata": result_metadata
                })
                
                # Stop once we have enough results
                if len(results) >= k:
                    break
                    
        return results[:k]
    
    def _matches_filter(self, idx: int, filter_metadata: Dict[str, Any]) -> bool:
        """Check if a memory's metadata matches the filter criteria."""
        item_metadata = self.metadata.get(idx, {})
        
        for key, value in filter_metadata.items():
            # Special case for timestamp range
            if key == 'timestamp_range' and isinstance(value, list) and len(value) == 2:
                if 'timestamp' not in item_metadata:
                    return False
                    
                start, end = value
                timestamp = item_metadata['timestamp']
                
                if (start and timestamp < start) or (end and timestamp > end):
                    return False
                continue
                
            # Normal exact matching
            if key not in item_metadata or item_metadata[key] != value:
                return False
                
        return True

    def save_index(self):
        """Persist the FAISS index to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, self.index_path)
        
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
            'text_data': serializable_text_data,
            'metadata': serializable_metadata,
            'next_id': self.next_id
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f)
    
    def _maybe_backup(self):
        """Backup data if enough time has passed since the last backup."""
        current_time = time.time()
        
        if (self.last_backup_time is None or 
            (current_time - self.last_backup_time) > self.backup_interval):
            self.save_index()
    
    def set_backup_interval(self, interval_seconds: int):
        """Set the automatic backup interval."""
        self.backup_interval = interval_seconds
    
    def optimize_index(self):
        """Optimize the FAISS index for better performance."""
        # This is a simplified version - in production you'd want more
        # sophisticated optimization based on dataset size
        
        if self.index.ntotal > 1000:
            # If we have many vectors, convert to IVF index for faster search
            if isinstance(self.index, faiss.IndexFlatIP):
                nlist = min(4096, self.index.ntotal // 10)  # Rule of thumb
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train on existing data
                train_vectors = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
                for i in range(self.index.ntotal):
                    idx_vector = faiss.rev_swig_ptr(self.index.get_xb() + i * self.embedding_dim, self.embedding_dim)
                    train_vectors[i] = idx_vector
                
                new_index.train(train_vectors)
                
                # Add vectors to new index
                new_index.add(train_vectors)
                
                # Replace old index
                self.index = new_index
                
                # Save optimized index
                self.save_index()
                
                return True
        return False
                
    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a specific memory by ID.
        Note: This marks as deleted in metadata but doesn't remove from FAISS index
        as FAISS doesn't support direct removal without rebuild.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if deleted successfully
        """
        if memory_id not in self.text_data:
            return False
            
        # Mark as deleted in metadata
        if memory_id in self.metadata:
            self.metadata[memory_id]['deleted'] = True
            self.metadata[memory_id]['deletion_timestamp'] = datetime.now().isoformat()
        
        # Save changes
        self._save_metadata()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            "total_vectors": self.index.ntotal,
            "active_memories": sum(1 for m in self.metadata.values() if not m.get('deleted', False)),
            "deleted_memories": sum(1 for m in self.metadata.values() if m.get('deleted', False)),
            "embedding_dimension": self.embedding_dim,
            "last_backup": self.last_backup_time
        }
