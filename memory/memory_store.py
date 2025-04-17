# memory/memory_store.py

import faiss
import numpy as np
import os
from typing import List, Dict

class MemoryStore:
    def __init__(self, embedding_dim: int, index_path: str = "vector_index.faiss"):
        self.index_path = index_path
        self.embedding_dim = embedding_dim

        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        # A simple in-memory store for IDs -> text
        self.text_data = {}
        self.next_id = 0

    def add_text(self, text: str, embedding: np.ndarray):
        """Add text + embedding to the index and store text in a dictionary."""
        vec = embedding.astype(np.float32)
        self.index.add(vec)
        current_id = self.next_id
        self.text_data[current_id] = text
        self.next_id += 1
        return current_id

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """Return the top-k most similar texts and scores."""
        query_vec = query_embedding.astype(np.float32)
        distances, indices = self.index.search(query_vec, k)
        results = []
        for dist_list, idx_list in zip(distances, indices):
            for dist, idx in zip(dist_list, idx_list):
                if idx == -1:
                    continue
                text = self.text_data[idx]
                results.append({
                    "id": idx,
                    "text": text,
                    "distance": float(dist)
                })
        return results

    def save_index(self):
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, self.index_path)
