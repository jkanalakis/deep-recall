# memory/memory_retriever.py

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class MemoryRetriever:
    def __init__(self, memory_store):
        # Any HuggingFace or other embedding model
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.memory_store = memory_store

    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to a vector embedding."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # A simple way to get sentence embedding is to mean-pool the token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings.reshape(1, -1)  # FAISS expects (batch, dim)

    def add_to_memory(self, text: str):
        emb = self.embed_text(text)
        self.memory_store.add_text(text, emb)

    def get_relevant_memory(self, query: str, k: int = 3):
        """Return top-k most relevant memories for the query."""
        query_emb = self.embed_text(query)
        results = self.memory_store.search(query_emb, k)
        return results
