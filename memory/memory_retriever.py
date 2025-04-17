# memory/memory_retriever.py

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Optional, Union, Any

class MemoryRetriever:
    def __init__(self, memory_store, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the memory retriever with a specified embedding model.
        
        Args:
            memory_store: Vector store for embeddings and text data
            model_name: HuggingFace model name for embeddings
        """
        # Allow configurable embedding model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.memory_store = memory_store
        
        # Default similarity threshold
        self.similarity_threshold = 0.7

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to a vector embedding.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # A simple way to get sentence embedding is to mean-pool the token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings.reshape(1, -1)  # FAISS expects (batch, dim)

    def add_to_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add text and its metadata to memory.
        
        Args:
            text: The text to add to memory
            metadata: Additional information like user_id, timestamp, session_id, and tags
            
        Returns:
            int: The ID of the stored memory
        """
        emb = self.embed_text(text)
        memory_id = self.memory_store.add_text(text, emb, metadata)
        return memory_id

    def get_relevant_memory(self, 
                           query: str, 
                           k: int = 3, 
                           threshold: Optional[float] = None,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Return top-k most relevant memories for the query.
        
        Args:
            query: The text query to find relevant memories for
            k: Number of results to return
            threshold: Optional similarity threshold (0-1), overrides default
            filter_metadata: Optional filter to apply on metadata fields
            
        Returns:
            List of dictionaries containing relevant memories with text, score, and metadata
        """
        query_emb = self.embed_text(query)
        sim_threshold = threshold if threshold is not None else self.similarity_threshold
        results = self.memory_store.search(
            query_emb, 
            k=k,
            threshold=sim_threshold,
            filter_metadata=filter_metadata
        )
        return results
        
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the default similarity threshold for memory retrieval.
        
        Args:
            threshold: Similarity threshold between 0 and 1
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
        else:
            raise ValueError("Similarity threshold must be between 0 and 1")
