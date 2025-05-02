# memory/memory_retriever.py

import asyncio
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from memory.embeddings import EmbeddingModel, EmbeddingModelFactory
from memory.semantic_search import SemanticSearch


class MemoryRetriever:
    def __init__(
        self,
        memory_store,
        embedding_model=None,
        model_type: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        **model_kwargs
    ):
        """
        Initialize the memory retriever with a specified embedding model.

        Args:
            memory_store: Vector store for embeddings and text data
            embedding_model: Optional pre-configured embedding model to use
            model_type: Type of embedding model to use ("transformer" or "sentence_transformer")
            model_name: Name of the model to use (from HuggingFace or sentence-transformers)
            **model_kwargs: Additional arguments to pass to the embedding model
        """
        # Use provided embedding model or create one through factory
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            # Create embedding model through factory
            self.model_config = {"model_name": model_name, **model_kwargs}
            self.embedding_model = EmbeddingModelFactory.create_model(
                model_type, **self.model_config
            )

        self.memory_store = memory_store

        # Default similarity threshold
        self.similarity_threshold = 0.7

        # Initialize semantic search component
        self.semantic_search = SemanticSearch(
            memory_store=memory_store,
            embedding_model=self.embedding_model,
            default_similarity_threshold=self.similarity_threshold,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to a vector embedding.

        Args:
            text: The text to embed

        Returns:
            numpy.ndarray: The embedding vector
        """
        return self.embedding_model.embed_text(text)

    async def embed_text_async(self, text: str) -> np.ndarray:
        """
        Asynchronously convert text to a vector embedding.

        Args:
            text: The text to embed

        Returns:
            numpy.ndarray: The embedding vector
        """
        return await self.embedding_model.embed_text_async(text)

    def add_to_memory(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
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

    async def add_to_memory_async(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Asynchronously add text and its metadata to memory.

        Args:
            text: The text to add to memory
            metadata: Additional information like user_id, timestamp, session_id, and tags

        Returns:
            int: The ID of the stored memory
        """
        emb = await self.embed_text_async(text)
        memory_id = self.memory_store.add_text(text, emb, metadata)
        return memory_id

    def get_relevant_memory(
        self,
        query: str,
        k: int = 3,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
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
        sim_threshold = (
            threshold if threshold is not None else self.similarity_threshold
        )
        results = self.memory_store.search(
            query_emb, k=k, threshold=sim_threshold, filter_metadata=filter_metadata
        )
        return results

    async def get_relevant_memory_async(
        self,
        query: str,
        k: int = 3,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Asynchronously return top-k most relevant memories for the query.

        Args:
            query: The text query to find relevant memories for
            k: Number of results to return
            threshold: Optional similarity threshold (0-1), overrides default
            filter_metadata: Optional filter to apply on metadata fields

        Returns:
            List of dictionaries containing relevant memories with text, score, and metadata
        """
        query_emb = await self.embed_text_async(query)
        sim_threshold = (
            threshold if threshold is not None else self.similarity_threshold
        )
        results = self.memory_store.search(
            query_emb, k=k, threshold=sim_threshold, filter_metadata=filter_metadata
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
            self.semantic_search.default_similarity_threshold = threshold
        else:
            raise ValueError("Similarity threshold must be between 0 and 1")

    def search_memory(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None,
        metric: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        page: int = 1,
        items_per_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Advanced semantic search with pagination and different similarity metrics.

        Args:
            query: The text query to find relevant memories for
            k: Maximum number of results to return
            threshold: Optional similarity threshold (0-1), overrides default
            metric: Similarity metric to use ("cosine", "euclidean", "dot")
            filter_metadata: Optional filter to apply on metadata fields
            page: Page number for pagination (starting at 1)
            items_per_page: Number of items per page

        Returns:
            Dictionary containing search results with pagination info
        """
        return self.semantic_search.search(
            query=query,
            k=k,
            threshold=threshold,
            metric=metric,
            filter_metadata=filter_metadata,
            page=page,
            items_per_page=items_per_page,
        )

    def hybrid_search_memory(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic similarity with keyword matching.

        Args:
            query: The text query to search for
            k: Maximum number of results to return
            semantic_weight: Weight for semantic search component (0-1)
            keyword_weight: Weight for keyword matching component (0-1)
            filter_metadata: Optional filter to apply on metadata fields

        Returns:
            List of search results sorted by combined score
        """
        return self.semantic_search.hybrid_search(
            query=query,
            k=k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            filter_metadata=filter_metadata,
        )

    def search_by_metadata(
        self, filter_metadata: Dict[str, Any], query: Optional[str] = None, k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories by metadata with optional semantic filtering.

        Args:
            filter_metadata: Metadata filters to apply (e.g., user_id, tags, timestamp_range)
            query: Optional text query for semantic refinement
            k: Maximum number of results to return

        Returns:
            List of memories matching the metadata criteria
        """
        if query:
            # If query is provided, use semantic search with metadata filtering
            return self.search_memory(
                query=query, k=k, filter_metadata=filter_metadata
            )["results"]

        # If no query, just filter based on metadata
        results = []
        for memory_id, metadata in self.memory_store.metadata.items():
            if self._matches_filter(metadata, filter_metadata):
                if memory_id in self.memory_store.text_data:
                    results.append(
                        {
                            "id": memory_id,
                            "text": self.memory_store.text_data[memory_id],
                            "similarity": 1.0,  # Full match on metadata
                            "metadata": metadata,
                        }
                    )

                    if len(results) >= k:
                        break

        return results

    @staticmethod
    def _matches_filter(
        metadata: Dict[str, Any], filter_metadata: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_metadata.items():
            # Special case for timestamp range
            if key == "timestamp_range" and isinstance(value, list) and len(value) == 2:
                if "timestamp" not in metadata:
                    return False

                start, end = value
                timestamp = metadata["timestamp"]

                if (start and timestamp < start) or (end and timestamp > end):
                    return False
                continue

            # Special case for tags (match any)
            if key == "tags" and isinstance(value, list):
                if "tags" not in metadata or not set(value).intersection(
                    set(metadata["tags"])
                ):
                    return False
                continue

            # Normal exact matching
            if key not in metadata or metadata[key] != value:
                return False

        return True
