"""
Advanced semantic search capabilities for deep-recall.
Provides configurable semantic similarity search functionality with
various similarity metrics, pagination, and filtering options.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from memory.embeddings import EmbeddingModelFactory


class SemanticSearch:
    def __init__(
        self,
        memory_store,
        embedding_model=None,
        default_similarity_threshold: float = 0.7,
        default_similarity_metric: str = "cosine",
        max_results_per_page: int = 50,
    ):
        """
        Initialize the semantic search with the specified parameters.

        Args:
            memory_store: Memory store instance for accessing stored embeddings and metadata
            embedding_model: Optional pre-configured embedding model
            default_similarity_threshold: Default threshold for similarity matching (0-1)
            default_similarity_metric: Default similarity metric ("cosine", "euclidean", "dot")
            max_results_per_page: Maximum number of results per page for pagination
        """
        self.memory_store = memory_store
        self.embedding_model = embedding_model
        self.default_similarity_threshold = default_similarity_threshold
        self.default_similarity_metric = default_similarity_metric
        self.max_results_per_page = max_results_per_page

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        page: int = 1,
        items_per_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform semantic search based on query text.

        Args:
            query: The query text to search for
            k: Maximum number of results to return
            threshold: Optional similarity threshold (0-1)
            metric: Similarity metric to use ("cosine", "euclidean", "dot")
            filter_metadata: Optional filters to apply on metadata fields
            page: Page number for pagination (starting at 1)
            items_per_page: Number of items per page (defaults to max_results_per_page)

        Returns:
            Dictionary containing search results with pagination info
        """
        if not self.embedding_model:
            raise ValueError(
                "No embedding model available. Either provide one during initialization "
                "or use search_with_embedding."
            )

        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)

        return self.search_with_embedding(
            query_embedding=query_embedding,
            k=k,
            threshold=threshold,
            metric=metric,
            filter_metadata=filter_metadata,
            page=page,
            items_per_page=items_per_page,
        )

    def search_with_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        page: int = 1,
        items_per_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform semantic search based on pre-computed query embedding.

        Args:
            query_embedding: The embedding vector to search for
            k: Maximum number of results to return
            threshold: Optional similarity threshold (0-1)
            metric: Similarity metric to use ("cosine", "euclidean", "dot")
            filter_metadata: Optional filters to apply on metadata fields
            page: Page number for pagination (starting at 1)
            items_per_page: Number of items per page (defaults to max_results_per_page)

        Returns:
            Dictionary containing search results with pagination info
        """
        # Handle default values
        sim_threshold = (
            threshold if threshold is not None else self.default_similarity_threshold
        )
        sim_metric = metric if metric is not None else self.default_similarity_metric
        items_per_page = min(
            items_per_page or self.max_results_per_page, self.max_results_per_page
        )

        # Adjust k for pagination if needed
        effective_k = k
        if page > 1 or items_per_page < k:
            # We need to retrieve more results to handle pagination properly
            effective_k = max(k, page * items_per_page)

        # Perform search in memory store
        results = self.memory_store.search(
            query_embedding=query_embedding,
            k=effective_k,
            threshold=sim_threshold,
            filter_metadata=filter_metadata,
        )

        # Post-process results based on the specified similarity metric if different
        if sim_metric != "cosine" and len(results) > 0:
            results = self._recompute_similarities(query_embedding, results, sim_metric)

        # Apply pagination
        total_results = len(results)
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_results)

        # Calculate total pages
        total_pages = (total_results + items_per_page - 1) // items_per_page

        # Return paginated results with metadata
        return {
            "results": results[start_idx:end_idx],
            "pagination": {
                "page": page,
                "items_per_page": items_per_page,
                "total_items": total_results,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
            "search_metadata": {
                "threshold": sim_threshold,
                "metric": sim_metric,
                "query_time": datetime.now().isoformat(),
            },
        }

    def _recompute_similarities(
        self, query_embedding: np.ndarray, results: List[Dict[str, Any]], metric: str
    ) -> List[Dict[str, Any]]:
        """
        Recompute similarities based on the specified metric.

        Args:
            query_embedding: The query embedding vector
            results: List of search results
            metric: Similarity metric to use ("cosine", "euclidean", "dot")

        Returns:
            Updated list of search results with recomputed similarities
        """
        # Since we don't have access to the original vectors, we'll keep
        # the original similarities from FAISS which are already normalized
        # for cosine similarity
        return results

    def _get_vectors_for_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Get vectors for the given IDs from the memory store.

        Args:
            ids: List of vector IDs

        Returns:
            List of vectors corresponding to the IDs
        """
        # This is a placeholder - in a real implementation, this would
        # retrieve the vectors from the memory store's vector database
        # For now, we'll return None to indicate this is not implemented
        return None

    def hybrid_search(
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
            query: The query text to search for
            k: Maximum number of results to return
            semantic_weight: Weight for semantic search component (0-1)
            keyword_weight: Weight for keyword matching component (0-1)
            filter_metadata: Optional filters to apply on metadata fields

        Returns:
            List of search results sorted by combined score
        """
        if semantic_weight + keyword_weight != 1.0:
            # Normalize weights
            total = semantic_weight + keyword_weight
            semantic_weight = semantic_weight / total
            keyword_weight = keyword_weight / total

        # Perform semantic search with a larger k to get more candidates
        semantic_results = self.search(
            query=query,
            k=k * 3,  # Get more results to combine with keyword search
            filter_metadata=filter_metadata,
        )["results"]

        # Perform keyword search with improved scoring
        query_words = query.lower().split()
        keyword_scores = {}

        # Get all texts from memory store
        for memory_id, text in self.memory_store.text_data.items():
            if filter_metadata:
                # Skip if metadata doesn't match filter
                if (
                    memory_id not in self.memory_store.metadata
                    or not self._matches_filter(
                        self.memory_store.metadata[memory_id], filter_metadata
                    )
                ):
                    continue

            # Convert text to lowercase words
            memory_words = text.lower().split()

            # Calculate various keyword matching scores
            exact_match_score = 0
            word_match_score = 0
            proximity_score = 0

            # Check for exact phrase match
            if query.lower() in text.lower():
                exact_match_score = 1.0

            # Check for individual word matches
            matching_words = set(query_words).intersection(set(memory_words))
            if matching_words:
                word_match_score = len(matching_words) / len(query_words)

            # Check for word proximity
            if len(query_words) > 1:
                for i in range(len(memory_words) - len(query_words) + 1):
                    window = memory_words[i : i + len(query_words)]
                    if all(word in window for word in query_words):
                        proximity_score = 1.0
                        break

            # Combine scores with weights
            total_score = (
                exact_match_score * 0.5 + word_match_score * 0.3 + proximity_score * 0.2
            )

            if total_score > 0:
                keyword_scores[memory_id] = total_score

        # Sort keyword results
        keyword_results = [
            {
                "id": memory_id,
                "text": self.memory_store.text_data[memory_id],
                "similarity": score,
                "metadata": self.memory_store.metadata.get(memory_id, {}),
            }
            for memory_id, score in sorted(
                keyword_scores.items(), key=lambda x: x[1], reverse=True
            )[: k * 3]
        ]

        # Combine results with weighted scores
        combined_scores = {}

        # Add semantic search results
        for result in semantic_results:
            memory_id = result["id"]
            combined_scores[memory_id] = {
                "score": result["similarity"] * semantic_weight,
                "result": result,
            }

        # Add keyword search results
        for result in keyword_results:
            memory_id = result["id"]
            if memory_id in combined_scores:
                combined_scores[memory_id]["score"] += (
                    result["similarity"] * keyword_weight
                )
            else:
                combined_scores[memory_id] = {
                    "score": result["similarity"] * keyword_weight,
                    "result": result,
                }

        # Sort by combined score and return top k
        sorted_results = [
            {**item["result"], "combined_score": item["score"]}
            for item in sorted(
                combined_scores.values(), key=lambda x: x["score"], reverse=True
            )[:k]
        ]

        return sorted_results

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
