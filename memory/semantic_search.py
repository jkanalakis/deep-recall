"""
Advanced semantic search capabilities for deep-recall.
Provides configurable semantic similarity search functionality with
various similarity metrics, pagination, and filtering options.
"""

import heapq
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

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
                "No embedding model available. Either provide one during initialization or use search_with_embedding."
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

        # Post-process results based on the specified similarity metric if different from default
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
        # Extract ids and similarities for processing
        result_ids = [result["id"] for result in results]

        # This would require access to the original vectors from memory store
        # Here we assume we can access them through a hypothetical method
        vectors = self._get_vectors_for_ids(result_ids)

        # Compute new similarities based on metric
        new_similarities = []
        for vec in vectors:
            if metric == "cosine":
                # Cosine similarity
                norm_q = np.linalg.norm(query_embedding)
                norm_v = np.linalg.norm(vec)
                similarity = (
                    np.dot(query_embedding, vec) / (norm_q * norm_v)
                    if norm_q * norm_v > 0
                    else 0
                )
            elif metric == "euclidean":
                # Convert Euclidean distance to similarity (1 / (1 + distance))
                distance = np.linalg.norm(query_embedding - vec)
                similarity = 1 / (1 + distance)
            elif metric == "dot":
                # Dot product similarity
                similarity = np.dot(query_embedding, vec)
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
            new_similarities.append(similarity)

        # Update results with new similarities
        for i, result in enumerate(results):
            result["similarity"] = float(new_similarities[i])

        # Re-sort based on new similarities
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results

    def _get_vectors_for_ids(self, ids: List[int]) -> List[np.ndarray]:
        """
        Retrieve the original vectors for the given ids.
        This is a placeholder method that would need to be implemented
        based on how vectors are stored and accessed in the memory store.

        Args:
            ids: List of vector ids to retrieve

        Returns:
            List of vector embeddings corresponding to the ids
        """
        # This is a placeholder. In a real implementation, this would
        # retrieve the actual vectors from the vector database
        return [np.zeros(self.memory_store.embedding_dim) for _ in ids]

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

        # Perform semantic search
        semantic_results = self.search(
            query=query,
            k=k * 2,  # Get more results to combine with keyword search
            filter_metadata=filter_metadata,
        )["results"]

        # Perform keyword search (basic implementation)
        query_words = set(query.lower().split())
        keyword_scores = {}

        # This is a simplified keyword search that would need to be replaced
        # with a proper text search implementation (e.g., using SQL or Elasticsearch)
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

            # Simple TF scoring - count matching words and normalize
            memory_words = set(text.lower().split())
            matching_words = query_words.intersection(memory_words)

            if matching_words:
                score = len(matching_words) / max(len(query_words), 1)
                keyword_scores[memory_id] = score

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
            )[: k * 2]
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
