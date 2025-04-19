"""
Qdrant vector database implementation.
"""

import json
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from memory.vector_db.base import VectorDB

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import (Distance, FieldCondition, Filter,
                                           FilterSelector, MatchAny,
                                           MatchValue, PointStruct,
                                           VectorParams)

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantVectorDB(VectorDB):
    """Vector database implementation using Qdrant."""

    def __init__(
        self,
        dimension: int,
        collection_name: str = "deep_recall",
        metric: str = "cosine",  # "cosine", "euclid" (L2), or "dot" (inner product)
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        in_memory: bool = True,
        on_disk_payload: bool = True,
        timeout: int = 60,
        **kwargs,
    ):
        """
        Initialize a Qdrant vector database.

        Args:
            dimension: Dimension of the vectors to store
            collection_name: Name of the Qdrant collection
            metric: Distance metric to use ('cosine', 'euclid', or 'dot')
            host: Qdrant server host
            port: Qdrant server REST API port
            grpc_port: Qdrant server gRPC port
            in_memory: Whether to use in-memory storage
            on_disk_payload: Whether to store payload on disk
            timeout: Timeout for Qdrant operations in seconds
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client is not available. Please install it with 'pip install qdrant-client'"
            )

        self.dimension = dimension
        self.collection_name = collection_name

        # Map to Qdrant distance metrics
        if metric == "cosine":
            self.metric = Distance.COSINE
        elif metric == "euclid" or metric == "l2":
            self.metric = Distance.EUCLID
        elif metric == "dot" or metric == "ip":
            self.metric = Distance.DOT
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.in_memory = in_memory
        self.on_disk_payload = on_disk_payload
        self.timeout = timeout

        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            prefer_grpc=True,
            timeout=self.timeout,
        )

        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()

        # Store next ID for auto-assignment
        self.next_id = self._get_current_max_id() + 1

    def _create_collection_if_not_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension, distance=self.metric
                    ),
                    on_disk_payload=self.on_disk_payload,
                )

                # Wait for collection to be created
                while True:
                    collections = self.client.get_collections().collections
                    collection_names = [collection.name for collection in collections]
                    if self.collection_name in collection_names:
                        break
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error creating Qdrant collection: {e}")
            raise

    def _get_current_max_id(self) -> int:
        """Get the maximum ID currently in the collection."""
        try:
            # Get collection info
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )

            if not points[0]:  # Empty collection
                return 0

            # Find max ID
            max_id = 0
            for point in points[0]:
                max_id = max(max_id, point.id)

            return max_id
        except Exception as e:
            print(f"Error getting max ID from Qdrant: {e}")
            return 0

    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> List[int]:
        """
        Add vectors to the database.

        Args:
            vectors: Matrix of vectors to add with shape (n_vectors, dim)
            ids: Optional list of IDs to assign to the vectors. If None, IDs will be auto-assigned.

        Returns:
            List of IDs assigned to the vectors
        """
        # Ensure vectors are in the right format
        vectors = vectors.astype(np.float32)
        n_vectors = vectors.shape[0]

        # Get or create IDs
        if ids is None:
            # Auto-assign IDs
            assigned_ids = list(range(self.next_id, self.next_id + n_vectors))
            self.next_id += n_vectors
        else:
            # Verify provided IDs
            if len(ids) != n_vectors:
                raise ValueError(
                    f"Number of IDs ({len(ids)}) doesn't match number of vectors ({n_vectors})"
                )
            assigned_ids = ids
            # Update next_id if necessary
            if ids and max(ids) >= self.next_id:
                self.next_id = max(ids) + 1

        # Create points
        points = []
        for i in range(n_vectors):
            points.append(
                PointStruct(
                    id=assigned_ids[i],
                    vector=vectors[i].tolist(),
                    payload={"created_at": time.time()},
                )
            )

        # Add points to Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

        return assigned_ids

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 5,
        filter_expressions: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the database.

        Args:
            query_vectors: Matrix of query vectors with shape (n_queries, dim)
            k: Number of results to return per query
            filter_expressions: Optional filters to apply to the search

        Returns:
            Tuple containing:
                - similarities: Matrix of similarity scores with shape (n_queries, k)
                - indices: Matrix of vector indices with shape (n_queries, k)
        """
        # Ensure vectors are in the right format
        query_vectors = query_vectors.astype(np.float32)
        n_queries = query_vectors.shape[0]

        # Convert filter expressions to Qdrant filter if provided
        qdrant_filter = None
        if filter_expressions:
            conditions = []
            for field, value in filter_expressions.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=field, match=MatchAny(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )

            qdrant_filter = Filter(must=conditions)

        # Initialize results arrays
        similarities = np.zeros((n_queries, k), dtype=np.float32)
        indices = np.zeros((n_queries, k), dtype=np.int64)

        # Search for each query vector
        for i in range(n_queries):
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vectors[i].tolist(),
                limit=k,
                filter=qdrant_filter,
                with_payload=False,
            )

            # Fill results
            for j, result in enumerate(search_result):
                similarities[i, j] = result.score
                indices[i, j] = result.id

        return similarities, indices

    def delete(self, ids: List[int]) -> bool:
        """
        Delete vectors from the database.

        Args:
            ids: List of vector IDs to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete points from Qdrant
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(points=ids),
            )
            return True
        except Exception as e:
            print(f"Error deleting points from Qdrant: {e}")
            return False

    def save(self, path: str) -> bool:
        """
        Save the vector database to disk.

        Args:
            path: Directory path where to save the database

        Returns:
            True if successful, False otherwise
        """
        if self.in_memory:
            try:
                os.makedirs(path, exist_ok=True)

                # Save next_id
                metadata_path = os.path.join(path, "qdrant_metadata.json")
                metadata = {
                    "next_id": self.next_id,
                    "dimension": self.dimension,
                    "collection_name": self.collection_name,
                    "metric": str(self.metric),
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                # For in-memory Qdrant, we need to snapshot the collection
                # This is implementation-specific and would depend on your setup
                # Here's a placeholder for snapshotting an in-memory collection
                snapshot_path = os.path.join(path, "qdrant_snapshot")
                os.makedirs(snapshot_path, exist_ok=True)

                # Create a snapshot
                snapshot_info = self.client.create_snapshot(
                    collection_name=self.collection_name
                )

                # Download the snapshot (implementation depends on your Qdrant setup)
                # This is a simplified example
                snapshot_file = snapshot_info.name
                self.client.download_snapshot(
                    collection_name=self.collection_name,
                    snapshot_name=snapshot_file,
                    target_path=os.path.join(snapshot_path, snapshot_file),
                )

                return True

            except Exception as e:
                print(f"Error saving Qdrant database: {e}")
                return False
        else:
            # For persistent Qdrant, the data is already saved on disk
            # Just save the metadata
            try:
                os.makedirs(path, exist_ok=True)

                metadata_path = os.path.join(path, "qdrant_metadata.json")
                metadata = {
                    "next_id": self.next_id,
                    "dimension": self.dimension,
                    "collection_name": self.collection_name,
                    "metric": str(self.metric),
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

                return True

            except Exception as e:
                print(f"Error saving Qdrant metadata: {e}")
                return False

    def load(self, path: str) -> bool:
        """
        Load the vector database from disk.

        Args:
            path: Directory path from where to load the database

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load metadata
            metadata_path = os.path.join(path, "qdrant_metadata.json")
            if not os.path.exists(metadata_path):
                return False

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.next_id = metadata.get("next_id", self.next_id)

            # For in-memory Qdrant, we need to restore from snapshot
            if self.in_memory:
                snapshot_path = os.path.join(path, "qdrant_snapshot")

                if os.path.exists(snapshot_path):
                    # Find the snapshot file
                    snapshot_files = os.listdir(snapshot_path)
                    if snapshot_files:
                        snapshot_file = snapshot_files[0]

                        # Upload the snapshot (implementation depends on your Qdrant setup)
                        with open(
                            os.path.join(snapshot_path, snapshot_file), "rb"
                        ) as f:
                            self.client.upload_snapshot(
                                collection_name=self.collection_name, snapshot_file=f
                            )

                        # Recover from snapshot
                        self.client.recover_snapshot(
                            collection_name=self.collection_name,
                            snapshot_name=snapshot_file,
                        )

            return True

        except Exception as e:
            print(f"Error loading Qdrant database: {e}")
            return False

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the database.

        Returns:
            Count of vectors
        """
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return collection_info.vectors_count
        except Exception as e:
            print(f"Error getting vector count from Qdrant: {e}")
            return 0

    def get_dimension(self) -> int:
        """
        Get the dimension of vectors in the database.

        Returns:
            Vector dimension
        """
        return self.dimension

    def optimize_index(self) -> bool:
        """
        Optimize the index for faster queries.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Trigger Qdrant index optimization
            # Note: Qdrant automatically optimizes indices, but we can trigger a manual optimization
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=20000  # Example threshold
                ),
            )
            return True
        except Exception as e:
            print(f"Error optimizing Qdrant index: {e}")
            return False
