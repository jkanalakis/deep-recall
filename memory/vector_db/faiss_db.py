"""
FAISS vector database implementation.
"""

import os
import faiss
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
import json

from memory.vector_db.base import VectorDB


class FaissVectorDB(VectorDB):
    """Vector database implementation using FAISS."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "ip",  # "ip" for inner product (cosine), "l2" for Euclidean distance
        nlist: int = 100,  # For IVF indices
        nprobe: int = 10,  # For IVF indices, higher=more accurate but slower
        **kwargs,
    ):
        """
        Initialize a FAISS vector database.

        Args:
            dimension: Dimension of the vectors to store
            index_type: Type of FAISS index to use ('flat', 'ivf', 'hnsw')
            metric: Distance metric to use ('ip' for inner product/cosine, 'l2' for Euclidean)
            nlist: Number of clusters for IVF indices
            nprobe: Number of clusters to visit during search for IVF indices
        """
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.nlist = nlist
        self.nprobe = nprobe
        self.next_id = 0
        self.id_map = {}  # Maps internal FAISS indices to external IDs
        self.index = self._create_index()

    def _create_index(self):
        """Create a FAISS index based on the specified parameters."""
        if self.metric == "ip":
            # Inner product for cosine similarity (vectors should be normalized)
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            # L2 (Euclidean) distance
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric type: {self.metric}")

        if self.index_type == "flat":
            # Simple flat index - exact but slow for large datasets
            if self.metric == "ip":
                return faiss.IndexFlatIP(self.dimension)
            else:
                return faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "ivf":
            # IVF index - faster but approximate
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, metric_type
            )
            # Index must be trained before use
            if not index.is_trained:
                # Need some data to train, will be populated at first add()
                pass
            return index

        elif self.index_type == "hnsw":
            # HNSW index - fast and accurate, good for medium datasets
            index = faiss.IndexHNSWFlat(
                self.dimension, 32, metric_type
            )  # 32 is M parameter in HNSW
            return index

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

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

        # Normalize vectors if using inner product distance
        if self.metric == "ip":
            faiss.normalize_L2(vectors)

        # Check if IVF index needs training
        if self.index_type == "ivf" and not self.index.is_trained and n_vectors > 0:
            self.index.train(vectors)

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

        # Track current index size before adding
        start_idx = self.get_vector_count()

        # Add vectors to FAISS
        self.index.add(vectors)

        # Map external IDs to FAISS internal indices
        for i, external_id in enumerate(assigned_ids):
            faiss_idx = start_idx + i
            self.id_map[faiss_idx] = external_id

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
            filter_expressions: Filters not supported in base FAISS

        Returns:
            Tuple containing:
                - similarities: Matrix of similarity scores with shape (n_queries, k)
                - indices: Matrix of vector indices with shape (n_queries, k)
        """
        # Ensure vectors are in the right format
        query_vectors = query_vectors.astype(np.float32)

        # Normalize query vectors if using inner product distance
        if self.metric == "ip":
            faiss.normalize_L2(query_vectors)

        # Set nprobe for IVF indices (higher = more accurate but slower)
        if self.index_type == "ivf":
            self.index.nprobe = self.nprobe

        # Perform search
        k = min(k, self.get_vector_count())  # Can't retrieve more than what exists
        if k == 0:
            n_queries = query_vectors.shape[0]
            return np.zeros((n_queries, 0), dtype=np.float32), np.zeros(
                (n_queries, 0), dtype=np.int64
            )

        similarities, faiss_indices = self.index.search(query_vectors, k)

        # Convert FAISS internal indices to external IDs
        external_indices = np.zeros_like(faiss_indices)
        for i in range(faiss_indices.shape[0]):
            for j in range(faiss_indices.shape[1]):
                faiss_idx = faiss_indices[i, j]
                if faiss_idx != -1 and faiss_idx in self.id_map:
                    external_indices[i, j] = self.id_map[faiss_idx]
                else:
                    external_indices[i, j] = -1

        return similarities, external_indices

    def delete(self, ids: List[int]) -> bool:
        """
        Delete vectors from the database.

        Args:
            ids: List of vector IDs to delete

        Returns:
            True if successful, False otherwise
        """
        # Basic FAISS doesn't support direct removal, need to rebuild index
        # Find FAISS internal indices that correspond to the external IDs
        faiss_indices_to_remove = []
        for faiss_idx, external_id in self.id_map.items():
            if external_id in ids:
                faiss_indices_to_remove.append(faiss_idx)

        if not faiss_indices_to_remove:
            return False  # None of the ids were found

        # For FAISS implementations that support direct removal (e.g., IndexIVFFlatDedup)
        # However, most FAISS indices don't support this, so we would need to rebuild
        # Here we just mark as removed in our id_map
        for faiss_idx in faiss_indices_to_remove:
            del self.id_map[faiss_idx]

        return True

    def save(self, path: str) -> bool:
        """
        Save the vector database to disk.

        Args:
            path: Directory path where to save the database

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(path, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)

            # Save metadata (id_map and next_id)
            metadata_path = os.path.join(path, "faiss_metadata.json")
            metadata = {
                "next_id": self.next_id,
                "id_map": {
                    str(k): v for k, v in self.id_map.items()
                },  # Convert keys to strings for JSON
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            return True

        except Exception as e:
            print(f"Error saving FAISS database: {e}")
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
            # Load FAISS index
            index_path = os.path.join(path, "faiss_index.bin")
            if not os.path.exists(index_path):
                return False

            self.index = faiss.read_index(index_path)

            # Load metadata
            metadata_path = os.path.join(path, "faiss_metadata.json")
            if not os.path.exists(metadata_path):
                return False

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.next_id = metadata.get("next_id", 0)
            self.id_map = {
                int(k): v for k, v in metadata.get("id_map", {}).items()
            }  # Convert keys back to ints
            self.dimension = metadata.get("dimension", self.dimension)
            self.index_type = metadata.get("index_type", self.index_type)
            self.metric = metadata.get("metric", self.metric)
            self.nlist = metadata.get("nlist", self.nlist)
            self.nprobe = metadata.get("nprobe", self.nprobe)

            # Set nprobe for IVF indices
            if self.index_type == "ivf":
                self.index.nprobe = self.nprobe

            return True

        except Exception as e:
            print(f"Error loading FAISS database: {e}")
            return False

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the database.

        Returns:
            Count of vectors
        """
        return self.index.ntotal

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
            # For flat indices, conversion to IVF can help with large datasets
            if self.index_type == "flat" and self.get_vector_count() > 10000:
                print("Converting flat index to IVF for better performance...")

                # Create a new IVF index
                quantizer = faiss.IndexFlatL2(self.dimension)
                if self.metric == "ip":
                    new_index = faiss.IndexIVFFlat(
                        quantizer,
                        self.dimension,
                        self.nlist,
                        faiss.METRIC_INNER_PRODUCT,
                    )
                else:
                    new_index = faiss.IndexIVFFlat(
                        quantizer, self.dimension, self.nlist, faiss.METRIC_L2
                    )

                # Extract and train on existing vectors
                if self.get_vector_count() > 0:
                    # We need to extract all vectors from the current index
                    # This is inefficient but necessary for the conversion
                    vectors = np.zeros(
                        (self.get_vector_count(), self.dimension), dtype=np.float32
                    )
                    for i in range(self.get_vector_count()):
                        vector = faiss.vector_float_to_array(
                            faiss.extract_index_vector(self.index, i)
                        )
                        vectors[i] = vector

                    # Train and add to new index
                    new_index.train(vectors)
                    new_index.add(vectors)

                    # Replace old index with new one
                    self.index = new_index
                    self.index_type = "ivf"

                    return True

            return False  # No optimization needed or possible

        except Exception as e:
            print(f"Error optimizing FAISS index: {e}")
            return False
