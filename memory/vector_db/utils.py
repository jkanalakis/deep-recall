"""
Utility functions and examples for working with vector databases.
"""

import numpy as np
import os
import time
from typing import List, Dict, Any, Optional

from memory.vector_db import VectorDBFactory
from memory.vector_db.base import VectorDB
from memory.vector_db.faiss_db import FaissVectorDB

try:
    from memory.vector_db.qdrant_db import QdrantVectorDB

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def compare_vector_db_performance(
    dimension: int = 768, num_vectors: int = 1000, num_queries: int = 10, k: int = 5
):
    """
    Compare performance of different vector database implementations.

    Args:
        dimension: Dimension of vectors
        num_vectors: Number of vectors to insert
        num_queries: Number of queries to run
        k: Number of results to return per query
    """
    # Create random vectors
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Create random query vectors
    query_vectors = np.random.random((num_queries, dimension)).astype(np.float32)
    # Normalize query vectors
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / query_norms

    # List of vector database types to test
    db_types = ["faiss"]
    if QDRANT_AVAILABLE:
        db_types.append("qdrant")

    # Dict to store results
    results = {}

    # Test each vector database
    for db_type in db_types:
        print(f"\nTesting {db_type} vector database...")

        # Create vector database
        db = VectorDBFactory.create_db(
            db_type=db_type,
            dimension=dimension,
            index_type="flat" if db_type == "faiss" else None,
            metric="ip" if db_type == "faiss" else "cosine",
            in_memory=True,
        )

        # Measure insertion time
        start_time = time.time()
        db.add(vectors)
        insert_time = time.time() - start_time
        print(f"Insertion time: {insert_time:.4f} seconds")

        # Measure query time
        start_time = time.time()
        similarities, indices = db.search(query_vectors, k=k)
        query_time = time.time() - start_time
        print(f"Query time: {query_time:.4f} seconds")

        # Save results
        results[db_type] = {
            "insert_time": insert_time,
            "query_time": query_time,
            "vectors_per_second": num_vectors / insert_time,
            "queries_per_second": num_queries / query_time,
        }

    # Print comparative results
    print("\nComparative Results:")
    for db_type, result in results.items():
        print(f"{db_type.upper()}:")
        print(f"  - Insertion: {result['vectors_per_second']:.2f} vectors/second")
        print(f"  - Query: {result['queries_per_second']:.2f} queries/second")


def example_faiss_db_usage():
    """Example of using the FAISS vector database."""
    # Create a FAISS vector database
    dimension = 128
    db = FaissVectorDB(
        dimension=dimension,
        index_type="flat",
        metric="ip",  # Inner product (cosine similarity with normalized vectors)
    )

    # Create some sample vectors
    vectors = np.random.random((10, dimension)).astype(np.float32)
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Add vectors to database
    ids = db.add(vectors)
    print(f"Added vectors with IDs: {ids}")

    # Create a query vector
    query_vector = np.random.random((1, dimension)).astype(np.float32)
    # Normalize query vector
    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_vector = query_vector / query_norm

    # Search for similar vectors
    similarities, indices = db.search(query_vector, k=3)
    print("Search results:")
    print(f"  - Similarities: {similarities[0]}")
    print(f"  - Indices: {indices[0]}")

    # Save the database
    db.save("faiss_example")
    print("Database saved to faiss_example/")

    # Create a new database and load from disk
    new_db = FaissVectorDB(dimension=dimension)
    new_db.load("faiss_example")
    print(f"Loaded database with {new_db.get_vector_count()} vectors")


def example_memory_store_with_vector_db():
    """Example of using MemoryStore with different vector databases."""
    from memory.memory_store import MemoryStore

    # Initialize with FAISS
    memory_store_faiss = MemoryStore(
        embedding_dim=128,
        db_type="faiss",
        db_path="faiss_memory_store",
        index_type="flat",
        metric="ip",
    )

    # Create a sample embedding
    embedding = np.random.random(128).astype(np.float32)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    # Add to memory
    memory_id = memory_store_faiss.add_text(
        text="This is a sample text",
        embedding=embedding,
        metadata={"user_id": "user123", "source": "example"},
    )

    print(f"Added memory with ID: {memory_id}")

    # Search with the same embedding
    results = memory_store_faiss.search(query_embedding=embedding, k=1, threshold=0.5)

    print("Search results:")
    for result in results:
        print(f"  - ID: {result['id']}")
        print(f"  - Text: {result['text']}")
        print(f"  - Similarity: {result['similarity']}")
        print(f"  - Metadata: {result['metadata']}")

    # Save to disk
    memory_store_faiss.save_index()
    print("Memory store saved to disk")

    # Get statistics
    stats = memory_store_faiss.get_stats()
    print("Memory store statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    # Run examples
    print("=" * 50)
    print("FAISS Vector Database Example")
    print("=" * 50)
    example_faiss_db_usage()

    print("\n" + "=" * 50)
    print("Memory Store with Vector Database Example")
    print("=" * 50)
    example_memory_store_with_vector_db()

    print("\n" + "=" * 50)
    print("Vector Database Performance Comparison")
    print("=" * 50)
    compare_vector_db_performance(
        dimension=128, num_vectors=10000, num_queries=100, k=5
    )
