#!/usr/bin/env python
"""
Example demonstrating semantic search capabilities of the deep-recall framework.
"""

import os
import numpy as np
from datetime import datetime, timedelta

from memory.memory_store import MemoryStore
from memory.memory_retriever import MemoryRetriever

# Create directory for vector db storage if it doesn't exist
os.makedirs("example_data", exist_ok=True)

# Initialize memory store with FAISS vector database
EMBEDDING_DIM = 768  # Dimension for all-MiniLM-L6-v2
memory_store = MemoryStore(
    embedding_dim=EMBEDDING_DIM,
    db_type="faiss",
    db_path="example_data/vector_db",
    metadata_path="example_data/memory_metadata.json",
)

# Initialize memory retriever with SentenceTransformer model
memory_retriever = MemoryRetriever(
    memory_store=memory_store,
    model_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2",  # This is a lightweight model good for examples
)

# Sample conversations for different users
conversations = {
    "user1": [
        "I want to learn about machine learning algorithms.",
        "Tell me more about neural networks and deep learning.",
        "What programming languages are best for AI development?",
        "How can I use TensorFlow for image recognition?",
        "Explain the concept of reinforcement learning.",
    ],
    "user2": [
        "What are good tourist attractions in Paris?",
        "I'd like to visit museums in New York City.",
        "Tell me about hiking trails in the Grand Canyon.",
        "What's the best time to visit Japan?",
        "Recommend some beaches in Hawaii.",
    ],
    "user3": [
        "How do I cook lasagna from scratch?",
        "Share a recipe for chocolate chip cookies.",
        "What's the best way to grill vegetables?",
        "How do I make sourdough bread at home?",
        "Tell me about different types of pasta dishes.",
    ],
}

print("Adding conversation memories to the system...")

# Add all conversations to memory with appropriate metadata
for user_id, texts in conversations.items():
    # Create timestamps starting from 7 days ago, with 1 day increments
    base_time = datetime.now() - timedelta(days=7)

    for i, text in enumerate(texts):
        # Create metadata for this conversation entry
        metadata = {
            "user_id": user_id,
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "session_id": f"{user_id}_session_{i//2}",  # Group every 2 messages as a session
            "tags": ["example"],
        }

        # Add more specific tags based on content
        if (
            "machine" in text.lower()
            or "learning" in text.lower()
            or "ai" in text.lower()
        ):
            metadata["tags"].append("ai")
        elif (
            "tourist" in text.lower()
            or "visit" in text.lower()
            or "travel" in text.lower()
        ):
            metadata["tags"].append("travel")
        elif (
            "cook" in text.lower() or "recipe" in text.lower() or "food" in text.lower()
        ):
            metadata["tags"].append("cooking")

        # Add to memory
        memory_id = memory_retriever.add_to_memory(text, metadata)
        print(f"Added memory {memory_id}: '{text[:50]}...' for {user_id}")

print("\n--- Basic Semantic Search ---")
# Perform basic semantic search
query = "Tell me about artificial intelligence and neural networks"
print(f"Query: '{query}'")

basic_results = memory_retriever.get_relevant_memory(query, k=3)
print(f"Found {len(basic_results)} results:")
for i, result in enumerate(basic_results):
    print(f"{i+1}. '{result['text']}' (score: {result['similarity']:.4f})")

print("\n--- Advanced Semantic Search with Pagination ---")
# Perform advanced semantic search with pagination
advanced_results = memory_retriever.search_memory(
    query="travel destinations in Europe",
    k=5,
    threshold=0.4,
    metric="cosine",
    page=1,
    items_per_page=2,
)

print(f"Query: 'travel destinations in Europe'")
print(
    f"Page {advanced_results['pagination']['page']} of {advanced_results['pagination']['total_pages']}"
)

for i, result in enumerate(advanced_results["results"]):
    print(f"{i+1}. '{result['text']}' (score: {result['similarity']:.4f})")

print("\n--- Hybrid Search (Semantic + Keyword) ---")
# Perform hybrid search combining semantic and keyword matching
hybrid_results = memory_retriever.hybrid_search_memory(
    query="how to cook pasta", k=3, semantic_weight=0.6, keyword_weight=0.4
)

print(f"Query: 'how to cook pasta'")
print(f"Found {len(hybrid_results)} results:")
for i, result in enumerate(hybrid_results):
    print(f"{i+1}. '{result['text']}' (score: {result['combined_score']:.4f})")

print("\n--- Search by Metadata ---")
# Search by metadata only
metadata_results = memory_retriever.search_by_metadata(
    filter_metadata={"user_id": "user3", "tags": ["cooking"]}
)

print(f"Metadata filter: user_id=user3, tags=cooking")
print(f"Found {len(metadata_results)} results:")
for i, result in enumerate(metadata_results):
    print(f"{i+1}. '{result['text']}'")

print("\n--- Combined Metadata and Semantic Search ---")
# Search by metadata with semantic query
combined_results = memory_retriever.search_by_metadata(
    filter_metadata={"user_id": "user1"}, query="TensorFlow and deep learning", k=2
)

print(f"Query: 'TensorFlow and deep learning' for user_id=user1")
print(f"Found {len(combined_results)} results:")
for i, result in enumerate(combined_results):
    print(f"{i+1}. '{result['text']}' (score: {result['similarity']:.4f})")

# Save the memory store to disk
memory_store.save_index()
print("\nMemory store saved to disk.")

print(
    "\nExample completed. Vector database and metadata saved to 'example_data/' directory."
)
