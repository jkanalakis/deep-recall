"""
Utility functions and examples for working with embedding models.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

import numpy as np

from memory.embeddings import (EmbeddingModelFactory, SentenceTransformerModel,
                               TransformerEmbeddingModel)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        float: Cosine similarity score (between -1 and 1, higher means more similar)
    """
    # Ensure vectors are flattened
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero norm
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


async def batch_embed_texts(
    texts: List[str],
    model_type: str = "sentence_transformer",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    **model_kwargs,
) -> np.ndarray:
    """
    Asynchronously embed a batch of texts.

    Args:
        texts: List of texts to embed
        model_type: Type of embedding model to use
        model_name: Name of the model to use
        batch_size: Size of batches to process
        **model_kwargs: Additional model configuration

    Returns:
        np.ndarray: Matrix of embeddings with shape (len(texts), embedding_dim)
    """
    # Create embedding model
    model = EmbeddingModelFactory.create_model(
        model_type, model_name=model_name, batch_size=batch_size, **model_kwargs
    )

    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = await model.embed_text_async(batch)
        all_embeddings.append(batch_embeddings)

    # Concatenate all batches
    return np.vstack(all_embeddings)


def example_transformer_embeddings():
    """Example of using a transformer model for embeddings."""
    # Create model
    model = TransformerEmbeddingModel(
        model_name="bert-base-uncased", pooling_strategy="mean"
    )

    # Generate embeddings
    texts = [
        "Hello, how are you?",
        "I'm doing well, thank you!",
        "The weather is nice today.",
    ]

    embeddings = model.embed_text(texts)
    print(f"Generated {len(texts)} embeddings with dimension {embeddings.shape[1]}")

    # Compute similarity between first two texts
    sim = compute_cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between texts 1 and 2: {sim:.4f}")


def example_sentence_transformer_embeddings():
    """Example of using a sentence-transformer model for embeddings."""
    # Create model
    model = SentenceTransformerModel(
        model_name="all-MiniLM-L6-v2", normalize_embeddings=True
    )

    # Generate embeddings
    texts = [
        "Hello, how are you?",
        "I'm doing well, thank you!",
        "This sentence is completely unrelated to the conversation.",
    ]

    embeddings = model.embed_text(texts)
    print(f"Generated {len(texts)} embeddings with dimension {embeddings.shape[1]}")

    # Compute similarity between texts
    sim1_2 = compute_cosine_similarity(embeddings[0], embeddings[1])
    sim1_3 = compute_cosine_similarity(embeddings[0], embeddings[2])

    print(f"Similarity between related texts: {sim1_2:.4f}")
    print(f"Similarity between unrelated texts: {sim1_3:.4f}")


async def example_async_embedding():
    """Example of asynchronous embedding."""
    model = SentenceTransformerModel(model_name="all-MiniLM-L6-v2")

    # Create multiple embedding tasks
    texts = ["First text", "Second text", "Third text"]
    tasks = [model.embed_text_async(text) for text in texts]

    # Wait for all tasks to complete
    embeddings = await asyncio.gather(*tasks)

    print(f"Asynchronously generated {len(embeddings)} embeddings")

    # Convert list of embeddings to numpy array
    embeddings_array = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")


if __name__ == "__main__":
    # Run synchronous examples
    print("Transformer example:")
    example_transformer_embeddings()

    print("\nSentence-Transformer example:")
    example_sentence_transformer_embeddings()

    # Run async example
    print("\nAsync example:")
    asyncio.run(example_async_embedding())
