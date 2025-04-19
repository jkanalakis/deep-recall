"""
Benchmarking module for evaluating embedding quality.

This module provides functionality to evaluate the quality of embeddings
using standard benchmarks like STS (Semantic Textual Similarity).
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class EmbeddingBenchmark:
    """
    A class to benchmark embedding quality using standard datasets.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding benchmark.

        Args:
            model_name: The name of the model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Initialized embedding benchmark with model: {model_name}")

    def evaluate_sts_benchmark(
        self, sts_data_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate embedding quality using the STS benchmark.

        Args:
            sts_data_path: Path to STS benchmark data file

        Returns:
            Dictionary with evaluation metrics
        """
        # Use default STS benchmark data if not provided
        if sts_data_path is None:
            # This would typically load from a standard dataset
            # For demonstration, we'll use a small sample
            sentences1 = [
                "The cat sits on the mat.",
                "Dogs are great pets.",
                "The movie was excellent.",
                "I love programming.",
                "The weather is nice today.",
            ]
            sentences2 = [
                "A cat is resting on a rug.",
                "I love dogs.",
                "The film was fantastic.",
                "Coding is my passion.",
                "It's sunny outside.",
            ]
            scores = [0.8, 0.9, 0.95, 0.85, 0.7]  # Ground truth similarity scores
        else:
            # Load STS benchmark data from file
            with open(sts_data_path, "r") as f:
                data = json.load(f)
                sentences1 = [item["sentence1"] for item in data]
                sentences2 = [item["sentence2"] for item in data]
                scores = [item["score"] for item in data]

        # Generate embeddings
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True)

        # Calculate cosine similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()

        # Calculate metrics
        correlation = np.corrcoef(scores, cosine_scores)[0, 1]
        mse = np.mean((scores - cosine_scores) ** 2)
        mae = np.mean(np.abs(scores - cosine_scores))

        results = {
            "correlation": float(correlation),
            "mse": float(mse),
            "mae": float(mae),
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"STS Benchmark Results: {results}")
        return results

    def evaluate_retrieval_accuracy(
        self, queries: List[str], documents: List[str], ground_truth: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy of embeddings.

        Args:
            queries: List of query texts
            documents: List of document texts
            ground_truth: List of indices of relevant documents for each query

        Returns:
            Dictionary with evaluation metrics
        """
        # Generate embeddings
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        doc_embeddings = self.model.encode(documents, convert_to_tensor=True)

        # Calculate similarities
        similarities = util.cos_sim(query_embeddings, doc_embeddings)

        # Get top-k results
        k = 5
        top_k_indices = similarities.argsort(descending=True)[:, :k].cpu().numpy()

        # Calculate metrics
        precision_at_k = []
        recall_at_k = []

        for i, relevant_docs in enumerate(ground_truth):
            retrieved = set(top_k_indices[i])
            relevant = set(relevant_docs)

            if len(relevant) == 0:
                continue

            precision = len(retrieved.intersection(relevant)) / len(retrieved)
            recall = len(retrieved.intersection(relevant)) / len(relevant)

            precision_at_k.append(precision)
            recall_at_k.append(recall)

        avg_precision = np.mean(precision_at_k)
        avg_recall = np.mean(recall_at_k)
        f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        results = {
            "precision@5": float(avg_precision),
            "recall@5": float(avg_recall),
            "f1@5": float(f1_score),
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Retrieval Accuracy Results: {results}")
        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save benchmark results to a file.

        Args:
            results: Dictionary with benchmark results
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load existing results if file exists
        existing_results = []
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = []

        # Append new results
        existing_results.append(results)

        # Save updated results
        with open(output_path, "w") as f:
            json.dump(existing_results, f, indent=2)

        logger.info(f"Saved benchmark results to {output_path}")
