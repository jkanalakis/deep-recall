#!/usr/bin/env python3
"""
Script to run benchmarks and generate reports.

This script runs the embedding and inference benchmarks and generates
reports with the results.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.benchmarking.embedding_benchmarks import EmbeddingBenchmark
from tests.benchmarking.inference_benchmarks import InferenceBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)

logger = logging.getLogger(__name__)


async def run_embedding_benchmarks(output_dir: str):
    """
    Run embedding benchmarks and save results.

    Args:
        output_dir: Directory to save results
    """
    logger.info("Running embedding benchmarks...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize benchmark
    benchmark = EmbeddingBenchmark()

    # Run STS benchmark
    sts_results = benchmark.evaluate_sts_benchmark()
    benchmark.save_results(
        sts_results, os.path.join(output_dir, "sts_benchmark_results.json")
    )

    # Run retrieval accuracy benchmark
    # This is a simplified example - in a real scenario, you would use a proper dataset
    queries = [
        "What is machine learning?",
        "How does neural networks work?",
        "Explain natural language processing",
    ]

    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data.",
        "Neural networks are computing systems inspired by the biological neural networks in human brains.",
        "Natural language processing is a field of artificial intelligence that focuses on the interaction between computers and human language.",
        "Deep learning is a subset of machine learning that uses multiple layers of neural networks.",
        "Supervised learning is a type of machine learning where the model is trained on labeled data.",
    ]

    # Ground truth: indices of relevant documents for each query
    ground_truth = [
        [0, 3],  # Documents 0 and 3 are relevant for query 0
        [1, 3],  # Documents 1 and 3 are relevant for query 1
        [2, 4],  # Documents 2 and 4 are relevant for query 2
    ]

    retrieval_results = benchmark.evaluate_retrieval_accuracy(
        queries, documents, ground_truth
    )
    benchmark.save_results(
        retrieval_results, os.path.join(output_dir, "retrieval_benchmark_results.json")
    )

    logger.info("Embedding benchmarks completed")


async def run_inference_benchmarks(
    output_dir: str, api_url: str, auth_token: str = None
):
    """
    Run inference benchmarks and save results.

    Args:
        output_dir: Directory to save results
        api_url: URL of the inference API
        auth_token: Optional authentication token
    """
    logger.info("Running inference benchmarks...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize benchmark
    benchmark = InferenceBenchmark(api_url, auth_token)

    # Run latency benchmark
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
    ]

    latency_results = await benchmark.benchmark_latency(
        prompts, batch_size=1, num_runs=5
    )
    benchmark.save_results(
        latency_results, os.path.join(output_dir, "latency_benchmark_results.json")
    )

    # Run throughput benchmark
    prompt = "What is the capital of France?"

    throughput_results = await benchmark.benchmark_throughput(
        prompt,
        duration_seconds=30,  # Short duration for demonstration
        max_concurrent=5,
    )
    benchmark.save_results(
        throughput_results,
        os.path.join(output_dir, "throughput_benchmark_results.json"),
    )

    logger.info("Inference benchmarks completed")


def generate_report(output_dir: str):
    """
    Generate a report from benchmark results.

    Args:
        output_dir: Directory containing benchmark results
    """
    logger.info("Generating benchmark report...")

    # Load results
    results = {}

    for file in os.listdir(output_dir):
        if file.endswith("_results.json"):
            with open(os.path.join(output_dir, file), "r") as f:
                results[file] = json.load(f)

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {"embedding_quality": {}, "inference_performance": {}},
        "details": results,
    }

    # Extract summary metrics
    if "sts_benchmark_results.json" in results:
        sts_results = results["sts_benchmark_results.json"][-1]  # Get the latest result
        report["summary"]["embedding_quality"]["sts_correlation"] = sts_results.get(
            "correlation", 0
        )

    if "retrieval_benchmark_results.json" in results:
        retrieval_results = results["retrieval_benchmark_results.json"][
            -1
        ]  # Get the latest result
        report["summary"]["embedding_quality"]["retrieval_f1"] = retrieval_results.get(
            "f1@5", 0
        )

    if "latency_benchmark_results.json" in results:
        latency_results = results["latency_benchmark_results.json"][
            -1
        ]  # Get the latest result
        report["summary"]["inference_performance"]["avg_latency_ms"] = (
            latency_results.get("avg_latency", 0) * 1000
        )
        report["summary"]["inference_performance"]["p95_latency_ms"] = (
            latency_results.get("p95_latency", 0) * 1000
        )

    if "throughput_benchmark_results.json" in results:
        throughput_results = results["throughput_benchmark_results.json"][
            -1
        ]  # Get the latest result
        report["summary"]["inference_performance"]["requests_per_second"] = (
            throughput_results.get("requests_per_second", 0)
        )

    # Save report
    with open(os.path.join(output_dir, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    with open(os.path.join(output_dir, "benchmark_report.md"), "w") as f:
        f.write("# Deep Recall Benchmark Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")

        f.write("### Embedding Quality\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(
            f"| STS Correlation | {report['summary']['embedding_quality'].get('sts_correlation', 0):.4f} |\n"
        )
        f.write(
            f"| Retrieval F1@5 | {report['summary']['embedding_quality'].get('retrieval_f1', 0):.4f} |\n"
        )

        f.write("\n### Inference Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(
            f"| Average Latency | {report['summary']['inference_performance'].get('avg_latency_ms', 0):.2f} ms |\n"
        )
        f.write(
            f"| P95 Latency | {report['summary']['inference_performance'].get('p95_latency_ms', 0):.2f} ms |\n"
        )
        f.write(
            f"| Requests per Second | {report['summary']['inference_performance'].get('requests_per_second', 0):.2f} |\n"
        )

        f.write("\n## Detailed Results\n\n")
        f.write("Detailed results are available in the JSON files in this directory.\n")

    logger.info(
        f"Benchmark report generated: {os.path.join(output_dir, 'benchmark_report.md')}"
    )


async def main():
    """Main function to run benchmarks and generate reports."""
    parser = argparse.ArgumentParser(description="Run benchmarks and generate reports")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="URL of the inference API",
    )
    parser.add_argument(
        "--auth-token", type=str, help="Authentication token for the inference API"
    )
    parser.add_argument(
        "--skip-embedding", action="store_true", help="Skip embedding benchmarks"
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip inference benchmarks"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmarks
    if not args.skip_embedding:
        await run_embedding_benchmarks(args.output_dir)

    if not args.skip_inference:
        await run_inference_benchmarks(args.output_dir, args.api_url, args.auth_token)

    # Generate report
    generate_report(args.output_dir)

    logger.info("Benchmarks completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
