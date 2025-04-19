"""
Benchmarking module for evaluating inference performance.

This module provides functionality to benchmark the latency and throughput
of LLM inference.
"""

import logging
import time
import json
import os
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import statistics
import asyncio
import httpx
import pytest

logger = logging.getLogger(__name__)

class InferenceBenchmark:
    """
    A class to benchmark inference performance.
    """
    
    def __init__(self, api_url: str, auth_token: Optional[str] = None):
        """
        Initialize the inference benchmark.
        
        Args:
            api_url: URL of the inference API
            auth_token: Optional authentication token
        """
        self.api_url = api_url
        self.auth_token = auth_token
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        
        logger.info(f"Initialized inference benchmark with API URL: {api_url}")
    
    async def benchmark_latency(self, 
                               prompts: List[str], 
                               batch_size: int = 1, 
                               num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference latency.
        
        Args:
            prompts: List of prompts to use for benchmarking
            batch_size: Batch size for inference
            num_runs: Number of runs to perform
            
        Returns:
            Dictionary with latency metrics
        """
        latencies = []
        
        async with httpx.AsyncClient() as client:
            for _ in range(num_runs):
                # Prepare batch of prompts
                batch = prompts[:batch_size]
                
                # Measure latency
                start_time = time.time()
                
                try:
                    response = await client.post(
                        self.api_url,
                        json={"prompts": batch},
                        headers=self.headers,
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Error during inference: {response.status_code} - {response.text}")
                        continue
                    
                    end_time = time.time()
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                except Exception as e:
                    logger.error(f"Exception during inference: {str(e)}")
        
        if not latencies:
            logger.warning("No successful inference runs to calculate metrics")
            return {
                "error": "No successful inference runs",
                "model": "unknown",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        results = {
            "avg_latency": float(avg_latency),
            "p50_latency": float(p50_latency),
            "p90_latency": float(p90_latency),
            "p95_latency": float(p95_latency),
            "p99_latency": float(p99_latency),
            "min_latency": float(min_latency),
            "max_latency": float(max_latency),
            "batch_size": batch_size,
            "num_runs": num_runs,
            "successful_runs": len(latencies),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Inference Latency Results: {results}")
        return results
    
    async def benchmark_throughput(self, 
                                  prompt: str, 
                                  duration_seconds: int = 60, 
                                  max_concurrent: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference throughput.
        
        Args:
            prompt: Prompt to use for benchmarking
            duration_seconds: Duration of the benchmark in seconds
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Dictionary with throughput metrics
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []
        
        async def make_request():
            nonlocal total_requests, successful_requests, failed_requests
            
            async with httpx.AsyncClient() as client:
                while time.time() < end_time:
                    total_requests += 1
                    request_start = time.time()
                    
                    try:
                        response = await client.post(
                            self.api_url,
                            json={"prompts": [prompt]},
                            headers=self.headers,
                            timeout=30.0
                        )
                        
                        if response.status_code == 200:
                            successful_requests += 1
                            latency = time.time() - request_start
                            latencies.append(latency)
                        else:
                            failed_requests += 1
                            logger.error(f"Error during inference: {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        failed_requests += 1
                        logger.error(f"Exception during inference: {str(e)}")
        
        # Create tasks for concurrent requests
        tasks = [make_request() for _ in range(max_concurrent)]
        
        # Run tasks concurrently
        await asyncio.gather(*tasks)
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        requests_per_second = total_requests / actual_duration
        successful_requests_per_second = successful_requests / actual_duration
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        else:
            avg_latency = 0
            p95_latency = 0
        
        results = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "requests_per_second": float(requests_per_second),
            "successful_requests_per_second": float(successful_requests_per_second),
            "avg_latency": float(avg_latency),
            "p95_latency": float(p95_latency),
            "duration_seconds": actual_duration,
            "max_concurrent": max_concurrent,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Inference Throughput Results: {results}")
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
            with open(output_path, 'r') as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = []
        
        # Append new results
        existing_results.append(results)
        
        # Save updated results
        with open(output_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_path}")


# Pytest fixtures and tests
@pytest.fixture
def inference_benchmark():
    """Fixture to create an inference benchmark instance."""
    api_url = os.environ.get("INFERENCE_API_URL", "http://localhost:8000/v1/completions")
    auth_token = os.environ.get("INFERENCE_AUTH_TOKEN")
    return InferenceBenchmark(api_url, auth_token)


@pytest.mark.asyncio
async def test_benchmark_latency(inference_benchmark):
    """Test latency benchmarking."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature."
    ]
    
    results = await inference_benchmark.benchmark_latency(prompts, batch_size=1, num_runs=5)
    
    assert "avg_latency" in results
    assert "p95_latency" in results
    assert results["successful_runs"] > 0


@pytest.mark.asyncio
async def test_benchmark_throughput(inference_benchmark):
    """Test throughput benchmarking."""
    prompt = "What is the capital of France?"
    
    results = await inference_benchmark.benchmark_throughput(
        prompt, 
        duration_seconds=10,  # Short duration for testing
        max_concurrent=3
    )
    
    assert "total_requests" in results
    assert "requests_per_second" in results
    assert results["total_requests"] >= 0 