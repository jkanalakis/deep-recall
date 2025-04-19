#!/usr/bin/env python3
"""
Test script for the Deep Recall Inference API

This script provides examples of how to call the various
inference API endpoints.
"""

import argparse
import asyncio
import json
import sys
import time
from pprint import pprint
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://localhost:8000"


async def test_health():
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")

        if response.status_code == 200:
            print("Health Status: OK")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_models():
    """Test the models endpoint"""
    print("\n=== Testing Models Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/models")

        if response.status_code == 200:
            print("Models: OK")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_completion(
    prompt: str, model: Optional[str] = None, stream: bool = False
):
    """Test the completion endpoint"""
    print(f"\n=== Testing Completion Endpoint ===")
    print(f"Prompt: {prompt}")
    print(f"Stream: {stream}")

    # Prepare request
    request_data = {
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": stream,
    }

    if model:
        request_data["model"] = model

    # Make request
    async with httpx.AsyncClient(timeout=30.0) as client:
        if not stream:
            # Regular completion
            response = await client.post(
                f"{BASE_URL}/v1/completions", json=request_data
            )

            if response.status_code == 200:
                print("Completion: OK")
                print("\nResponse:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        else:
            # Streaming completion
            start_time = time.time()
            async with client.stream(
                "POST", f"{BASE_URL}/v1/completions", json=request_data
            ) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(await response.text())
                    return

                print("\nStreaming response:")
                full_text = ""
                async for chunk in response.aiter_text():
                    if chunk.startswith("data:"):
                        data = chunk.replace("data:", "").strip()
                        if data == "[DONE]":
                            break

                        try:
                            json_data = json.loads(data)
                            if "choices" in json_data and json_data["choices"]:
                                content = (
                                    json_data["choices"][0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if content:
                                    print(content, end="", flush=True)
                                    full_text += content
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data}")

                print("\n\nTotal time: {:.2f}s".format(time.time() - start_time))
                print(f"Full text length: {len(full_text)} characters")


async def test_batch_completion(prompts: List[str], model: Optional[str] = None):
    """Test the batch completion endpoint"""
    print(f"\n=== Testing Batch Completion Endpoint ===")
    print(f"Number of prompts: {len(prompts)}")

    # Prepare request
    request_data = {
        "prompts": prompts,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    if model:
        request_data["model"] = model

    # Make request
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/batch_completions", json=request_data
        )

        if response.status_code == 200:
            print("Batch Completion: OK")
            result = response.json()
            print(f"Total time: {result['total_time']:.2f}s")
            print(f"Request ID: {result['request_id']}")
            print("\nResponses:")
            for i, resp in enumerate(result["responses"]):
                print(f"\n--- Response {i+1} ---")
                print(f"Prompt: {prompts[i]}")
                print(f"Response: {resp['text']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_personalized_completion(
    prompt: str, user_id: str, use_mock: bool = True, stream: bool = False
):
    """Test the personalized completion endpoint"""
    print(f"\n=== Testing Personalized Completion Endpoint ===")
    print(f"Prompt: {prompt}")
    print(f"User ID: {user_id}")
    print(f"Use Mock Memory: {use_mock}")
    print(f"Stream: {stream}")

    # Prepare request
    request_data = {
        "prompt": prompt,
        "user_id": user_id,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_mock_memory": use_mock,
        "stream": stream,
    }

    # Make request
    async with httpx.AsyncClient(timeout=30.0) as client:
        if not stream:
            # Regular completion
            response = await client.post(
                f"{BASE_URL}/v1/personalized_completions", json=request_data
            )

            if response.status_code == 200:
                print("Personalized Completion: OK")
                print("\nResponse:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        else:
            # Streaming completion
            start_time = time.time()
            async with client.stream(
                "POST", f"{BASE_URL}/v1/personalized_completions", json=request_data
            ) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(await response.text())
                    return

                print("\nStreaming response:")
                full_text = ""
                async for chunk in response.aiter_text():
                    if chunk.startswith("data:"):
                        data = chunk.replace("data:", "").strip()
                        if data == "[DONE]":
                            break

                        try:
                            json_data = json.loads(data)
                            if "choices" in json_data and json_data["choices"]:
                                content = (
                                    json_data["choices"][0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if content:
                                    print(content, end="", flush=True)
                                    full_text += content
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data}")

                print("\n\nTotal time: {:.2f}s".format(time.time() - start_time))
                print(f"Full text length: {len(full_text)} characters")


async def test_gpu_status():
    """Test the GPU status endpoint"""
    print("\n=== Testing GPU Status Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/gpu_status")

        if response.status_code == 200:
            print("GPU Status: OK")
            result = response.json()

            if result["available"]:
                print(f"Available GPUs: {result['count']}")
                print(f"Total Memory: {result['total_memory_gb']:.2f} GB")
                print(f"Used Memory: {result['used_memory_gb']:.2f} GB")
                print(f"Free Memory: {result['free_memory_gb']:.2f} GB")

                print("\nGPU Devices:")
                for i, device in enumerate(result["devices"]):
                    print(f"  Device {i}: {device['name']}")
                    print(
                        f"    Memory: {device['free_memory_gb']:.2f} GB free / {device['total_memory_gb']:.2f} GB total"
                    )
            else:
                print("No GPUs available")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_load_model_with_optimizations(
    model_name: str, quantization: str = None, parallel_mode: str = None
):
    """Test loading a model with optimizations"""
    print(f"\n=== Testing Loading Model with Optimizations ===")
    print(f"Model: {model_name}")
    print(f"Quantization: {quantization or 'default'}")
    print(f"Parallel Mode: {parallel_mode or 'default'}")

    # Prepare request
    request_data = {
        "model_name": model_name,
    }

    if quantization:
        request_data["quantization"] = quantization

    if parallel_mode:
        request_data["parallel_mode"] = parallel_mode

    # Make request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/models/{model_name}/load", json=request_data
        )

        if response.status_code == 202:
            print("Model Loading: Started")
            print(response.json()["message"])
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_clear_gpu_memory():
    """Test clearing GPU memory"""
    print(f"\n=== Testing Clear GPU Memory ===")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/v1/clear_gpu_memory")

        if response.status_code == 200:
            print("GPU Memory Cleared: OK")
            result = response.json()
            print(f"Memory Before: {result['memory_before_gb']:.2f} GB")
            print(f"Memory After: {result['memory_after_gb']:.2f} GB")
            print(f"Memory Freed: {result['freed_gb']:.2f} GB")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_offload_model(model_name: str):
    """Test offloading a model to CPU"""
    print(f"\n=== Testing Offload Model ===")
    print(f"Model: {model_name}")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/v1/models/{model_name}/offload")

        if response.status_code == 200:
            print(f"Model Offloaded: OK")
            result = response.json()
            print(f"Memory Before: {result['memory_before_gb']:.2f} GB")
            print(f"Memory After: {result['memory_after_gb']:.2f} GB")
            print(f"Memory Freed: {result['freed_gb']:.2f} GB")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_reload_model(model_name: str):
    """Test reloading a model to GPU"""
    print(f"\n=== Testing Reload Model ===")
    print(f"Model: {model_name}")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/v1/models/{model_name}/reload")

        if response.status_code == 200:
            print(f"Model Reloaded: OK")
            result = response.json()
            print(f"Current GPU Memory: {result['current_memory_gb']:.2f} GB")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_batch_completion_performance(
    prompts: List[str], model: Optional[str] = None
):
    """Test batch completion performance"""
    print(f"\n=== Testing Batch Completion Performance ===")
    print(f"Number of prompts: {len(prompts)}")

    # First, get current GPU memory usage
    async with httpx.AsyncClient() as client:
        gpu_response = await client.get(f"{BASE_URL}/v1/gpu_status")
        if gpu_response.status_code == 200:
            gpu_data = gpu_response.json()
            if gpu_data["available"]:
                print(f"Initial GPU Memory Usage: {gpu_data['used_memory_gb']:.2f} GB")

    # Prepare request
    request_data = {
        "prompts": prompts,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    if model:
        request_data["model"] = model

    # Measure completion time
    start_time = time.time()

    # Make request
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/batch_completions", json=request_data
        )

        if response.status_code == 200:
            end_time = time.time()
            total_time = end_time - start_time

            result = response.json()
            api_reported_time = result["total_time"]

            print(f"Batch Completion: OK")
            print(f"Total client time: {total_time:.2f}s")
            print(f"API reported time: {api_reported_time:.2f}s")
            print(f"Network overhead: {(total_time - api_reported_time):.2f}s")
            print(f"Average time per prompt: {(api_reported_time / len(prompts)):.2f}s")

            # Check GPU memory after inference
            gpu_response = await client.get(f"{BASE_URL}/v1/gpu_status")
            if gpu_response.status_code == 200:
                gpu_data = gpu_response.json()
                if gpu_data["available"]:
                    print(
                        f"Final GPU Memory Usage: {gpu_data['used_memory_gb']:.2f} GB"
                    )
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def main():
    parser = argparse.ArgumentParser(description="Test the Deep Recall Inference API")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL for the API"
    )
    parser.add_argument(
        "--health", action="store_true", help="Test the health endpoint"
    )
    parser.add_argument(
        "--models", action="store_true", help="Test the models endpoint"
    )
    parser.add_argument(
        "--completion", action="store_true", help="Test the completion endpoint"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Test the batch completion endpoint"
    )
    parser.add_argument(
        "--personalized",
        action="store_true",
        help="Test the personalized completion endpoint",
    )
    parser.add_argument(
        "--prompt",
        default="What is deep learning?",
        help="Prompt to use for completion",
    )
    parser.add_argument(
        "--prompts", help="Comma-separated list of prompts for batch completion"
    )
    parser.add_argument("--model", help="Model to use for completion")
    parser.add_argument(
        "--user", default="test-user", help="User ID for personalized completion"
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Don't use mock memory for personalized completion",
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")

    # GPU optimization test arguments
    parser.add_argument(
        "--gpu-status", action="store_true", help="Test the GPU status endpoint"
    )
    parser.add_argument(
        "--load-optimized",
        action="store_true",
        help="Test loading a model with optimizations",
    )
    parser.add_argument(
        "--quantization",
        help="Quantization mode for model loading (none, int8, int4, gptq, awq)",
    )
    parser.add_argument(
        "--parallel", help="Model parallelism mode (none, tensor, pipeline, expert)"
    )
    parser.add_argument(
        "--clear-gpu", action="store_true", help="Test clearing GPU memory"
    )
    parser.add_argument(
        "--offload", action="store_true", help="Test offloading a model to CPU"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Test reloading a model to GPU"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Test batch performance with detailed metrics",
    )

    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    # Run tests
    if args.health or not any(
        [
            args.health,
            args.models,
            args.completion,
            args.batch,
            args.personalized,
            args.gpu_status,
            args.load_optimized,
            args.clear_gpu,
            args.offload,
            args.reload,
            args.performance,
        ]
    ):
        await test_health()

    if args.models or not any(
        [
            args.health,
            args.models,
            args.completion,
            args.batch,
            args.personalized,
            args.gpu_status,
            args.load_optimized,
            args.clear_gpu,
            args.offload,
            args.reload,
            args.performance,
        ]
    ):
        await test_models()

    if args.completion:
        await test_completion(args.prompt, args.model, args.stream)

    if args.batch:
        prompts = (
            args.prompts.split(",")
            if args.prompts
            else [
                "What is deep learning?",
                "How does a neural network work?",
                "Explain backpropagation",
            ]
        )
        await test_batch_completion(prompts, args.model)

    if args.personalized:
        await test_personalized_completion(
            args.prompt, args.user, not args.no_mock, args.stream
        )

    # GPU optimization tests
    if args.gpu_status:
        await test_gpu_status()

    if args.load_optimized:
        model_name = args.model or "deepseek_r1"
        await test_load_model_with_optimizations(
            model_name, args.quantization, args.parallel
        )

    if args.clear_gpu:
        await test_clear_gpu_memory()

    if args.offload:
        model_name = args.model or "deepseek_r1"
        await test_offload_model(model_name)

    if args.reload:
        model_name = args.model or "deepseek_r1"
        await test_reload_model(model_name)

    if args.performance:
        prompts = (
            args.prompts.split(",")
            if args.prompts
            else [
                "What is deep learning?",
                "How does a neural network work?",
                "Explain backpropagation",
                "What is transfer learning?",
                "Describe how convolutional neural networks work",
            ]
        )
        await test_batch_completion_performance(prompts, args.model)


if __name__ == "__main__":
    asyncio.run(main())
