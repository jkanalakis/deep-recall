#!/usr/bin/env python3
"""
Test script for the Deep Recall Inference API

This script provides examples of how to call the various
inference API endpoints.
"""

import argparse
import asyncio
import json
import httpx
import sys
from typing import Dict, Any, List, Optional
import time
from pprint import pprint

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

async def test_completion(prompt: str, model: Optional[str] = None, stream: bool = False):
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
        "stream": stream
    }
    
    if model:
        request_data["model"] = model
    
    # Make request
    async with httpx.AsyncClient(timeout=30.0) as client:
        if not stream:
            # Regular completion
            response = await client.post(
                f"{BASE_URL}/v1/completions",
                json=request_data
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
                "POST", 
                f"{BASE_URL}/v1/completions", 
                json=request_data
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
                                content = json_data["choices"][0].get("delta", {}).get("content", "")
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
        "top_p": 0.9
    }
    
    if model:
        request_data["model"] = model
    
    # Make request
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/batch_completions",
            json=request_data
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

async def test_personalized_completion(prompt: str, user_id: str, use_mock: bool = True, stream: bool = False):
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
        "stream": stream
    }
    
    # Make request
    async with httpx.AsyncClient(timeout=30.0) as client:
        if not stream:
            # Regular completion
            response = await client.post(
                f"{BASE_URL}/v1/personalized_completions",
                json=request_data
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
                "POST", 
                f"{BASE_URL}/v1/personalized_completions", 
                json=request_data
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
                                content = json_data["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_text += content
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data}")
                
                print("\n\nTotal time: {:.2f}s".format(time.time() - start_time))
                print(f"Full text length: {len(full_text)} characters")

async def main():
    parser = argparse.ArgumentParser(description="Test the Deep Recall Inference API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for the API")
    parser.add_argument("--health", action="store_true", help="Test the health endpoint")
    parser.add_argument("--models", action="store_true", help="Test the models endpoint")
    parser.add_argument("--completion", action="store_true", help="Test the completion endpoint")
    parser.add_argument("--batch", action="store_true", help="Test the batch completion endpoint")
    parser.add_argument("--personalized", action="store_true", help="Test the personalized completion endpoint")
    parser.add_argument("--prompt", default="What is deep learning?", help="Prompt to use for completion")
    parser.add_argument("--prompts", help="Comma-separated list of prompts for batch completion")
    parser.add_argument("--model", help="Model to use for completion")
    parser.add_argument("--user", default="test-user", help="User ID for personalized completion")
    parser.add_argument("--no-mock", action="store_true", help="Don't use mock memory for personalized completion")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    # Run tests
    if args.health or not any([args.health, args.models, args.completion, args.batch, args.personalized]):
        await test_health()
        
    if args.models or not any([args.health, args.models, args.completion, args.batch, args.personalized]):
        await test_models()
        
    if args.completion:
        await test_completion(args.prompt, args.model, args.stream)
        
    if args.batch:
        prompts = args.prompts.split(",") if args.prompts else [
            "What is deep learning?",
            "How does a neural network work?",
            "Explain backpropagation"
        ]
        await test_batch_completion(prompts, args.model)
        
    if args.personalized:
        await test_personalized_completion(args.prompt, args.user, not args.no_mock, args.stream)

if __name__ == "__main__":
    asyncio.run(main()) 