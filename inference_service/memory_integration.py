"""
Memory integration utilities for the Deep Recall inference service.

This module provides helper functions and classes for integrating
with the Deep Recall memory system.
"""

from typing import Dict, List, Any, Optional
import json
import httpx
import os
import asyncio
from pydantic import BaseModel, Field

# Default memory service URL
MEMORY_SERVICE_URL = os.environ.get("MEMORY_SERVICE_URL", "http://memory-service:8080")

class MemoryRequest(BaseModel):
    """Request to retrieve memories from the memory service"""
    user_id: str
    query: str
    max_results: int = Field(5, ge=1, le=20)
    similarity_threshold: float = Field(0.7, ge=0, le=1.0)

class MemoryItem(BaseModel):
    """A single memory item returned from the memory service"""
    id: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]
    similarity_score: float

class MemoryResponse(BaseModel):
    """Response from the memory service"""
    memories: List[MemoryItem]
    total_found: int
    query_time_ms: float

async def retrieve_memories(user_id: str, query: str, max_results: int = 5) -> Optional[MemoryResponse]:
    """
    Retrieve relevant memories for a given user and query
    
    Args:
        user_id: The ID of the user
        query: The query text to search for
        max_results: Maximum number of memory items to retrieve
        
    Returns:
        MemoryResponse object with retrieved memories or None if failed
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMORY_SERVICE_URL}/memories/search",
                json={
                    "user_id": user_id,
                    "query": query,
                    "max_results": max_results,
                    "similarity_threshold": 0.7
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return MemoryResponse(**result)
            else:
                print(f"Error retrieving memories: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Exception retrieving memories: {str(e)}")
        return None

def format_memories_as_context(memories: List[MemoryItem]) -> List[Dict[str, str]]:
    """
    Format memory items as context messages for the LLM
    
    Args:
        memories: List of memory items from the memory service
        
    Returns:
        List of context messages in the format expected by the LLM
    """
    context = []
    
    # Add a system message explaining the memories
    if memories:
        context.append({
            "role": "system",
            "content": "The following are relevant memories from previous conversations with this user:"
        })
    
    # Add each memory as a message pair
    for memory in memories:
        # Parse the content if it's in the expected format
        try:
            # If the memory content is already JSON with role/content
            memory_data = json.loads(memory.content)
            if isinstance(memory_data, dict) and "role" in memory_data and "content" in memory_data:
                context.append(memory_data)
                continue
        except:
            # Not JSON or not in expected format, continue with default handling
            pass
            
        # Format the memory as a pair of messages
        if "query" in memory.metadata and "response" in memory.metadata:
            # Add as a conversation pair
            context.append({
                "role": "user",
                "content": memory.metadata["query"]
            })
            context.append({
                "role": "assistant",
                "content": memory.metadata["response"]
            })
        else:
            # Just add the raw content as a system message
            context.append({
                "role": "system",
                "content": f"Previous interaction: {memory.content}"
            })
    
    return context

async def get_context_for_prompt(user_id: str, prompt: str, max_memories: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve and format memories as context for a given user and prompt
    
    Args:
        user_id: User ID to retrieve memories for
        prompt: Current user prompt
        max_memories: Maximum number of memory items to retrieve
        
    Returns:
        List of context messages in the format expected by the LLM
    """
    # Retrieve memories
    memories = await retrieve_memories(user_id, prompt, max_memories)
    
    # If no memories or error, return empty context
    if not memories or not memories.memories:
        return []
    
    # Format memories as context
    return format_memories_as_context(memories.memories)

# Mock function for testing without memory service
def get_mock_context(user_id: str, prompt: str) -> List[Dict[str, str]]:
    """
    Generate mock context for testing without a memory service
    
    Args:
        user_id: User ID 
        prompt: Current prompt
        
    Returns:
        Mocked context messages
    """
    return [
        {
            "role": "system",
            "content": "You are talking to a user who has previously asked about Python programming."
        },
        {
            "role": "user", 
            "content": "How do I read a file in Python?"
        },
        {
            "role": "assistant",
            "content": "In Python, you can read a file using the built-in `open()` function. Here's an example:\n\n```python\nwith open('filename.txt', 'r') as file:\n    content = file.read()\n    print(content)\n```\n\nThis will open the file, read its contents into a string, and then print it."
        }
    ] 