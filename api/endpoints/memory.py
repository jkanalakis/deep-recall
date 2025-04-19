#!/usr/bin/env python3
# api/endpoints/memory.py

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time
from loguru import logger

# Import auth middleware for dependency injection
from api.middleware.auth import authenticate_request, TokenData, check_scope


# Memory models for request/response
class MemoryInput(BaseModel):
    text: str = Field(..., description="The text to store in memory")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata for the memory"
    )


class MemoryResponse(BaseModel):
    id: int = Field(..., description="Unique identifier for the memory")
    text: str = Field(..., description="The stored text")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata associated with the memory"
    )
    timestamp: str = Field(..., description="ISO timestamp when the memory was created")


class MemoryQueryInput(BaseModel):
    query: str = Field(..., description="The text query to find relevant memories")
    k: int = Field(default=3, description="Number of results to return")
    threshold: Optional[float] = Field(
        default=None, description="Optional similarity threshold (0-1)"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filtering criteria"
    )


class DeleteResponse(BaseModel):
    success: bool = Field(
        ..., description="Whether the delete operation was successful"
    )
    memory_id: int = Field(..., description="ID of the deleted memory")


# Create router
router = APIRouter()


# Example of memory service client (to be replaced with actual client)
class MemoryServiceClient:
    """Client for interacting with the Memory Service"""

    async def add_memory(
        self, text: str, metadata: Optional[Dict[str, Any]] = None, user_id: str = None
    ) -> MemoryResponse:
        """Add a memory to the service"""
        # This would make an actual request to the memory service
        # For now, we'll simulate a response
        memory_id = int(time.time() * 1000) % 100000  # Simulate an ID

        # Add user_id to metadata if not present
        if metadata is None:
            metadata = {}
        if user_id and "user_id" not in metadata:
            metadata["user_id"] = user_id

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return MemoryResponse(
            id=memory_id, text=text, metadata=metadata, timestamp=timestamp
        )

    async def get_memories(
        self,
        query: str,
        k: int = 3,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        user_id: str = None,
    ) -> List[MemoryResponse]:
        """Retrieve relevant memories"""
        # This would make an actual request to the memory service
        # For now, we'll simulate a response

        # Add user filtering by default if not explicitly overridden
        if filter_metadata is None:
            filter_metadata = {}
        if user_id and "user_id" not in filter_metadata:
            filter_metadata["user_id"] = user_id

        # Simulate response with dummy data
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        results = []
        for i in range(min(k, 3)):  # Simulate fewer results than requested
            memory_id = int(time.time() * 1000) % 100000 + i
            results.append(
                MemoryResponse(
                    id=memory_id,
                    text=f"Related memory for query: {query} (result {i+1})",
                    metadata={
                        "relevance": 0.9 - (i * 0.1),
                        "user_id": user_id,
                        **filter_metadata,
                    },
                    timestamp=timestamp,
                )
            )

        return results

    async def delete_memory(self, memory_id: int, user_id: str = None) -> bool:
        """Delete a memory by ID"""
        # This would make an actual request to the memory service
        # For now, we'll simulate success
        return True


# Get memory service client (would be dependency injected in production)
def get_memory_service():
    return MemoryServiceClient()


@router.post("/", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(
    memory_input: MemoryInput,
    token: TokenData = Depends(authenticate_request),
    memory_service: MemoryServiceClient = Depends(get_memory_service),
):
    """
    Store new text in memory with optional metadata
    """
    try:
        # Use the authenticated user ID for the memory
        result = await memory_service.add_memory(
            text=memory_input.text,
            metadata=memory_input.metadata,
            user_id=token.user_id,
        )
        return result
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store memory",
        )


@router.post("/query", response_model=List[MemoryResponse])
async def query_memories(
    query_input: MemoryQueryInput,
    token: TokenData = Depends(authenticate_request),
    memory_service: MemoryServiceClient = Depends(get_memory_service),
):
    """
    Query memories by semantic similarity
    """
    try:
        # Retrieve memories, filtering by the user's ID by default
        results = await memory_service.get_memories(
            query=query_input.query,
            k=query_input.k,
            threshold=query_input.threshold,
            filter_metadata=query_input.filter_metadata,
            user_id=token.user_id,
        )
        return results
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query memories",
        )


@router.delete("/{memory_id}", response_model=DeleteResponse)
async def delete_memory(
    memory_id: int = Path(..., description="ID of the memory to delete"),
    token: TokenData = Depends(check_scope("delete:memory")),  # Requires specific scope
    memory_service: MemoryServiceClient = Depends(get_memory_service),
):
    """
    Delete a specific memory by ID
    """
    try:
        success = await memory_service.delete_memory(memory_id, user_id=token.user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory with ID {memory_id} not found or already deleted",
            )

        return DeleteResponse(success=True, memory_id=memory_id)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete memory",
        )
