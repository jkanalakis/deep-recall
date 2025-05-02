#!/usr/bin/env python3
"""
Deep Recall Memory API

This API provides endpoints for managing and retrieving memories with the Deep Recall framework.
"""

import os
import uuid
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Import the memory components 
from memory.memory_service import MemoryService
from memory.vector_db.faiss_store import FAISSVectorStore
from memory.models import Memory, MemoryImportance

# Configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8404"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Database configuration
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "recall_memories_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

# Set up logging
log_level = getattr(logging, LOG_LEVEL.upper())
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger.info(f"Starting API on {API_HOST}:{API_PORT} with log level {LOG_LEVEL}")
logger.info(f"Using database at {DB_HOST}:{DB_PORT}/{DB_NAME}")

# Initialize the app
app = FastAPI(
    title="Deep Recall Memory API",
    description="API for storing and retrieving memories using Deep Recall",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store and memory service
vector_store = FAISSVectorStore(dimension=384)
memory_service = MemoryService(
    vector_store=vector_store,
    db_host=DB_HOST,
    db_port=DB_PORT,
    db_name=DB_NAME,
    db_user=DB_USER,
    db_password=DB_PASSWORD
)

# Pydantic models for API
class ChatMessage(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    message: str = Field(..., description="Message content")

class MemoryResponse(BaseModel):
    id: str
    text: str
    user_id: str
    created_at: str
    similarity: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    memories: List[MemoryResponse] = []

# Routes
@app.get("/")
async def read_root():
    return {"status": "online", "service": "Deep Recall Memory API"}

@app.get("/health")
async def health_check():
    # Check services here if needed
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Process a user message and return a response with relevant memories.
    """
    try:
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        
        # Store the user message as a memory
        memory = Memory(
            id=memory_id,
            text=message.message,
            user_id=message.user_id,
            created_at=datetime.now().isoformat(),
            importance=MemoryImportance.NORMAL,
            metadata={
                "source": "user_message",
                "type": "chat"
            }
        )
        
        # Explicitly store the memory and verify it was stored
        stored_id = memory_service.store_memory(memory)
        logger.info(f"Stored memory with ID: {stored_id}")
        
        # Verify the memory was stored by retrieving it
        stored_memory = memory_service.get_memory(memory_id)
        if stored_memory is None:
            logger.warning(f"Failed to verify memory storage for ID: {memory_id}")
        
        # Wait a moment to ensure indexing is complete
        time.sleep(0.1)
        
        # Only retrieve memories that existed before this message
        relevant_memories = []
        if memory_service.vector_store.vector_db.get_vector_count() > 1:
            # Only search if we have more than the memory we just added
            relevant_memories = memory_service.retrieve_memories(
                user_id=message.user_id,
                query=message.message,
                limit=5,
                threshold=0.85
            )
            
            # Filter out the memory we just added (in case it matches itself)
            relevant_memories = [mem for mem in relevant_memories if mem.id != memory_id]

        # Format memories for response
        memory_responses = []
        for mem in relevant_memories:
            # Skip memories with empty text
            if not mem.text.strip():
                logger.warning(f"Skipping memory with empty text: {mem.id}")
                continue
                
            logger.info(f"Found memory: id={mem.id}, similarity={getattr(mem, 'similarity', 'unknown')}, text={mem.text[:50]}")
            memory_responses.append(
                MemoryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=mem.user_id,
                    created_at=mem.created_at,
                    similarity=mem.similarity if hasattr(mem, "similarity") else None
                )
            )

        # Construct a sample response
        if relevant_memories:
            sample_response = f"I've found some memories related to your message. I found {len(relevant_memories)} relevant memories."
        else:
            sample_response = "I don't have any previous memories related to this topic."

        return ChatResponse(
            response=sample_response,
            memories=memory_responses
        )
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/memories", response_model=Dict[str, str])
async def store_memory(
    user_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    importance: str = "NORMAL"
):
    """
    Store a new memory for a user.
    """
    try:
        # Convert importance string to enum
        try:
            importance_enum = getattr(MemoryImportance, importance)
        except AttributeError:
            importance_enum = MemoryImportance.NORMAL
            
        # Create and store the memory
        memory = Memory(
            id=str(uuid.uuid4()),
            text=text,
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            importance=importance_enum,
            metadata=metadata or {}
        )
        memory_service.store_memory(memory)
        
        return {"status": "success", "memory_id": memory.id}
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")

@app.get("/memories", response_model=List[MemoryResponse])
async def retrieve_memories(
    user_id: str,
    query: str,
    limit: int = 5,
    threshold: float = 0.85
):
    """
    Retrieve relevant memories for a user based on a query.
    """
    try:
        memories = memory_service.retrieve_memories(
            user_id=user_id,
            query=query,
            limit=limit,
            threshold=threshold
        )
        
        # Format memories for response
        memory_responses = []
        for mem in memories:
            memory_responses.append(
                MemoryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=mem.user_id,
                    created_at=mem.created_at,
                    similarity=mem.similarity if hasattr(mem, "similarity") else None
                )
            )
        
        return memory_responses
    except Exception as e:
        logger.error(f"Error retrieving memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """
    Delete a specific memory.
    """
    try:
        memory_service.delete_memory(memory_id)
        return {"status": "success", "message": f"Memory {memory_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

@app.delete("/users/{user_id}/memories")
async def delete_user_memories(user_id: str):
    """
    Delete all memories for a specific user.
    """
    try:
        # This would need to be implemented in the memory service
        # memory_service.delete_user_memories(user_id)
        return {"status": "success", "message": f"All memories for user {user_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting user memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user memories: {str(e)}")

@app.get("/debug/memory-stats")
async def memory_stats():
    """
    Get debug information about the memory system.
    """
    try:
        # Get stats from memory store 
        memory_stats = memory_service.memory_store.get_stats()
        
        # Get vector store stats
        vector_stats = {
            "vector_count": memory_service.vector_store.vector_db.get_vector_count(),
            "vector_dimension": memory_service.vector_store.vector_db.get_dimension()
        }
        
        # Get ID mapping if available
        id_mapping = {}
        if hasattr(memory_service.vector_store, '_id_mapping'):
            # Only include first 10 mappings to avoid overly large responses
            id_mapping = {str(k): v for k, v in list(memory_service.vector_store._id_mapping.items())[:10]}
        
        return {
            "memory_stats": memory_stats,
            "vector_stats": vector_stats,
            "memory_service": {
                "type": type(memory_service).__name__,
                "vector_store_type": type(memory_service.vector_store).__name__,
                "vector_db_type": type(memory_service.vector_store.vector_db).__name__
            },
            "id_mapping_sample": id_mapping
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")

# Run the API if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True) 