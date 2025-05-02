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

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the memory components from the main project
from memory.memory_service import MemoryService
from memory.vector_db.vector_store import VectorStore
from memory.vector_db.faiss_store import FAISSVectorStore

# Patch the embeddings import before importing MemoryService
import sys
from memory.embeddings.base import EmbeddingModel
sys.modules['memory.embeddings.embedding_model'] = sys.modules['memory.embeddings.base']

try:
    from memory.embeddings.embedding_model_factory import EmbeddingModelFactory
except ImportError:
    # Create a simple local version if the import fails
    from memory.embeddings.sentence_transformer import SentenceTransformerModel
    
    class EmbeddingModelFactory:
        """Simple factory for embedding models"""
        
        @staticmethod
        def create_model(model_type: str, **kwargs) -> EmbeddingModel:
            """Create an embedding model"""
            if model_type == "SentenceTransformer":
                model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
                return SentenceTransformerModel(model_name=model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

from memory.models import Memory, MemoryImportance

# Configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8404"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DB_HOST = os.environ.get("DB_HOST", "db")
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

# Initialize embedding model and memory service
embedding_model = EmbeddingModelFactory.create_model(
    "SentenceTransformer", 
    model_name="BAAI/bge-base-en-v1.5"
)
# Create a vector store using the embedding model
vector_store = FAISSVectorStore(dimension=embedding_model.get_embedding_dim())
memory_service = MemoryService(
    vector_store=vector_store
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
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        arbitrary_types_allowed = True

class ChatResponse(BaseModel):
    response: str
    memories: List[MemoryResponse] = []

# Routes
@app.get("/")
async def read_root():
    return {"status": "online", "service": "Deep Recall Memory API"}

@app.get("/test")
async def test_endpoint():
    return {"status": "success", "message": "API is working correctly"}

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
        # Generate embedding for the message
        message_embedding = embedding_model.embed_text(message.message)
        
        # Format embedding for pgvector - it needs a 1D array, not 2D
        if message_embedding.ndim > 1 and message_embedding.shape[0] == 1:
            message_embedding = message_embedding.flatten()

        # Store the user message as a memory
        memory = Memory(
            id=str(uuid.uuid4()),
            text=message.message,
            user_id=message.user_id,
            created_at=datetime.now().isoformat(),
            importance=MemoryImportance.NORMAL,
            metadata={
                "source": "user_message",
                "type": "chat"
            },
            embedding=message_embedding
        )
        memory_service.store_memory(memory)

        # Retrieve relevant memories
        relevant_memories = memory_service.retrieve_memories(
            user_id=message.user_id,
            query=message.message,
            limit=5,
            threshold=0.95
        )

        # Format memories for response
        memory_responses = []
        for mem in relevant_memories:
            # Ensure created_at is a string
            created_at = mem.created_at
            if not isinstance(created_at, str):
                created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
            
            memory_responses.append(
                MemoryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=mem.user_id,
                    created_at=created_at,
                    similarity=mem.similarity if hasattr(mem, "similarity") else None
                )
            )

        # Construct a sample response (in a real app, this would use an LLM)
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
        
        # Generate embedding for the text
        text_embedding = embedding_model.embed_text(text)
        
        # Format embedding for pgvector - it needs a 1D array, not 2D
        if text_embedding.ndim > 1 and text_embedding.shape[0] == 1:
            text_embedding = text_embedding.flatten()
            
        # Create and store the memory
        memory = Memory(
            id=str(uuid.uuid4()),
            text=text,
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            importance=importance_enum,
            metadata=metadata or {},
            embedding=text_embedding
        )
        memory_id = memory_service.store_memory(memory)
        
        return {"status": "success", "memory_id": memory_id}
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")

@app.get("/memories", response_model=List[MemoryResponse])
async def retrieve_memories(
    user_id: str,
    query: str,
    limit: int = 5,
    threshold: float = 0.95
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
            # Ensure created_at is a string
            created_at = mem.created_at
            if not isinstance(created_at, str):
                created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
            
            memory_responses.append(
                MemoryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=mem.user_id,
                    created_at=created_at,
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
        success = memory_service.delete_memory(memory_id)
        if success:
            return {"status": "success", "message": f"Memory {memory_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

@app.delete("/users/{user_id}/memories")
async def delete_user_memories(user_id: str):
    """
    Delete all memories for a specific user.
    """
    try:
        # Get all memories for the user
        memories = memory_service.get_user_memories(user_id)
        
        # Delete each memory
        deleted_count = 0
        for memory in memories:
            if memory_service.delete_memory(memory.id):
                deleted_count += 1
                
        return {"status": "success", "message": f"Deleted {deleted_count} memories for user {user_id}"}
    except Exception as e:
        logger.error(f"Error deleting user memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user memories: {str(e)}")

# Run the API if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True) 