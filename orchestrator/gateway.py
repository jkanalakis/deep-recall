#!/usr/bin/env python3
# orchestrator/gateway.py

"""
API Gateway for the Deep Recall Framework.

This module implements the API Gateway functionality, providing a unified
interface for client applications to access memory and inference services.
"""

from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uuid
import time
import json
from loguru import logger
from datetime import datetime

from orchestrator.routing import RequestRouter

# Define API request and response models
class InferenceRequest(BaseModel):
    """Request model for inference with memory integration."""
    prompt: str = Field(..., description="User's input prompt/query")
    use_memory: bool = Field(True, description="Whether to use memory for context")
    memory_k: int = Field(10, description="Number of memories to retrieve", ge=1, le=50)
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling", ge=0, le=2)
    model: str = Field("default", description="Model to use for inference")
    top_p: float = Field(1.0, description="Nucleus sampling parameter", ge=0, le=1)
    top_k: int = Field(50, description="Top-k sampling parameter", ge=0)
    memory_filters: Dict[str, Any] = Field({}, description="Filters for memory retrieval")
    store_interaction: bool = Field(True, description="Whether to store this interaction")
    include_memory_ids: bool = Field(False, description="Whether to return memory IDs in response")
    
class InferenceResponse(BaseModel):
    """Response model for inference with memory metadata."""
    text: str = Field(..., description="Generated response text")
    status: str = Field("success", description="Status of the request")
    context_metadata: Dict[str, Any] = Field({}, description="Metadata about context used")
    memory_ids: Optional[List[str]] = Field(None, description="IDs of memories used")
    
class FeedbackRequest(BaseModel):
    """Request model for storing user feedback."""
    interaction_id: str = Field(..., description="ID of the interaction to give feedback on")
    rating: int = Field(..., description="Rating (1-5)", ge=1, le=5)
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    
class StoreMemoryRequest(BaseModel):
    """Request model for storing a new memory."""
    text: str = Field(..., description="Memory text content")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    source: Optional[str] = Field(None, description="Source of the memory")
    
class ApiGateway:
    """
    API Gateway for serving and coordinating Deep Recall services.
    
    This class is responsible for:
    1. Providing a unified API for client applications
    2. Handling authentication and authorization
    3. Routing requests to the appropriate services
    4. Managing error handling and response formatting
    """
    
    def __init__(
        self,
        router: RequestRouter,
        enable_cors: bool = True,
        allowed_origins: List[str] = None
    ):
        """
        Initialize the API Gateway.
        
        Args:
            router: Request router for handling service coordination
            enable_cors: Whether to enable CORS
            allowed_origins: List of allowed origins for CORS
        """
        self.router = router
        self.app = FastAPI(
            title="Deep Recall API",
            description="API for memory-enhanced AI interactions",
            version="1.0.0"
        )
        
        # Set up CORS if enabled
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Set up routes
        self._setup_routes()
        
        logger.info("API Gateway initialized")
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.post("/api/v1/inference", response_model=InferenceResponse)
        async def inference(
            request: InferenceRequest,
            user_info = Depends(self._get_user_info)
        ):
            """Generate a response with memory context."""
            try:
                user_id = user_info["user_id"]
                session_id = user_info.get("session_id")
                
                response = await self.router.route_inference_request(
                    request_data=request.dict(),
                    user_id=user_id,
                    session_id=session_id
                )
                
                if "error" in response:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                        detail=response["error"]
                    )
                    
                return response
                
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Inference request failed: {str(e)}"
                )
        
        @self.app.post("/api/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
        async def store_feedback(
            request: FeedbackRequest,
            user_info = Depends(self._get_user_info)
        ):
            """Store feedback about a response."""
            try:
                user_id = user_info["user_id"]
                
                result = await self.router.store_user_feedback(
                    feedback_data=request.dict(),
                    user_id=user_id
                )
                
                if result.get("status") == "error":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=result.get("message", "Failed to store feedback")
                    )
                    
                return {"status": "success", "message": "Feedback received"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Feedback error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to store feedback: {str(e)}"
                )
        
        @self.app.post("/api/v1/memory", status_code=status.HTTP_201_CREATED)
        async def store_memory(
            request: StoreMemoryRequest,
            user_info = Depends(self._get_user_info)
        ):
            """Store a new memory."""
            try:
                user_id = user_info["user_id"]
                
                if not self.router.memory_client:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Memory service not available"
                    )
                
                # Prepare memory entry
                memory_entry = {
                    "user_id": user_id,
                    "text": request.text,
                    "source": request.source or "api",
                    "metadata": request.metadata or {}
                }
                
                # Add session ID if available
                if "session_id" in user_info:
                    memory_entry["metadata"]["session_id"] = user_info["session_id"]
                
                # Store the memory
                result = await self.router.memory_client.store_memory(memory_entry)
                
                return {"status": "success", "memory_id": result.get("id", "unknown")}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Memory storage error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to store memory: {str(e)}"
                )
        
        @self.app.get("/api/v1/memories")
        async def get_memories(
            query: Optional[str] = None,
            k: int = 10,
            user_info = Depends(self._get_user_info)
        ):
            """Retrieve memories based on a query."""
            try:
                user_id = user_info["user_id"]
                
                if not self.router.memory_client:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Memory service not available"
                    )
                
                # Get memories
                filters = {}
                if "session_id" in user_info:
                    filters["session_id"] = user_info["session_id"]
                
                memories = await self.router.memory_client.get_memories(
                    query=query or "",
                    k=k,
                    user_id=user_id,
                    filters=filters
                )
                
                return {"memories": memories, "count": len(memories)}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Memory retrieval error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve memories: {str(e)}"
                )
    
    async def _get_user_info(self, request: Request) -> Dict[str, Any]:
        """
        Extract user information from request.
        
        In a real implementation, this would validate authentication tokens
        and extract user ID and other information. For now, we'll use headers
        or query parameters as a simple placeholder.
        
        Args:
            request: The incoming request
            
        Returns:
            Dictionary with user information
        """
        # Get user ID from header or query param (for demo/testing purposes)
        user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
        
        # In a real implementation, this would decode and validate a JWT token
        # and extract user information from the token claims
        
        if not user_id:
            # For demo purposes, generate a temporary user ID
            # In production, this should require proper authentication
            user_id = f"temp-user-{uuid.uuid4()}"
            logger.warning(f"No user ID provided, using temporary ID: {user_id}")
        
        # Get optional session ID
        session_id = request.headers.get("X-Session-ID") or request.query_params.get("session_id")
        
        # Return user information
        user_info = {"user_id": user_id}
        if session_id:
            user_info["session_id"] = session_id
            
        return user_info
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app 