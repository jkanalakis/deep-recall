from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import os
import time
import torch
import yaml
import json
from loguru import logger
import importlib
import sys
from typing import Dict, Any, Optional, List, Union, AsyncIterator
import asyncio
from uuid import uuid4
from inference_service.memory_integration import get_context_for_prompt, get_mock_context

# Configure logger
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(title="Deep Recall Inference Service", 
              description="API for LLM inference with DeepSeek R1 and other models")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
MODEL = None
TOKENIZER = None
MODEL_TYPE = os.environ.get("MODEL_TYPE", "deepseek_r1")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/model_cache")
MODEL_CONFIG_PATH = os.environ.get("MODEL_CONFIG_PATH", "/app/config/model_config.yaml")
LOADED_MODELS = {}  # Dictionary to store multiple loaded models
DEFAULT_MODEL = None  # Default model to use

# Load configuration
CONFIG = {}
try:
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {MODEL_CONFIG_PATH}")
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    CONFIG = {}

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(1024, ge=1, le=8192, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=0.0, le=2.0, description="Penalty for token frequency")
    presence_penalty: float = Field(0.0, ge=0.0, le=2.0, description="Penalty for token presence")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences where generation should stop")
    model: Optional[str] = Field(None, description="Model to use for inference")
    context: Optional[List[Dict[str, str]]] = Field(None, description="Context messages for the conversation")
    stream: bool = Field(False, description="Whether to stream the response")

    @validator("prompt")
    def validate_prompt_length(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        return v

class BatchInferenceRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=10, description="List of prompts to process")
    max_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    model: Optional[str] = None

class ModelInfo(BaseModel):
    model_id: str
    revision: Optional[str] = None
    quantization: Optional[str] = None
    max_sequence_length: int
    loaded: bool = False
    default: bool = False

class InferenceResponse(BaseModel):
    text: str
    usage: Dict[str, int]
    model: str
    inference_time: float
    request_id: str = Field(default_factory=lambda: str(uuid4()))

class BatchInferenceResponse(BaseModel):
    responses: List[InferenceResponse]
    total_time: float
    request_id: str = Field(default_factory=lambda: str(uuid4()))

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str

class HealthResponse(BaseModel):
    status: str
    model: str
    gpu_available: bool
    gpu_count: int
    loaded_models: List[str]
    version: str

class PersonalizedInferenceRequest(BaseModel):
    prompt: str
    user_id: str
    max_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    model: Optional[str] = None
    stream: bool = Field(False, description="Whether to stream the response")
    use_mock_memory: bool = Field(False, description="Whether to use mock memory for testing")
    max_memories: int = Field(5, ge=0, le=10, description="Maximum number of memories to retrieve")

def get_model_by_name(model_name: Optional[str] = None):
    """
    Get the specified model or the default model
    """
    global LOADED_MODELS, DEFAULT_MODEL
    
    if model_name and model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]
    
    if not model_name and DEFAULT_MODEL and DEFAULT_MODEL in LOADED_MODELS:
        return LOADED_MODELS[DEFAULT_MODEL]
    
    # If model not found or no default model
    if len(LOADED_MODELS) > 0:
        # Return the first available model
        first_model = next(iter(LOADED_MODELS.values()))
        return first_model
    
    return None

async def load_model(model_name: str):
    """
    Load a model by name from config
    """
    global LOADED_MODELS, CONFIG, MODEL_CACHE_DIR, DEFAULT_MODEL
    
    if model_name in LOADED_MODELS:
        logger.info(f"Model {model_name} already loaded")
        return LOADED_MODELS[model_name]
    
    if not CONFIG or "models" not in CONFIG:
        raise ValueError("Configuration not loaded or missing models section")
    
    if model_name not in CONFIG["models"]:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    logger.info(f"Loading model {model_name}...")
    
    try:
        if "deepseek" in model_name.lower():
            # Import the DeepSeek model integration
            from models.deepseek_r1_integration import DeepSeekR1Model
            model_instance = DeepSeekR1Model(MODEL_CACHE_DIR)
            LOADED_MODELS[model_name] = model_instance
            
            # Set as default if no default exists or if this is deepseek_r1
            if not DEFAULT_MODEL or model_name == "deepseek_r1":
                DEFAULT_MODEL = model_name
                
            logger.info(f"Successfully loaded {model_name} model")
            return model_instance
        else:
            # For other models, we can add similar import logic
            raise ValueError(f"Unsupported model type: {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        logger.exception(e)
        raise

@app.on_event("startup")
async def startup_event():
    """
    Initialize models on startup
    """
    global CONFIG, DEFAULT_MODEL
    
    if not CONFIG or "models" not in CONFIG:
        logger.warning("No configuration found or missing models section")
        # Attempt to load the default model anyway
        try:
            await load_model(MODEL_TYPE)
        except Exception as e:
            logger.error(f"Failed to load default model: {str(e)}")
        return
    
    # Try to load the default model first
    if MODEL_TYPE in CONFIG["models"]:
        try:
            await load_model(MODEL_TYPE)
            DEFAULT_MODEL = MODEL_TYPE
        except Exception as e:
            logger.error(f"Failed to load default model {MODEL_TYPE}: {str(e)}")
    
    # If default model failed to load, try the first model in config
    if not DEFAULT_MODEL and CONFIG["models"]:
        first_model = next(iter(CONFIG["models"].keys()))
        try:
            await load_model(first_model)
            DEFAULT_MODEL = first_model
        except Exception as e:
            logger.error(f"Failed to load first model {first_model}: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    global LOADED_MODELS, DEFAULT_MODEL
    
    if not LOADED_MODELS:
        return {
            "status": "unhealthy", 
            "message": "No models loaded",
            "model": "none",
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "loaded_models": [],
            "version": getattr(importlib.import_module("inference_service"), "__version__", "unknown")
        }
    
    return {
        "status": "healthy", 
        "model": DEFAULT_MODEL or "none",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "loaded_models": list(LOADED_MODELS.keys()),
        "version": getattr(importlib.import_module("inference_service"), "__version__", "unknown")
    }

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models
    """
    global CONFIG, LOADED_MODELS, DEFAULT_MODEL
    
    if not CONFIG or "models" not in CONFIG:
        raise HTTPException(status_code=500, detail="Configuration not loaded or missing models section")
    
    models = []
    for model_name, model_config in CONFIG["models"].items():
        models.append(
            ModelInfo(
                model_id=model_config.get("model_id", model_name),
                revision=model_config.get("revision"),
                quantization=model_config.get("quantization"),
                max_sequence_length=model_config.get("max_sequence_length", 4096),
                loaded=model_name in LOADED_MODELS,
                default=(model_name == DEFAULT_MODEL)
            )
        )
    
    return {
        "models": models,
        "default_model": DEFAULT_MODEL or "none"
    }

@app.post("/v1/models/{model_name}/load", status_code=status.HTTP_202_ACCEPTED)
async def load_model_endpoint(model_name: str, background_tasks: BackgroundTasks):
    """
    Load a model dynamically
    """
    global CONFIG
    
    if not CONFIG or "models" not in CONFIG:
        raise HTTPException(status_code=500, detail="Configuration not loaded or missing models section")
    
    if model_name not in CONFIG["models"]:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found in configuration")
    
    if model_name in LOADED_MODELS:
        return {"message": f"Model {model_name} already loaded"}
    
    # Load model in background
    background_tasks.add_task(load_model, model_name)
    
    return {"message": f"Model {model_name} loading started"}

async def generate_stream_response(model, request: InferenceRequest) -> AsyncIterator[str]:
    """
    Generate streaming response
    """
    start_time = time.time()
    request_id = str(uuid4())
    
    try:
        # Format the context if provided
        context = []
        if request.context:
            context = request.context
            
        # Generate intro payload
        intro_payload = {
            "request_id": request_id,
            "model": model.model_id,
            "created": int(time.time()),
            "streaming": True,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(intro_payload)}\n\n"
        
        # Placeholder for actual streaming 
        # In a real implementation, this would use the model's streaming capability
        # For now we'll simulate streaming with the full response
        
        # Get full response
        full_text = model.generate_reply(
            request.prompt, 
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Split into simulated chunks (for demo purposes)
        chunks = []
        for i in range(0, len(full_text), 10):
            if i + 10 < len(full_text):
                chunks.append(full_text[i:i+10])
            else:
                chunks.append(full_text[i:])
        
        # Stream each chunk
        for i, chunk in enumerate(chunks):
            # Simulate thinking time
            await asyncio.sleep(0.05)
            
            payload = {
                "request_id": request_id,
                "model": model.model_id,
                "created": int(time.time()),
                "streaming": True,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(payload)}\n\n"
        
        # Send final completion with token counts
        inference_time = time.time() - start_time
        
        # Rough token count
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(full_text.split())
        
        completion_payload = {
            "request_id": request_id,
            "model": model.model_id,
            "created": int(time.time()),
            "streaming": False,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "inference_time": inference_time
        }
        yield f"data: {json.dumps(completion_payload)}\n\n"
        yield f"data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming inference error: {str(e)}")
        error_payload = {
            "request_id": request_id,
            "error": str(e),
            "streaming": True
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield f"data: [DONE]\n\n"

@app.post("/v1/completions", response_model=InferenceResponse)
async def generate_completion(request: InferenceRequest):
    """
    Generate text completion using the loaded model
    """
    model_name = request.model
    model = get_model_by_name(model_name)
    
    if not model:
        if not model_name:
            raise HTTPException(status_code=503, detail="No models loaded")
        else:
            # Try to load the requested model
            try:
                model = await load_model(model_name)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Model {model_name} could not be loaded: {str(e)}")
    
    # Handle streaming response if requested
    if request.stream:
        return StreamingResponse(
            generate_stream_response(model, request),
            media_type="text/event-stream"
        )
    
    # Regular synchronous response
    start_time = time.time()
    
    try:
        # Generate the text
        generated_text = model.generate_reply(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Calculate token usage (rough estimation)
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(generated_text.split())
        
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            text=generated_text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            model=model.model_id if hasattr(model, "model_id") else MODEL_TYPE,
            inference_time=inference_time,
            request_id=str(uuid4())
        )
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/v1/batch_completions", response_model=BatchInferenceResponse)
async def batch_generate_completions(request: BatchInferenceRequest):
    """
    Generate multiple completions in batch
    """
    model_name = request.model
    model = get_model_by_name(model_name)
    
    if not model:
        if not model_name:
            raise HTTPException(status_code=503, detail="No models loaded")
        else:
            # Try to load the requested model
            try:
                model = await load_model(model_name)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Model {model_name} could not be loaded: {str(e)}")
    
    start_time = time.time()
    responses = []
    
    # Process each prompt
    for prompt in request.prompts:
        prompt_start_time = time.time()
        
        try:
            # Generate the text
            generated_text = model.generate_reply(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # Calculate token usage (rough estimation)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
            
            prompt_inference_time = time.time() - prompt_start_time
            
            responses.append(InferenceResponse(
                text=generated_text,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                model=model.model_id if hasattr(model, "model_id") else MODEL_TYPE,
                inference_time=prompt_inference_time,
                request_id=str(uuid4())
            ))
            
        except Exception as e:
            logger.error(f"Batch inference error for prompt: {prompt[:50]}...: {str(e)}")
            # Continue with the next prompt rather than failing the entire batch
            responses.append(InferenceResponse(
                text=f"Error: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=model.model_id if hasattr(model, "model_id") else MODEL_TYPE,
                inference_time=time.time() - prompt_start_time,
                request_id=str(uuid4())
            ))
    
    total_time = time.time() - start_time
    
    return BatchInferenceResponse(
        responses=responses,
        total_time=total_time,
        request_id=str(uuid4())
    )

@app.post("/v1/personalized_completions", response_model=InferenceResponse)
async def generate_personalized_completion(request: PersonalizedInferenceRequest):
    """
    Generate a personalized completion using retrieved memories
    """
    model_name = request.model
    model = get_model_by_name(model_name)
    
    if not model:
        if not model_name:
            raise HTTPException(status_code=503, detail="No models loaded")
        else:
            # Try to load the requested model
            try:
                model = await load_model(model_name)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Model {model_name} could not be loaded: {str(e)}")
    
    # Retrieve context from memory service
    start_time = time.time()
    
    try:
        # Get context from either mock or real memory service
        if request.use_mock_memory:
            context = get_mock_context(request.user_id, request.prompt)
            logger.info(f"Using mock memory context with {len(context)} items")
        else:
            context = await get_context_for_prompt(
                user_id=request.user_id, 
                prompt=request.prompt,
                max_memories=request.max_memories
            )
            logger.info(f"Retrieved memory context with {len(context)} items")
        
        # Handle streaming if requested
        if request.stream:
            # Create a request object with the context included
            inference_request = InferenceRequest(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                context=context,
                stream=True
            )
            
            return StreamingResponse(
                generate_stream_response(model, inference_request),
                media_type="text/event-stream"
            )
        
        # Generate the text with context
        generated_text = model.generate_reply(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            context=context
        )
        
        # Get token counts
        token_info = model.get_token_count(request.prompt, context)
        
        # Calculate completion tokens (rough estimate)
        completion_tokens = len(generated_text.split())
        
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            text=generated_text,
            usage={
                "prompt_tokens": token_info["prompt_tokens"],
                "context_tokens": token_info["context_tokens"],
                "input_tokens": token_info["total_input_tokens"],
                "completion_tokens": completion_tokens,
                "total_tokens": token_info["total_input_tokens"] + completion_tokens
            },
            model=model.model_id if hasattr(model, "model_id") else MODEL_TYPE,
            inference_time=inference_time,
            request_id=str(uuid4())
        )
    
    except Exception as e:
        logger.error(f"Personalized inference error: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Personalized inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 