from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import torch
from loguru import logger
import importlib
import sys
from typing import Dict, Any, Optional, List

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

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None

class InferenceResponse(BaseModel):
    text: str
    usage: Dict[str, int]
    model: str
    inference_time: float

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup
    """
    global MODEL, TOKENIZER, MODEL_TYPE
    
    logger.info(f"Loading {MODEL_TYPE} model...")
    
    try:
        if "deepseek" in MODEL_TYPE.lower():
            # Import the DeepSeek model integration
            from models.deepseek_r1_integration import DeepSeekR1Model
            model_instance = DeepSeekR1Model(MODEL_CACHE_DIR)
            MODEL = model_instance
            logger.info(f"Successfully loaded {MODEL_TYPE} model")
        else:
            # For other models, we can add similar import logic
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.exception(e)
        # Continue startup but log the error - we'll return errors on API calls

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if MODEL is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    
    return {
        "status": "healthy", 
        "model": MODEL_TYPE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/v1/completions", response_model=InferenceResponse)
async def generate_completion(request: InferenceRequest):
    """
    Generate text completion using the loaded model
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Call model's generate method
        generated_text = MODEL.generate_reply(request.prompt)
        
        # Calculate token usage (this would be more accurate with actual tokenizer)
        prompt_tokens = len(request.prompt.split())  # Very rough estimate
        completion_tokens = len(generated_text.split())
        
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            text=generated_text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            model=MODEL_TYPE,
            inference_time=inference_time
        )
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 