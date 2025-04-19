import asyncio
import importlib
import json
import os
import sys
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from uuid import uuid4

import torch
import yaml
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Request,
                     status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

# Import common logging
from common.logging import get_tracer, setup_logger, setup_tracing
from common.logging.context import RequestContextMiddleware
from common.logging.middleware import RequestLoggingMiddleware
# Import OpenTelemetry instrumentation
from common.logging.tracing import instrument_fastapi, instrument_httpx_client
# Import internal modules
from inference_service.memory_integration import (get_context_for_prompt,
                                                  get_mock_context)
from inference_service.metrics import get_metrics_exporter, setup_metrics
from models.gpu_optimizations import (ParallelMode, QuantizationMode,
                                      clear_gpu_memory, optimize_cuda_memory)

# Initialize service name
SERVICE_NAME = "inference-service"

# Setup logging and tracing
logger = setup_logger(
    service_name=SERVICE_NAME,
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    log_file=os.environ.get("LOG_FILE"),
    json_logs=os.environ.get("JSON_LOGS", "").lower() == "true",
)

# Setup tracing
tracer_provider = setup_tracing(
    service_name=SERVICE_NAME,
    otlp_endpoint=os.environ.get("OTLP_ENDPOINT"),
    debug=os.environ.get("DEBUG_TRACING", "").lower() == "true",
)

# Get module tracer
tracer = get_tracer(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deep Recall Inference Service",
    description="API for LLM inference with DeepSeek R1 and other models",
)

# Add request context middleware
app.add_middleware(RequestContextMiddleware)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware, service_name=SERVICE_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
LOADED_MODELS: Dict[str, Any] = {}
DEFAULT_MODEL: Optional[str] = None
CONFIG: Optional[Dict[str, Any]] = None
MODEL_CACHE_DIR: Optional[str] = None
MODEL_CONFIG_PATH: str = os.path.join(
    os.path.dirname(__file__), "config", "models.yaml"
)
MODEL_TYPE: str = "deepseek_r1"  # Default model type

# Load configuration
try:
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, "r") as f:
            CONFIG = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {MODEL_CONFIG_PATH}")
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    CONFIG = {}

# Apply CUDA memory optimizations on startup
if torch.cuda.is_available():
    optimize_cuda_memory()
    logger.info(
        f"Applied CUDA memory optimizations. Available GPUs: {torch.cuda.device_count()}"
    )

# Initialize metrics
metrics_port = int(os.environ.get("METRICS_PORT", "8000"))
metrics_exporter = setup_metrics(port=metrics_port)
logger.info(f"Metrics exporter initialized on port {metrics_port}")


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(
        1024, ge=1, le=8192, description="Maximum number of tokens to generate"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(
        0.0, ge=0.0, le=2.0, description="Penalty for token frequency"
    )
    presence_penalty: float = Field(
        0.0, ge=0.0, le=2.0, description="Penalty for token presence"
    )
    stop_sequences: Optional[List[str]] = Field(
        None, description="Sequences where generation should stop"
    )
    model: Optional[str] = Field(None, description="Model to use for inference")
    context: Optional[List[Dict[str, str]]] = Field(
        None, description="Context messages for the conversation"
    )
    stream: bool = Field(False, description="Whether to stream the response")

    @validator("prompt")
    def validate_prompt_length(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        return v


class BatchInferenceRequest(BaseModel):
    prompts: List[str] = Field(
        ..., min_items=1, max_items=10, description="List of prompts to process"
    )
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
    use_mock_memory: bool = Field(
        False, description="Whether to use mock memory for testing"
    )
    max_memories: int = Field(
        5, ge=0, le=10, description="Maximum number of memories to retrieve"
    )


class ModelLoadRequest(BaseModel):
    model_name: str
    quantization: Optional[str] = Field(None, description="Quantization mode")
    parallel_mode: Optional[str] = Field(None, description="Model parallelism mode")
    gpu_ids: Optional[List[int]] = Field(None, description="Specific GPU IDs to use")
    max_memory: Optional[Dict[int, str]] = Field(
        None, description="Maximum memory per GPU"
    )
    prefer_gpu: bool = Field(True, description="Whether to prefer GPU over CPU")


class GPUStatusResponse(BaseModel):
    available: bool
    count: int
    devices: List[Dict[str, Any]]
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float


def get_model_by_name(model_name: Optional[str] = None):
    """
    Get the specified model or the default model
    """
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


async def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    parallel_mode: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,
    max_memory: Optional[Dict[int, str]] = None,
):
    """
    Load a model with the specified configuration
    """
    try:
        if model_name == "deepseek_r1":
            # Import the model module
            from models.deepseek_r1 import DeepSeekR1

            # Create model instance
            model_instance = DeepSeekR1(
                quantization=quantization,
                parallel_mode=parallel_mode,
                available_gpus=gpu_ids,
                max_memory_per_gpu=max_memory,
            )

            # Update global model state
            LOADED_MODELS[model_name] = model_instance

            # Set as default if no default exists or if this is deepseek_r1
            global DEFAULT_MODEL
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
    Initialize models and instrumentation on startup
    """
    # Import OpenTelemetry instrumentation
    from common.logging.tracing import (instrument_fastapi,
                                        instrument_httpx_client)

    # Instrument FastAPI application
    instrument_fastapi(app)

    # Create and instrument HTTPX client if needed
    import httpx

    client = httpx.AsyncClient()
    instrument_httpx_client(client)

    # Log startup information
    logger.info(f"Starting {SERVICE_NAME} with OpenTelemetry tracing")

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
            global DEFAULT_MODEL
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

    logger.info(f"Startup completed. Default model: {DEFAULT_MODEL or 'None'}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    if not LOADED_MODELS:
        return {
            "status": "unhealthy",
            "message": "No models loaded",
            "model": "none",
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "loaded_models": [],
            "version": getattr(
                importlib.import_module("inference_service"), "__version__", "unknown"
            ),
        }

    return {
        "status": "healthy",
        "model": DEFAULT_MODEL or "none",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "loaded_models": list(LOADED_MODELS.keys()),
        "version": getattr(
            importlib.import_module("inference_service"), "__version__", "unknown"
        ),
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List all available models and their status
    """
    try:
        models = []
        for model_name, config in CONFIG.get("models", {}).items():
            model_info = ModelInfo(
                model_id=model_name,
                revision=config.get("revision"),
                quantization=config.get("quantization"),
                max_sequence_length=config.get("max_sequence_length", 2048),
                loaded=model_name in LOADED_MODELS,
                default=model_name == DEFAULT_MODEL,
            )
            models.append(model_info)

        return ModelsResponse(models=models, default_model=DEFAULT_MODEL or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/gpu_status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """
    Get GPU status and usage information
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "count": 0,
            "devices": [],
            "total_memory_gb": 0,
            "used_memory_gb": 0,
            "free_memory_gb": 0,
        }

    # Get GPU information
    gpu_count = torch.cuda.device_count()
    devices = []
    total_memory = 0
    used_memory = 0

    for i in range(gpu_count):
        prop = torch.cuda.get_device_properties(i)
        total_mem = prop.total_memory / (1024**3)  # Convert to GB
        reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
        free_mem = total_mem - reserved_mem

        total_memory += total_mem
        used_memory += allocated_mem

        devices.append(
            {
                "id": i,
                "name": prop.name,
                "total_memory_gb": round(total_mem, 2),
                "reserved_memory_gb": round(reserved_mem, 2),
                "allocated_memory_gb": round(allocated_mem, 2),
                "free_memory_gb": round(free_mem, 2),
                "compute_capability": f"{prop.major}.{prop.minor}",
            }
        )

    return {
        "available": True,
        "count": gpu_count,
        "devices": devices,
        "total_memory_gb": round(total_memory, 2),
        "used_memory_gb": round(used_memory, 2),
        "free_memory_gb": round(total_memory - used_memory, 2),
    }


@app.post("/v1/models/{model_name}/load", status_code=status.HTTP_202_ACCEPTED)
async def load_model_endpoint(
    model_name: str, request: ModelLoadRequest, background_tasks: BackgroundTasks
):
    """
    Load a model with the specified configuration
    """
    try:
        if model_name in LOADED_MODELS:
            return {
                "status": "success",
                "message": f"Model {model_name} is already loaded",
            }

        background_tasks.add_task(
            load_model,
            model_name=model_name,
            quantization=request.quantization,
            parallel_mode=request.parallel_mode,
            gpu_ids=request.gpu_ids,
            max_memory=request.max_memory,
        )

        return {"status": "success", "message": f"Loading model {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/clear_gpu_memory")
async def clear_gpu_memory_endpoint():
    """
    Clear GPU memory and unload all models
    """
    try:
        global DEFAULT_MODEL
        clear_gpu_memory()
        LOADED_MODELS.clear()
        DEFAULT_MODEL = None
        return {
            "status": "success",
            "message": "GPU memory cleared and models unloaded",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/models/{model_name}/offload")
async def offload_model_to_cpu(model_name: str):
    """
    Offload a model to CPU memory
    """
    try:
        if model_name not in LOADED_MODELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not loaded",
            )

        model = LOADED_MODELS[model_name]
        model.to("cpu")

        return {"status": "success", "message": f"Model {model_name} offloaded to CPU"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/models/{model_name}/reload")
async def reload_model_to_gpu(model_name: str):
    """
    Reload a model back to GPU memory
    """
    try:
        if model_name not in LOADED_MODELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not loaded",
            )

        model = LOADED_MODELS[model_name]
        model.to("cuda")

        return {"status": "success", "message": f"Model {model_name} reloaded to GPU"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_stream_response(
    model, request: InferenceRequest
) -> AsyncIterator[str]:
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
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
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
            top_p=request.top_p,
        )

        # Split into simulated chunks (for demo purposes)
        chunks = []
        for i in range(0, len(full_text), 10):
            if i + 10 < len(full_text):
                chunks.append(full_text[i : i + 10])
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
                    {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                ],
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
                {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "inference_time": inference_time,
        }
        yield f"data: {json.dumps(completion_payload)}\n\n"
        yield f"data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming inference error: {str(e)}")
        error_payload = {"request_id": request_id, "error": str(e), "streaming": True}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield f"data: [DONE]\n\n"


@app.post("/v1/completions", response_model=InferenceResponse)
async def generate_completion(request: InferenceRequest):
    """
    Generate a completion for a single prompt
    """
    # Get model
    model = get_model_by_name(request.model)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No model available"
        )

    # Generate completion
    start_time = time.time()
    response = await generate_stream_response(model, request)
    end_time = time.time()

    return InferenceResponse(
        text=response,
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },  # TODO: Implement token counting
        model=request.model or DEFAULT_MODEL or "unknown",
        inference_time=end_time - start_time,
    )


@app.post("/v1/batch_completions", response_model=BatchInferenceResponse)
async def batch_generate_completions(request: BatchInferenceRequest):
    """
    Generate completions for multiple prompts
    """
    # Get model
    model = get_model_by_name(request.model)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No model available"
        )

    # Generate completions
    start_time = time.time()
    responses = []
    for prompt in request.prompts:
        single_request = InferenceRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            model=request.model,
        )
        response = await generate_stream_response(model, single_request)
        responses.append(
            InferenceResponse(
                text=response,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },  # TODO: Implement token counting
                model=request.model or DEFAULT_MODEL or "unknown",
                inference_time=0,  # Individual times not tracked in batch
            )
        )
    end_time = time.time()

    return BatchInferenceResponse(responses=responses, total_time=end_time - start_time)


@app.post("/v1/personalized_completions", response_model=InferenceResponse)
async def generate_personalized_completion(request: PersonalizedInferenceRequest):
    """
    Generate a completion with personalized context from memory
    """
    # Get model
    model = get_model_by_name(request.model)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No model available"
        )

    # Get memories for user
    memories = []
    if not request.use_mock_memory:
        # TODO: Implement memory retrieval
        pass

    # Add memories to context
    context = request.context or []
    for memory in memories[: request.max_memories]:
        context.append(
            {"role": "system", "content": f"Previous interaction: {memory['content']}"}
        )

    # Create inference request with context
    inference_request = InferenceRequest(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        model=request.model,
        context=context,
        stream=request.stream,
    )

    # Generate completion
    start_time = time.time()
    response = await generate_stream_response(model, inference_request)
    end_time = time.time()

    return InferenceResponse(
        text=response,
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },  # TODO: Implement token counting
        model=request.model or DEFAULT_MODEL or "unknown",
        inference_time=end_time - start_time,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
