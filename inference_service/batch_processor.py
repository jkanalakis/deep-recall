"""
Batch processor for Deep Recall inference service.

This module implements dynamic batching to improve GPU utilization
and throughput for LLM inference requests.
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Awaitable
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import PriorityQueue, Empty
import torch

logger = logging.getLogger(__name__)

# Configuration from environment variables with defaults
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
MAX_WAITING_TIME_MS = int(os.environ.get("MAX_WAITING_TIME_MS", "100"))
ENABLE_BATCH_PREFETCH = os.environ.get("ENABLE_BATCH_PREFETCH", "false").lower() == "true"
ENABLE_DYNAMIC_BATCHING = os.environ.get("ENABLE_DYNAMIC_BATCHING", "true").lower() == "true"


@dataclass
class InferenceRequest:
    """A request for model inference."""
    id: str  # Unique request ID
    prompt: str
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    context: Optional[List[Dict[str, str]]] = None
    priority: int = 0  # Higher number = higher priority
    timestamp: float = 0.0  # Time when the request was added
    future: Optional[asyncio.Future] = None


@dataclass
class InferenceResponse:
    """A response from model inference."""
    request_id: str
    text: str
    usage: Dict[str, int]
    model: str
    inference_time: float
    error: Optional[str] = None


class BatchProcessor:
    """
    Processes inference requests in batches for improved throughput.
    """
    def __init__(self, model_registry, max_batch_size: int = MAX_BATCH_SIZE,
                 max_waiting_time_ms: int = MAX_WAITING_TIME_MS,
                 enable_dynamic_batching: bool = ENABLE_DYNAMIC_BATCHING,
                 enable_prefetch: bool = ENABLE_BATCH_PREFETCH):
        """
        Initialize the batch processor.
        
        Args:
            model_registry: Registry of available models
            max_batch_size: Maximum batch size for inference
            max_waiting_time_ms: Maximum time to wait for batching in milliseconds
            enable_dynamic_batching: Whether to enable dynamic batching
            enable_prefetch: Whether to prefetch batches
        """
        self.model_registry = model_registry
        self.max_batch_size = max_batch_size
        self.max_waiting_time_ms = max_waiting_time_ms
        self.enable_dynamic_batching = enable_dynamic_batching
        self.enable_prefetch = enable_prefetch
        
        # Priority queue for requests
        self.request_queue = PriorityQueue()
        
        # Dictionary to track batches by model
        self.batches: Dict[str, List[InferenceRequest]] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Thread pool for processing batches
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Background task for processing batches
        self.processing_task = None
        self.running = False
        
        # Start the batch processor
        self.start()
        
        logger.info(f"Batch processor initialized with max_batch_size={max_batch_size}, "
                   f"max_waiting_time_ms={max_waiting_time_ms}, "
                   f"enable_dynamic_batching={enable_dynamic_batching}, "
                   f"enable_prefetch={enable_prefetch}")

    def start(self):
        """Start the batch processor."""
        if not self.running:
            self.running = True
            self.processing_task = asyncio.create_task(self._process_batches())
            logger.info("Batch processor started")

    async def stop(self):
        """Stop the batch processor."""
        if self.running:
            self.running = False
            if self.processing_task:
                try:
                    self.processing_task.cancel()
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            self.executor.shutdown(wait=False)
            logger.info("Batch processor stopped")

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """
        Submit a request for processing.
        
        Args:
            request: The inference request
            
        Returns:
            InferenceResponse: The inference response
        """
        # Create a future to wait for the result
        request.future = asyncio.Future()
        request.timestamp = time.time()
        
        # Add the request to the queue
        self.request_queue.put((
            -request.priority,  # Negative because PriorityQueue returns smallest first
            request.timestamp,
            request
        ))
        
        # Wait for the result
        return await request.future

    async def _process_batches(self):
        """Process batches of requests."""
        while self.running:
            try:
                # Collect requests into batches
                await self._collect_batch()
                
                # Process all ready batches
                await self._process_ready_batches()
                
                # Short sleep to avoid busy waiting
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)  # Avoid tight loop in case of errors

    async def _collect_batch(self):
        """Collect requests into batches by model."""
        # Check if we should collect more requests
        batch_ready = False
        
        # Process all requests in the queue up to max_batch_size
        for _ in range(self.max_batch_size):
            try:
                # Get a request with a timeout of 1ms
                _, _, request = self.request_queue.get_nowait()
                
                # Get or create batch for this model
                with self.lock:
                    if request.model_name not in self.batches:
                        self.batches[request.model_name] = []
                    
                    # Add request to batch
                    self.batches[request.model_name].append(request)
                    
                    # Check if batch is ready
                    if len(self.batches[request.model_name]) >= self.max_batch_size:
                        batch_ready = True
                
                # If batch is ready and we're not using dynamic batching, break
                if batch_ready and not self.enable_dynamic_batching:
                    break
            
            except Empty:
                # No more requests in queue
                break
            except Exception as e:
                logger.error(f"Error collecting batch: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)  # Avoid tight loop in case of errors

    async def _process_ready_batches(self):
        """Process all batches that are ready."""
        ready_models = []
        
        # Check for ready batches
        with self.lock:
            current_time = time.time()
            for model_name, batch in self.batches.items():
                if not batch:
                    continue
                
                # Check if batch is full or has been waiting too long
                batch_size = len(batch)
                oldest_request_time = min(req.timestamp for req in batch)
                waiting_time_ms = (current_time - oldest_request_time) * 1000
                
                if (batch_size >= self.max_batch_size or
                    waiting_time_ms >= self.max_waiting_time_ms):
                    ready_models.append(model_name)
        
        # Process ready batches
        for model_name in ready_models:
            with self.lock:
                batch = self.batches.pop(model_name, [])
            
            if batch:
                # Process batch in a separate task to avoid blocking
                asyncio.create_task(self._process_batch(model_name, batch))

    async def _process_batch(self, model_name: str, batch: List[InferenceRequest]):
        """
        Process a batch of requests.
        
        Args:
            model_name: Name of the model to use
            batch: List of requests to process
        """
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Get the model
            model = self.model_registry.get(model_name)
            if model is None:
                # Try to load the model
                try:
                    await self.model_registry.load_model(model_name)
                    model = self.model_registry.get(model_name)
                except Exception as e:
                    # Failed to load model, return error for all requests
                    error_msg = f"Failed to load model {model_name}: {str(e)}"
                    logger.error(error_msg)
                    for request in batch:
                        if not request.future.done():
                            request.future.set_result(InferenceResponse(
                                request_id=request.id,
                                text="",
                                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                model=model_name,
                                inference_time=0.0,
                                error=error_msg
                            ))
                    return
            
            # Prepare batch inputs
            prompts = [req.prompt for req in batch]
            contexts = [req.context for req in batch] if all(req.context is not None for req in batch) else None
            
            # Use the first request's parameters for the entire batch
            max_tokens = batch[0].max_tokens
            temperature = batch[0].temperature
            top_p = batch[0].top_p
            
            # Process batch with model
            batch_start_time = time.time()
            
            # Get batch responses using the model's batch_generate_replies method
            try:
                responses = await asyncio.to_thread(
                    model.batch_generate_replies,
                    prompts,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    contexts=contexts
                )
                
                batch_time = (time.time() - batch_start_time) * 1000  # ms
                
                # Calculate token counts (approximate)
                prompt_tokens = sum(len(prompt.split()) for prompt in prompts)
                completion_tokens = sum(len(response.split()) for response in responses)
                
                # Set results
                for i, (request, response_text) in enumerate(zip(batch, responses)):
                    # Estimate individual tokens
                    req_prompt_tokens = len(request.prompt.split())
                    req_completion_tokens = len(response_text.split())
                    
                    # Calculate per-request latency (approximate)
                    per_request_time = batch_time / len(batch)
                    
                    if not request.future.done():
                        request.future.set_result(InferenceResponse(
                            request_id=request.id,
                            text=response_text,
                            usage={
                                "prompt_tokens": req_prompt_tokens,
                                "completion_tokens": req_completion_tokens,
                                "total_tokens": req_prompt_tokens + req_completion_tokens
                            },
                            model=model_name,
                            inference_time=per_request_time / 1000  # Convert to seconds
                        ))
                
                # Log batch processing statistics
                logger.info(f"Processed batch of {len(batch)} requests for model {model_name} "
                           f"in {batch_time:.2f}ms ({batch_time / len(batch):.2f}ms per request)")
                
            except Exception as e:
                logger.error(f"Error processing batch for model {model_name}: {str(e)}", exc_info=True)
                # Set error for all requests in batch
                for request in batch:
                    if not request.future.done():
                        request.future.set_result(InferenceResponse(
                            request_id=request.id,
                            text="",
                            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            model=model_name,
                            inference_time=0.0,
                            error=str(e)
                        ))
        
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
            # Set error for all requests in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_result(InferenceResponse(
                        request_id=request.id,
                        text="",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        model=model_name,
                        inference_time=0.0,
                        error=str(e)
                    ))

# Singleton instance
_batch_processor = None

def get_batch_processor(model_registry=None):
    """
    Get or create the batch processor instance.
    
    Args:
        model_registry: Registry of available models
        
    Returns:
        BatchProcessor: The batch processor instance
    """
    global _batch_processor
    if _batch_processor is None and model_registry is not None:
        _batch_processor = BatchProcessor(model_registry)
    return _batch_processor 