"""
Metrics exporter for Deep Recall Inference Service

This module provides metrics collection and export for the inference service,
particularly focused on metrics used for horizontal pod autoscaling.
"""

import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import torch
import logging
import os
from collections import deque

logger = logging.getLogger(__name__)

# Global metrics
# Queue metrics
QUEUE_LENGTH = Gauge('inference_queue_length', 'Number of requests in the inference queue')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests being processed')

# Performance metrics
INFERENCE_LATENCY = Histogram('inference_latency_ms', 'Inference latency in milliseconds', 
                             buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
TOKEN_GENERATION_SPEED = Gauge('tokens_per_second', 'Token generation speed (tokens/sec)')
PROMPT_TOKEN_COUNT = Counter('prompt_tokens_total', 'Total number of prompt tokens processed')
COMPLETION_TOKEN_COUNT = Counter('completion_tokens_total', 'Total number of completion tokens generated')

# Resource utilization metrics
GPU_UTILIZATION = Gauge('nvidia_gpu_utilization', 'NVIDIA GPU utilization percentage', ['gpu_id'])
GPU_MEMORY_USAGE = Gauge('nvidia_gpu_memory_used_bytes', 'NVIDIA GPU memory used in bytes', ['gpu_id'])
MODEL_MEMORY_USAGE = Gauge('model_memory_bytes', 'Memory used by model in bytes')
MEMORY_PER_REQUEST = Gauge('memory_per_request_mb', 'Average memory usage per request in MB')

# Request metrics
REQUEST_COUNTER = Counter('inference_requests_total', 'Total number of inference requests', ['model', 'endpoint'])
BATCH_SIZE = Histogram('batch_size', 'Batch sizes used for inference', buckets=[1, 2, 4, 8, 16, 32])
REQUEST_RATE = Gauge('requests_per_second', 'Number of requests per second')

# Error metrics
ERROR_COUNTER = Counter('inference_errors_total', 'Total number of inference errors', ['error_type'])

# Moving average window for calculating rates
RATE_WINDOW_SIZE = 30  # 30 seconds for moving average
request_times = deque(maxlen=RATE_WINDOW_SIZE)
token_counts = deque(maxlen=RATE_WINDOW_SIZE)

class MetricsExporter:
    """
    Metrics exporter for Deep Recall Inference Service
    """
    def __init__(self, port: int = 8000, path: str = "/metrics"):
        """
        Initialize the metrics exporter
        
        Args:
            port: Port to expose metrics on
            path: Path to expose metrics on
        """
        self.port = port
        self.path = path
        self.running = False
        self.update_thread = None
        self.lock = threading.Lock()
        
        # Initialize additional tracking state
        self.queue_length = 0
        self.active_requests = 0
        
        # Start the metrics server
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
        
        # Start the update thread for resource metrics
        self.start_update_thread()
    
    def start_update_thread(self):
        """Start the update thread for resource metrics"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_resource_metrics, daemon=True)
        self.update_thread.start()
    
    def stop_update_thread(self):
        """Stop the update thread for resource metrics"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
    
    def _update_resource_metrics(self):
        """Update resource metrics in a background thread"""
        while self.running:
            try:
                # Update GPU metrics if GPU is available
                if torch.cuda.is_available():
                    # Get the number of GPUs
                    gpu_count = torch.cuda.device_count()
                    
                    # Update metrics for each GPU
                    for gpu_id in range(gpu_count):
                        # Use NVIDIA Management Library (NVML) via torch.cuda
                        gpu_utilization = 0
                        gpu_memory_used = torch.cuda.memory_allocated(gpu_id)
                        
                        # Update metrics
                        GPU_UTILIZATION.labels(gpu_id=str(gpu_id)).set(gpu_utilization)
                        GPU_MEMORY_USAGE.labels(gpu_id=str(gpu_id)).set(gpu_memory_used)
                
                # Calculate request rate
                self._calculate_request_rate()
                
                # Calculate token generation speed
                self._calculate_token_generation_speed()
                
            except Exception as e:
                logger.error(f"Error updating resource metrics: {str(e)}")
            
            # Sleep for a short time
            time.sleep(1.0)
    
    def _calculate_request_rate(self):
        """Calculate the current request rate"""
        if not request_times:
            REQUEST_RATE.set(0)
            return
        
        # Get current time
        now = time.time()
        
        # Remove old entries
        while request_times and now - request_times[0] > RATE_WINDOW_SIZE:
            request_times.popleft()
        
        # Calculate rate
        if len(request_times) > 1:
            time_span = now - request_times[0]
            if time_span > 0:
                rate = len(request_times) / time_span
                REQUEST_RATE.set(rate)
    
    def _calculate_token_generation_speed(self):
        """Calculate the token generation speed"""
        if not token_counts:
            TOKEN_GENERATION_SPEED.set(0)
            return
        
        # Get current time
        now = time.time()
        
        # Calculate total tokens in window
        total_tokens = sum(count for _, count in token_counts)
        
        # Calculate time span
        if len(token_counts) > 1:
            time_span = token_counts[-1][0] - token_counts[0][0]
            if time_span > 0:
                speed = total_tokens / time_span
                TOKEN_GENERATION_SPEED.set(speed)
    
    def track_request_start(self, model: str, endpoint: str):
        """
        Track the start of a request
        
        Args:
            model: Model name
            endpoint: API endpoint
        """
        with self.lock:
            self.queue_length += 1
            QUEUE_LENGTH.set(self.queue_length)
        
        REQUEST_COUNTER.labels(model=model, endpoint=endpoint).inc()
        request_times.append(time.time())
    
    def track_request_processing(self, batch_size: int = 1):
        """
        Track a request moving from queue to processing
        
        Args:
            batch_size: Batch size for the request
        """
        with self.lock:
            self.queue_length -= 1
            self.active_requests += 1
            QUEUE_LENGTH.set(self.queue_length)
            ACTIVE_REQUESTS.set(self.active_requests)
        
        BATCH_SIZE.observe(batch_size)
    
    def track_request_complete(self, latency_ms: float, prompt_tokens: int, completion_tokens: int):
        """
        Track the completion of a request
        
        Args:
            latency_ms: Latency in milliseconds
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        with self.lock:
            self.active_requests -= 1
            ACTIVE_REQUESTS.set(self.active_requests)
        
        INFERENCE_LATENCY.observe(latency_ms)
        PROMPT_TOKEN_COUNT.inc(prompt_tokens)
        COMPLETION_TOKEN_COUNT.inc(completion_tokens)
        
        # Track tokens for generation speed calculation
        token_counts.append((time.time(), completion_tokens))
    
    def track_error(self, error_type: str):
        """
        Track an error
        
        Args:
            error_type: Type of error
        """
        ERROR_COUNTER.labels(error_type=error_type).inc()
    
    def update_memory_usage(self, model_memory_bytes: int, per_request_mb: float):
        """
        Update memory usage metrics
        
        Args:
            model_memory_bytes: Memory used by model in bytes
            per_request_mb: Average memory per request in MB
        """
        MODEL_MEMORY_USAGE.set(model_memory_bytes)
        MEMORY_PER_REQUEST.set(per_request_mb)


# Create a global metrics exporter instance
metrics_exporter = None

def setup_metrics(port: int = 8000):
    """
    Set up the metrics exporter
    
    Args:
        port: Port to expose metrics on
    """
    global metrics_exporter
    if metrics_exporter is None:
        metrics_exporter = MetricsExporter(port=port)
    return metrics_exporter

def get_metrics_exporter() -> MetricsExporter:
    """
    Get the metrics exporter instance
    
    Returns:
        MetricsExporter instance
    """
    global metrics_exporter
    if metrics_exporter is None:
        metrics_exporter = setup_metrics()
    return metrics_exporter 