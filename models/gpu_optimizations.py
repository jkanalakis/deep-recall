"""
GPU Optimization Utilities for Deep Recall

This module provides utilities for optimizing LLM inference on GPUs,
including quantization, tensor parallelism, and memory management.
"""

import gc
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class QuantizationMode(str, Enum):
    """Supported quantization modes"""

    NONE = "none"  # No quantization
    INT8 = "int8"  # 8-bit quantization
    INT4 = "int4"  # 4-bit quantization
    GPTQ = "gptq"  # GPTQ quantization
    AWQ = "awq"  # AWQ (Activation-aware Weight Quantization)


class ParallelMode(str, Enum):
    """Model parallelism modes"""

    NONE = "none"  # No parallelism
    TENSOR = "tensor"  # Tensor parallelism
    PIPELINE = "pipeline"  # Pipeline parallelism
    EXPERT = "expert"  # Expert parallelism (MoE)


def optimize_cuda_memory():
    """
    Configure PyTorch for optimal CUDA memory usage
    """
    # Enable CUDA memory allocation caching
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Enable TF32 precision for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set CUDA stream priority for more efficient scheduling
    if hasattr(torch.cuda, "Stream"):
        torch.cuda.Stream(priority=-1)


def get_device_map(
    model_size_gb: float, available_gpus: Optional[List[int]] = None
) -> Dict[str, Union[int, str]]:
    """
    Create an optimal device map for model parallelism based on available GPUs and model size

    Args:
        model_size_gb: Estimated model size in gigabytes
        available_gpus: List of GPU indices to use (default: all available)

    Returns:
        Device map dictionary for HuggingFace's .from_pretrained()
    """
    if available_gpus is None:
        available_gpus = list(range(torch.cuda.device_count()))

    if not available_gpus:
        return {"": "cpu"}  # CPU only

    if len(available_gpus) == 1:
        # Single GPU - check if model fits
        gpu_id = available_gpus[0]
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (
            1024**3
        )  # Convert to GB

        if model_size_gb < gpu_memory * 0.8:  # Leave 20% margin
            return {"": gpu_id}  # Entire model on one GPU
        else:
            # Model doesn't fit on GPU, use disk offloading
            return {"": "auto"}

    # Multiple GPUs available - create a balanced distribution
    total_gpu_memory = sum(
        torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        for gpu_id in available_gpus
    )

    if model_size_gb < total_gpu_memory * 0.8:
        # Model can be distributed across available GPUs
        return {"": "auto"}  # Let HF decide the optimal distribution
    else:
        # Model is too large even for all GPUs, use disk offloading
        return {"": "auto", "offload_folder": "offload_folder"}


def clear_gpu_memory():
    """
    Clear CUDA cache and garbage collect to free GPU memory
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_batch_size(
    model_size_gb: float, sequence_length: int, dtype: torch.dtype = torch.float16
) -> int:
    """
    Calculate optimal batch size based on available GPU memory

    Args:
        model_size_gb: Model size in gigabytes
        sequence_length: Maximum sequence length in tokens
        dtype: Data type used for computations

    Returns:
        Optimal batch size for inference
    """
    if not torch.cuda.is_available():
        return 1  # Default to 1 for CPU

    # Get available GPU memory
    gpu_id = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    reserved_memory = torch.cuda.memory_reserved(gpu_id)
    allocated_memory = torch.cuda.memory_allocated(gpu_id)
    free_memory = total_memory - reserved_memory - allocated_memory

    # Convert to GB
    free_memory_gb = free_memory / (1024**3)

    # Estimate memory per token based on model size and data type
    bytes_per_parameter = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)

    # Estimate memory needed for KV cache per token
    kv_cache_memory_per_token = model_size_gb * 0.012  # Empirical factor

    # Calculate memory needed per sequence
    memory_per_sequence = kv_cache_memory_per_token * sequence_length

    # Calculate optimal batch size, leaving 20% memory as buffer
    optimal_batch_size = int((free_memory_gb * 0.8) / memory_per_sequence)

    return max(1, optimal_batch_size)


def setup_tensor_parallelism(num_gpus: Optional[int] = None) -> bool:
    """
    Set up tensor parallelism for distributed inference

    Args:
        num_gpus: Number of GPUs to use for tensor parallelism (default: all available)

    Returns:
        Whether tensor parallelism was successfully set up
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot set up tensor parallelism")
        return False

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        logger.info("Only one GPU available, tensor parallelism not needed")
        return False

    try:
        # Initialize process group for distributed processing
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://", world_size=num_gpus, rank=0
            )

        # Set up device for current process
        local_rank = 0  # Assuming single node, primary process
        torch.cuda.set_device(local_rank)

        logger.info(f"Tensor parallelism set up with {num_gpus} GPUs")
        return True

    except Exception as e:
        logger.error(f"Failed to set up tensor parallelism: {str(e)}")
        return False


def estimate_model_size(
    model_id: str, num_layers: Optional[int] = None, hidden_size: Optional[int] = None
) -> float:
    """
    Estimate model size in GB based on model id or architecture parameters

    Args:
        model_id: Hugging Face model ID or name
        num_layers: Number of transformer layers (optional)
        hidden_size: Hidden dimension size (optional)

    Returns:
        Estimated model size in GB
    """
    # Common model size estimates
    model_sizes = {
        "deepseek-ai/deepseek-coder-7b": 14.0,
        "deepseek-ai/deepseek-coder-33b": 66.0,
        "llama2-7b": 14.0,
        "llama2-13b": 26.0,
        "llama2-70b": 140.0,
        "llama3-8b": 16.0,
        "llama3-70b": 140.0,
    }

    # Try to find the model in known sizes
    for known_id, size in model_sizes.items():
        if known_id.lower() in model_id.lower():
            return size

    # If model parameters are provided, estimate size
    if num_layers and hidden_size:
        # Approximate formula for transformer-based LLMs
        vocab_size = 32000  # Typical vocabulary size
        intermediate_size = hidden_size * 4  # Typical intermediate size

        # Parameters in embedding layers
        embedding_params = vocab_size * hidden_size

        # Parameters per layer
        params_per_layer = (
            # Self-attention
            3 * hidden_size * hidden_size  # QKV projections
            + hidden_size * hidden_size  # Output projection
            +
            # FFN
            hidden_size * intermediate_size  # Up projection
            + intermediate_size * hidden_size  # Down projection
            +
            # Layer norms
            2 * hidden_size  # 2 layer norms (pre-attention, pre-FFN)
            +
            # Biases
            5 * hidden_size
            + intermediate_size  # Attention + FFN biases
        )

        # Total parameters
        total_params = (
            embedding_params
            + (params_per_layer * num_layers)
            + hidden_size * vocab_size
        )

        # Convert to GB (assuming float16/bfloat16 - 2 bytes per parameter)
        return (total_params * 2) / (1024**3)

    # Default fallback - medium-sized model
    return 16.0
