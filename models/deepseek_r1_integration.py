# models/deepseek_r1_integration.py

# Pseudocode demonstrating how you might wrap an LLM

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.gpu_optimizations import (
    ParallelMode,
    QuantizationMode,
    clear_gpu_memory,
    estimate_model_size,
    get_device_map,
    get_optimal_batch_size,
    optimize_cuda_memory,
)

logger = logging.getLogger(__name__)


class DeepSeekR1Model:
    def __init__(
        self,
        model_path: str,
        quantization: Union[str, QuantizationMode] = QuantizationMode.NONE,
        parallel_mode: Union[str, ParallelMode] = ParallelMode.NONE,
        available_gpus: Optional[List[int]] = None,
        max_memory_per_gpu: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the DeepSeek R1 model with GPU optimizations

        Args:
            model_path: Path to model weights or HF model ID
            quantization: Quantization mode to use
            parallel_mode: Parallelism mode to use
            available_gpus: List of GPU indices to use
            max_memory_per_gpu: Maximum memory per GPU in format {0: "14GiB", 1: "12GiB"}
        """
        self.model_path = model_path
        self.model_id = "deepseek-ai/deepseek-coder-7b-instruct"  # Default model

        # Initialize quantization and parallelism settings
        self.quantization = (
            quantization
            if isinstance(quantization, QuantizationMode)
            else QuantizationMode(quantization)
        )
        self.parallel_mode = (
            parallel_mode
            if isinstance(parallel_mode, ParallelMode)
            else ParallelMode(parallel_mode)
        )

        # Check if custom model path specified in env
        custom_model = os.environ.get("DEEPSEEK_MODEL_ID")
        if custom_model:
            self.model_id = custom_model

        # Apply CUDA memory optimizations
        optimize_cuda_memory()

        # Configure device and parallelism
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_gpus = available_gpus
        self.max_memory_per_gpu = max_memory_per_gpu

        # Calculate model size estimate for device mapping
        self.model_size_gb = estimate_model_size(self.model_id)

        # Log configuration
        logger.info(f"Initializing DeepSeek model with:")
        logger.info(f"  - Model ID: {self.model_id}")
        logger.info(f"  - Quantization: {self.quantization.value}")
        logger.info(f"  - Parallel mode: {self.parallel_mode.value}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Model size (est.): {self.model_size_gb:.2f} GB")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Load model with appropriate settings
        self._load_model()

        # Set optimal batch size
        self.optimal_batch_size = get_optimal_batch_size(
            model_size_gb=self.model_size_gb,
            sequence_length=2048,  # Default sequence length
            dtype=self.model.dtype if hasattr(self.model, "dtype") else torch.float16,
        )

        logger.info(
            f"DeepSeek R1 model loaded with optimal batch size: {self.optimal_batch_size}"
        )

    def _load_model(self):
        """
        Load the model with the specified optimizations
        """
        load_kwargs = {
            "trust_remote_code": True,
        }

        # Configure quantization
        if self.quantization == QuantizationMode.INT8:
            load_kwargs["load_in_8bit"] = True
        elif self.quantization == QuantizationMode.INT4:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
            load_kwargs["bnb_4bit_use_double_quant"] = True
        elif self.quantization in [QuantizationMode.GPTQ, QuantizationMode.AWQ]:
            # For GPTQ/AWQ, we need to ensure the quantized model exists
            if self.quantization == QuantizationMode.GPTQ:
                load_kwargs["quantization_config"] = {"bits": 4, "group_size": 128}
            else:  # AWQ
                load_kwargs["quantization_config"] = {"bits": 4}

        # Configure device map for model parallelism
        if self.device == "cuda":
            # Setup device map based on model size and available GPUs
            if self.parallel_mode == ParallelMode.TENSOR:
                # Use tensor parallelism by letting HF handle the distribution
                load_kwargs["device_map"] = "auto"

                # Specify max memory per GPU if provided
                if self.max_memory_per_gpu:
                    load_kwargs["max_memory"] = self.max_memory_per_gpu
            else:
                # Use auto device map based on model size
                load_kwargs["device_map"] = get_device_map(
                    model_size_gb=self.model_size_gb, available_gpus=self.available_gpus
                )

            # Set appropriate data type for faster inference
            if "load_in_8bit" not in load_kwargs and "load_in_4bit" not in load_kwargs:
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        else:
            # CPU only
            load_kwargs["device_map"] = {"": "cpu"}

        # Load the model with specified settings
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs
            )

            # Apply additional optimizations for inference
            self.model.eval()  # Set to evaluation mode
            if hasattr(self.model, "config"):
                # Enable gradient checkpointing if available
                if hasattr(self.model.config, "gradient_checkpointing"):
                    self.model.config.gradient_checkpointing = False

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def format_conversation(
        self, prompt: str, context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a conversation with optional context

        Args:
            prompt: The current user prompt
            context: List of previous messages in the conversation
                     Each dict should have 'role' and 'content' keys

        Returns:
            Formatted conversation string for the model
        """
        # If no context, just format the prompt
        if not context:
            return f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        formatted_conversation = ""

        # Add previous messages
        for message in context:
            role = message.get("role", "user").lower()
            content = message.get("content", "")

            # Ensure role is valid
            if role not in ["user", "assistant", "system"]:
                role = "user"

            formatted_conversation += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"

        # Add the current prompt
        formatted_conversation += (
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        )

        return formatted_conversation

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def batch_generate_replies(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        contexts: Optional[List[List[Dict[str, str]]]] = None,
    ) -> List[str]:
        """
        Generate responses for multiple prompts in a batch

        Args:
            prompts: List of input text prompts
            max_new_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            contexts: Optional list of conversation contexts for each prompt

        Returns:
            List of generated text responses
        """
        if len(prompts) == 0:
            return []

        # Ensure contexts is the same length as prompts if provided
        if contexts and len(contexts) != len(prompts):
            raise ValueError("Number of contexts must match number of prompts")

        # Format prompts with contexts
        formatted_prompts = []
        for i, prompt in enumerate(prompts):
            context = contexts[i] if contexts else None
            formatted_prompts.append(self.format_conversation(prompt, context))

        # Tokenize all prompts
        tokenized_inputs = self.tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Calculate optimal batch size if needed
        batch_size = min(self.optimal_batch_size, len(prompts))
        responses = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_inputs = {
                k: v[i : i + batch_size] for k, v in tokenized_inputs.items()
            }

            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode outputs
            batch_responses = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Extract assistant responses
            for response in batch_responses:
                assistant_response = response.split("<|im_start|>assistant\n")[
                    -1
                ].split("<|im_end|>")[0]
                responses.append(assistant_response.strip())

        return responses

    def generate_reply(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response using the DeepSeek model

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            context: Optional conversation context

        Returns:
            Generated text response
        """
        # Use batch generation with a single prompt for consistency
        responses = self.batch_generate_replies(
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            contexts=[context] if context else None,
        )

        return responses[0] if responses else ""

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        context: Optional[List[Dict[str, str]]] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response from the model token by token

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            context: Optional conversation context
            callback: Optional callback function to receive each token

        Yields:
            Generated text tokens as they become available
        """
        # Format the prompt according to DeepSeek's expected format
        formatted_prompt = self.format_conversation(prompt, context)

        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Set up streaming generation with the 🤗 Transformers streamer
        try:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Start generation in a separate thread
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Run generation in a separate thread to avoid blocking
            import threading

            thread = threading.Thread(
                target=self.model.generate, kwargs=generation_kwargs
            )
            thread.start()

            # Yield tokens as they become available
            partial_text = ""
            for new_text in streamer:
                if new_text:
                    # Call the callback if provided
                    if callback:
                        callback(new_text)

                    partial_text += new_text
                    if "<|im_end|>" in partial_text:
                        # Extract only the assistant's response
                        assistant_part = partial_text.split("<|im_end|>")[0]
                        yield assistant_part
                        partial_text = ""
                    else:
                        yield new_text

            # Wait for the generation to complete
            thread.join()

        except ImportError:
            # Fallback to non-streaming implementation if streamer not available
            logger.warning(
                "TextIteratorStreamer not available, falling back to non-streaming implementation"
            )

            # Generate the full response first
            full_response = self.generate_reply(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                context=context,
            )

            # Simulate streaming by yielding chunks of the response
            words = full_response.split()

            # Stream the response word by word with a small delay
            for i in range(len(words)):
                chunk = words[i] + (" " if i < len(words) - 1 else "")

                # Call the callback if provided
                if callback:
                    callback(chunk)

                yield chunk

                # Small artificial delay
                await asyncio.sleep(0.05)

    def get_token_count(
        self, prompt: str, context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, int]:
        """
        Get token counts for a prompt and context

        Args:
            prompt: The user prompt
            context: Optional conversation context

        Returns:
            Dictionary with token count information
        """
        # Count prompt tokens
        prompt_tokens = self.count_tokens(prompt)

        # Count context tokens if provided
        context_tokens = 0
        if context:
            context_text = ""
            for message in context:
                content = message.get("content", "")
                context_text += content + " "
            context_tokens = self.count_tokens(context_text)

        return {
            "prompt_tokens": prompt_tokens,
            "context_tokens": context_tokens,
            "total_input_tokens": prompt_tokens + context_tokens,
        }

    def offload_to_cpu(self):
        """
        Offload model weights to CPU to save GPU memory
        """
        if self.device == "cuda" and hasattr(self, "model"):
            logger.info("Offloading model to CPU")
            self.model = self.model.cpu()
            self.device = "cpu"
            clear_gpu_memory()

    def reload_to_gpu(self):
        """
        Reload model to GPU after CPU offloading
        """
        if (
            self.device == "cpu"
            and torch.cuda.is_available()
            and hasattr(self, "model")
        ):
            logger.info("Reloading model to GPU")
            # Reload with previous settings
            self.device = "cuda"
            self._load_model()

    def is_on_gpu(self) -> bool:
        """Check if model is on GPU"""
        if not hasattr(self, "model"):
            return False

        if hasattr(self.model, "hf_device_map"):
            # Check if any part of the model is on GPU
            return any(
                "cuda" in str(device) for device in self.model.hf_device_map.values()
            )

        # Check if model parameters are on GPU
        return next(self.model.parameters()).is_cuda
