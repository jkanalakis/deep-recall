# models/deepseek_r1_integration.py

# Pseudocode demonstrating how you might wrap an LLM

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import Dict, Any, Optional, List, AsyncIterator, Callable
import asyncio

class DeepSeekR1Model:
    def __init__(self, model_path: str):
        """
        Initialize the DeepSeek R1 model
        
        Args:
            model_path: Path to model weights or HF model ID
        """
        self.model_path = model_path
        self.model_id = "deepseek-ai/deepseek-coder-7b-instruct"  # Default model
        
        # Check if custom model path specified in env
        custom_model = os.environ.get("DEEPSEEK_MODEL_ID")
        if custom_model:
            self.model_id = custom_model
            
        # Configure device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Load model with appropriate settings for either GPU or CPU
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
                trust_remote_code=True,
                device_map="auto"  # Let transformers handle GPU assignment
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                device_map="auto"
            )
        
        print(f"DeepSeek R1 model loaded on {self.device}")

    def format_conversation(self, prompt: str, context: Optional[List[Dict[str, str]]] = None) -> str:
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
        formatted_conversation += f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        
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

    def generate_reply(self, 
                      prompt: str, 
                      max_new_tokens: int = 1024, 
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      context: Optional[List[Dict[str, str]]] = None) -> str:
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
        # Format the prompt according to DeepSeek's expected format
        formatted_prompt = self.format_conversation(prompt, context)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response, removing the prompt
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        
        return assistant_response.strip()
    
    async def generate_stream(self, 
                             prompt: str, 
                             max_new_tokens: int = 1024, 
                             temperature: float = 0.7,
                             top_p: float = 0.9,
                             context: Optional[List[Dict[str, str]]] = None,
                             callback: Optional[Callable[[str], None]] = None) -> AsyncIterator[str]:
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
        # This is a simulated streaming implementation. In a production system,
        # we would use the model's true streaming capabilities
                
        # Format the prompt according to DeepSeek's expected format
        formatted_prompt = self.format_conversation(prompt, context)
        
        # Generate the full response first
        full_response = self.generate_reply(prompt, max_new_tokens, temperature, top_p, context)
        
        # Now simulate streaming by yielding chunks of the response
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
            
    def get_token_count(self, prompt: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, int]:
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
            "total_input_tokens": prompt_tokens + context_tokens
        }
