#!/usr/bin/env python3
# api/orchestrator.py

"""
Orchestrator for coordinating memory retrieval and inference.

This module handles context aggregation and formatting for LLM inference.
It retrieves relevant memories and integrates them into the LLM prompt.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger


class ContextAggregator:
    """
    Aggregates and formats context from memories for LLM inference.

    This class handles:
    1. Retrieving relevant memories from the memory service
    2. Formatting memories into a coherent context
    3. Creating a properly structured prompt for the LLM
    """

    def __init__(self, max_memories: int = 5, max_context_tokens: int = 2048):
        """
        Initialize the context aggregator.

        Args:
            max_memories: Maximum number of memories to include
            max_context_tokens: Maximum token budget for context
        """
        self.max_memories = max_memories
        self.max_context_tokens = max_context_tokens

    async def aggregate_context(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate memories into a context for LLM inference.

        Args:
            query: User query/prompt
            memories: List of relevant memories
            user_metadata: Optional metadata about the user

        Returns:
            Dict containing formatted context and other information
        """
        # Sort memories by relevance
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get("metadata", {}).get("relevance", 0),
            reverse=True,
        )

        # Limit number of memories
        selected_memories = sorted_memories[: self.max_memories]

        # Format memories into context
        formatted_memories = self._format_memories(selected_memories)

        # Create system instruction with user info if available
        system_instruction = self._create_system_instruction(user_metadata)

        # Construct final context
        context = {
            "system_instruction": system_instruction,
            "memories": formatted_memories,
            "user_query": query,
            "memory_count": len(selected_memories),
        }

        return context

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories into a context string."""
        if not memories:
            return ""

        formatted_texts = []
        for i, memory in enumerate(memories):
            # Extract timestamp in a readable format if available
            timestamp = memory.get("timestamp", "")
            if timestamp:
                try:
                    # Simple formatting of ISO timestamp for readability
                    # Could be enhanced with proper datetime parsing
                    timestamp = timestamp.replace("T", " ").replace("Z", "")
                    timestamp = f"({timestamp})"
                except (ValueError, AttributeError) as e:
                    # Fallback if timestamp isn't in expected format
                    timestamp = f"(Memory {i+1})"
            else:
                timestamp = f"(Memory {i+1})"

            # Format the memory text with timestamp
            memory_text = memory.get("text", "")
            formatted_text = f"Memory {timestamp}: {memory_text}"
            formatted_texts.append(formatted_text)

        # Join all formatted memories
        return "\n\n".join(formatted_texts)

    def _create_system_instruction(
        self, user_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a system instruction with user context if available."""
        instruction = "You are a helpful assistant with access to the user's memories. "

        if user_metadata:
            # Add user-specific information if available
            if "name" in user_metadata:
                instruction += f"You are talking to {user_metadata['name']}. "

            if "preferences" in user_metadata:
                prefs = user_metadata["preferences"]
                if isinstance(prefs, dict):
                    pref_str = ", ".join(f"{k}: {v}" for k, v in prefs.items())
                    instruction += f"Their preferences include: {pref_str}. "
                elif isinstance(prefs, list):
                    pref_str = ", ".join(prefs)
                    instruction += f"Their preferences include: {pref_str}. "

        instruction += (
            "Below are relevant memories that might help you respond to the user's query. "
            "Use this context to provide a personalized and accurate response."
        )

        return instruction

    def format_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format the aggregated context for LLM input.

        Takes the context dictionary and formats it as a prompt string
        suitable for sending to the LLM.

        Args:
            context: The context dictionary from aggregate_context

        Returns:
            String formatted for LLM input
        """
        # Create prompt components
        system_instruction = context["system_instruction"]
        memories = context["memories"]
        user_query = context["user_query"]

        # Construct full prompt
        prompt_parts = [
            f"{system_instruction}\n\n",
        ]

        # Add memories if available
        if memories:
            prompt_parts.append(f"RELEVANT MEMORIES:\n{memories}\n\n")

        # Add user query
        prompt_parts.append(f"USER QUERY: {user_query}\n\n")
        prompt_parts.append("YOUR RESPONSE:")

        # Join all parts
        return "".join(prompt_parts)


class RequestRouter:
    """
    Routes requests to appropriate services and aggregates responses.

    This class is responsible for:
    1. Determining which services to call based on request type
    2. Calling services in the appropriate order
    3. Aggregating responses from multiple services
    """

    async def route_inference_request(
        self,
        request_data: Dict[str, Any],
        memory_service_client,
        inference_service_client,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Route an inference request through memory retrieval and LLM inference.

        Args:
            request_data: Request parameters
            memory_service_client: Client for memory service
            inference_service_client: Client for inference service
            user_id: ID of the authenticated user

        Returns:
            Response from the inference service with added context
        """
        prompt = request_data.get("prompt", "")
        use_memory = request_data.get("use_memory", True)

        # Step 1: Get memories if needed
        memories = []
        if use_memory:
            try:
                memories = await memory_service_client.get_memories(
                    query=prompt, k=request_data.get("memory_k", 5), user_id=user_id
                )
                logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")
                # Continue without memories

        # Step 2: Aggregate context
        aggregator = ContextAggregator()
        context = await aggregator.aggregate_context(
            query=prompt, memories=memories, user_metadata={"user_id": user_id}
        )

        # Step 3: Format for LLM
        formatted_prompt = aggregator.format_for_llm(context)

        # Step 4: Call inference service
        try:
            response = await inference_service_client.generate_response(
                prompt=formatted_prompt,
                max_tokens=request_data.get("max_tokens", 1024),
                temperature=request_data.get("temperature", 0.7),
                model=request_data.get("model", "default"),
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {"error": str(e)}

        # Step 5: Enhance response with memory info
        response["memories_used"] = memories
        response["context_metadata"] = {
            "memory_count": context["memory_count"],
            "system_instruction_used": True,
        }

        return response
