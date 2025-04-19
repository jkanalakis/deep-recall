#!/usr/bin/env python3
# orchestrator/routing.py

"""
Request Router for the Deep Recall Framework.

This module implements the routing logic for handling user requests,
coordinating between memory retrieval and inference services, and
managing the flow of data through the system.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from orchestrator.aggregator import ContextAggregator


class RequestRouter:
    """
    Routes requests to appropriate services and aggregates responses.

    This class is responsible for:
    1. Determining which services to call based on request type
    2. Calling services in the appropriate order
    3. Aggregating responses from multiple services
    4. Managing error handling and fallbacks
    """

    def __init__(
        self,
        memory_client=None,
        inference_client=None,
        context_aggregator=None,
        max_memories: int = 10,
        default_model: str = "default",
    ):
        """
        Initialize the request router with service clients.

        Args:
            memory_client: Client for the memory service
            inference_client: Client for the inference service
            context_aggregator: Optional custom context aggregator
            max_memories: Maximum number of memories to retrieve
            default_model: Default model to use for inference
        """
        self.memory_client = memory_client
        self.inference_client = inference_client
        self.context_aggregator = context_aggregator or ContextAggregator(
            max_memories=max_memories
        )
        self.max_memories = max_memories
        self.default_model = default_model

        logger.info("RequestRouter initialized")

    async def route_inference_request(
        self,
        request_data: Dict[str, Any],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route an inference request through memory retrieval and LLM inference.

        Args:
            request_data: Request parameters including prompt, model settings, etc.
            user_id: ID of the authenticated user
            session_id: Optional session ID for context

        Returns:
            Response from the inference service with added context metadata
        """
        # Extract request parameters
        prompt = request_data.get("prompt", "")
        use_memory = request_data.get("use_memory", True)
        memory_k = request_data.get("memory_k", self.max_memories)

        # Collect session context if available
        session_context = {}
        if session_id:
            session_context["session_id"] = session_id

            # Optionally retrieve additional session metadata from memory service
            try:
                if self.memory_client:
                    session_metadata = await self.memory_client.get_session_metadata(
                        session_id
                    )
                    if session_metadata:
                        session_context.update(session_metadata)
            except Exception as e:
                logger.warning(f"Failed to retrieve session metadata: {e}")

        # Step 1: Get memories if needed
        memories = []
        memory_retrieval_time = 0
        if use_memory and self.memory_client:
            try:
                start_time = asyncio.get_event_loop().time()

                memories = await self.memory_client.get_memories(
                    query=prompt,
                    k=memory_k,
                    user_id=user_id,
                    session_id=session_id,
                    filters=request_data.get("memory_filters", {}),
                )

                memory_retrieval_time = asyncio.get_event_loop().time() - start_time
                logger.info(
                    f"Retrieved {len(memories)} memories in {memory_retrieval_time:.3f}s for user {user_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")
                # Continue without memories

        # Step 2: Get user metadata
        user_metadata = None
        if self.memory_client:
            try:
                user_metadata = await self.memory_client.get_user_metadata(user_id)
                logger.debug(f"Retrieved user metadata for {user_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve user metadata: {e}")
                # Continue without user metadata
                user_metadata = {"user_id": user_id}
        else:
            user_metadata = {"user_id": user_id}

        # Step 3: Aggregate context
        start_time = asyncio.get_event_loop().time()

        # Prepare inference parameters
        inference_params = {
            "max_tokens": request_data.get("max_tokens", 1024),
            "temperature": request_data.get("temperature", 0.7),
            "model": request_data.get("model", self.default_model),
            "top_p": request_data.get("top_p", 1.0),
            "top_k": request_data.get("top_k", 50),
        }

        context = await self.context_aggregator.aggregate_context(
            query=prompt,
            memories=memories,
            user_metadata=user_metadata,
            session_context=session_context,
            inference_params=inference_params,
        )

        context_aggregation_time = asyncio.get_event_loop().time() - start_time
        logger.debug(
            f"Context aggregation completed in {context_aggregation_time:.3f}s"
        )

        # Step 4: Format for LLM
        formatted_context = self.context_aggregator.format_for_llm(context)

        # Step 5: Call inference service
        inference_start_time = asyncio.get_event_loop().time()

        if self.inference_client:
            # Determine the best format based on the model
            model_name = inference_params["model"]
            prompt_format = "chat" if "chat" in model_name.lower() else "text"

            try:
                if prompt_format == "chat" and "chat_messages" in formatted_context:
                    # Use chat format
                    response = await self.inference_client.generate_chat_response(
                        messages=formatted_context["chat_messages"], **inference_params
                    )
                else:
                    # Use text format
                    response = await self.inference_client.generate_response(
                        prompt=formatted_context["text_prompt"], **inference_params
                    )

                inference_time = asyncio.get_event_loop().time() - inference_start_time
                logger.info(f"Inference completed in {inference_time:.3f}s")

            except Exception as e:
                logger.error(f"Inference service error: {e}")
                # Return error response
                return {
                    "error": f"Inference failed: {str(e)}",
                    "status": "error",
                    "context_metadata": {
                        "memory_count": context.get("memory_count", 0),
                        "memory_retrieval_time": memory_retrieval_time,
                        "context_aggregation_time": context_aggregation_time,
                    },
                }
        else:
            # No inference client available
            return {"error": "Inference service not available", "status": "error"}

        # Step 6: Enhance response with memory and context info
        enhanced_response = {
            **response,
            "context_metadata": {
                "memory_count": context.get("memory_count", 0),
                "memory_was_summarized": context.get("was_summarized", False),
                "memory_retrieval_time": memory_retrieval_time,
                "context_aggregation_time": context_aggregation_time,
                "inference_time": asyncio.get_event_loop().time()
                - inference_start_time,
                "total_time": memory_retrieval_time
                + context_aggregation_time
                + (asyncio.get_event_loop().time() - inference_start_time),
            },
        }

        # Optionally include memory IDs in response for tracking/logging
        if request_data.get("include_memory_ids", False) and memories:
            enhanced_response["memory_ids"] = [m.get("id", "unknown") for m in memories]

        # Step 7: Store the interaction in memory if configured
        if request_data.get("store_interaction", True) and self.memory_client:
            try:
                # Create the memory entry
                memory_entry = {
                    "user_id": user_id,
                    "text": f"USER: {prompt}\nASSISTANT: {response.get('text', '')}",
                    "metadata": {
                        "type": "conversation",
                        "session_id": session_id,
                        "prompt": prompt,
                        "response": response.get("text", ""),
                    },
                }

                # Store asynchronously without waiting for result
                asyncio.create_task(self.memory_client.store_memory(memory_entry))
                logger.debug(f"Scheduled memory storage for interaction")
            except Exception as e:
                logger.warning(f"Failed to store interaction in memory: {e}")

        return enhanced_response

    async def store_user_feedback(
        self, feedback_data: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """
        Store user feedback about a response.

        Args:
            feedback_data: Feedback parameters including interaction ID, rating, etc.
            user_id: ID of the authenticated user

        Returns:
            Status of the feedback storage operation
        """
        if not self.memory_client:
            return {"status": "error", "message": "Memory service not available"}

        try:
            # Extract feedback parameters
            interaction_id = feedback_data.get("interaction_id")
            rating = feedback_data.get("rating")
            feedback_text = feedback_data.get("feedback_text", "")

            if not interaction_id or rating is None:
                return {
                    "status": "error",
                    "message": "Missing required feedback parameters",
                }

            # Store feedback
            result = await self.memory_client.store_feedback(
                user_id=user_id,
                interaction_id=interaction_id,
                rating=rating,
                feedback_text=feedback_text,
            )

            return {
                "status": "success",
                "message": "Feedback stored successfully",
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return {"status": "error", "message": f"Failed to store feedback: {str(e)}"}
