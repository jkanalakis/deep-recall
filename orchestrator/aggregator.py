#!/usr/bin/env python3
# orchestrator/aggregator.py

"""
Context Aggregator for the Deep Recall Framework.

This module implements the context aggregation logic for preparing
comprehensive, contextually enriched prompts for LLM inference by:
1. Processing and prioritizing relevant memory retrievals
2. Formatting memories with appropriate context
3. Summarizing context when needed to respect token limits
4. Creating structured prompts optimized for LLM input
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from loguru import logger
from transformers import AutoTokenizer
import asyncio
from datetime import datetime


class ContextAggregator:
    """
    Aggregates and formats context from retrieved memories for LLM inference.

    This class is responsible for:
    1. Prioritizing memories based on relevance, recency, and importance
    2. Formatting memory data into coherent context
    3. Summarizing context when token limits are exceeded
    4. Creating properly structured prompts for the LLM
    5. Adding metadata to enhance personalization
    """

    def __init__(
        self,
        max_memories: int = 10,
        max_context_tokens: int = 2048,
        tokenizer_name: str = "gpt2",
        summarize_threshold: float = 0.8,
    ):
        """
        Initialize the context aggregator.

        Args:
            max_memories: Maximum number of memories to include
            max_context_tokens: Maximum token budget for context
            tokenizer_name: Name of the tokenizer to use for token counting
            summarize_threshold: Threshold % of token limit to trigger summarization
        """
        self.max_memories = max_memories
        self.max_context_tokens = max_context_tokens
        self.summarize_threshold = summarize_threshold

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Initialized tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None

    async def aggregate_context(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        user_metadata: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
        inference_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate memories into a comprehensive context for LLM inference.

        Args:
            query: User query/prompt
            memories: List of relevant memories from vector search
            user_metadata: Optional metadata about the user
            session_context: Optional context from the current session
            inference_params: Optional parameters for inference

        Returns:
            Dict containing formatted context and metadata
        """
        # Sort and filter memories by multiple criteria
        selected_memories = self._prioritize_memories(
            memories, user_metadata=user_metadata, session_context=session_context
        )

        # Format memories into context
        formatted_memories = self._format_memories(selected_memories)

        # Check token count and summarize if needed
        formatted_memories, was_summarized = await self._check_and_summarize(
            formatted_memories, query
        )

        # Create system instruction with user info
        system_instruction = self._create_system_instruction(
            user_metadata=user_metadata, session_context=session_context
        )

        # Construct final context dictionary
        context = {
            "system_instruction": system_instruction,
            "memories": formatted_memories,
            "user_query": query,
            "memory_count": len(selected_memories),
            "was_summarized": was_summarized,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "memory_sources": [
                    m.get("source", "unknown") for m in selected_memories
                ],
                "memory_timestamps": [
                    m.get("timestamp", "") for m in selected_memories
                ],
            },
        }

        # Add inference parameters if provided
        if inference_params:
            context["inference_params"] = inference_params

        return context

    def _prioritize_memories(
        self,
        memories: List[Dict[str, Any]],
        user_metadata: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prioritize memories based on multiple criteria.

        Combines relevance scores with recency, importance tags, and other factors
        to select the most appropriate memories for the context.

        Args:
            memories: List of memories from vector search
            user_metadata: Optional user information
            session_context: Optional session context

        Returns:
            Filtered and sorted list of memories
        """
        if not memories:
            logger.info("No memories provided to prioritize")
            return []

        # Calculate a composite score for each memory
        scored_memories = []
        for memory in memories:
            # Start with base relevance score
            base_score = memory.get("metadata", {}).get("relevance", 0.5)

            # Apply recency boost if timestamp available
            timestamp = memory.get("timestamp", "")
            recency_boost = 0
            if timestamp:
                try:
                    # Parse timestamp and calculate recency factor
                    if isinstance(timestamp, str):
                        memory_time = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        now = datetime.utcnow()
                        # Calculate days difference and apply decay factor
                        days_old = (now - memory_time).days
                        recency_boost = max(0, 0.2 * (1 - min(days_old / 30, 1)))
                except Exception as e:
                    logger.debug(f"Failed to parse timestamp for recency boost: {e}")

            # Apply importance boost if tagged as important
            importance_boost = 0
            if memory.get("metadata", {}).get("importance", 0) > 0.7:
                importance_boost = 0.15

            # Apply session relevance boost if memory is from current session
            session_boost = 0
            if session_context and session_context.get("session_id"):
                if memory.get("metadata", {}).get("session_id") == session_context.get(
                    "session_id"
                ):
                    session_boost = 0.1

            # Calculate final composite score
            final_score = base_score + recency_boost + importance_boost + session_boost

            scored_memories.append((memory, final_score))

        # Sort by final score and select top memories
        sorted_memories = [
            m[0] for m in sorted(scored_memories, key=lambda x: x[1], reverse=True)
        ]
        selected_memories = sorted_memories[: self.max_memories]

        logger.info(
            f"Selected {len(selected_memories)} memories from {len(memories)} candidates"
        )
        return selected_memories

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories into a coherent context string.

        Creates a formatted string representation of memories with
        appropriate metadata, timestamps, and categorization.

        Args:
            memories: List of selected memories

        Returns:
            Formatted string of memories
        """
        if not memories:
            return ""

        # Group memories by category if available
        categorized_memories = {}
        uncategorized_memories = []

        for memory in memories:
            category = memory.get("metadata", {}).get("category", None)
            if category:
                if category not in categorized_memories:
                    categorized_memories[category] = []
                categorized_memories[category].append(memory)
            else:
                uncategorized_memories.append(memory)

        # Format categorized memories
        formatted_sections = []

        # Add categorized memories first
        for category, category_memories in categorized_memories.items():
            category_texts = []
            for memory in category_memories:
                memory_text = self._format_single_memory(memory)
                category_texts.append(memory_text)

            if category_texts:
                formatted_category = f"--- {category.upper()} ---\n" + "\n".join(
                    category_texts
                )
                formatted_sections.append(formatted_category)

        # Add uncategorized memories
        if uncategorized_memories:
            uncategorized_texts = []
            for memory in uncategorized_memories:
                memory_text = self._format_single_memory(memory)
                uncategorized_texts.append(memory_text)

            # If we have both categorized and uncategorized, add a header
            if categorized_memories:
                formatted_uncategorized = "--- OTHER MEMORIES ---\n" + "\n".join(
                    uncategorized_texts
                )
            else:
                formatted_uncategorized = "\n".join(uncategorized_texts)

            formatted_sections.append(formatted_uncategorized)

        # Join all sections
        return "\n\n".join(formatted_sections)

    def _format_single_memory(self, memory: Dict[str, Any]) -> str:
        """Format a single memory with timestamp and metadata."""
        # Extract timestamp in a readable format if available
        timestamp_str = ""
        timestamp = memory.get("timestamp", "")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    # Parse and format the timestamp
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
                    timestamp_str = f"({timestamp_str})"
            except Exception:
                # Fallback if timestamp parsing fails
                timestamp_str = f"(Timestamp: {timestamp})"

        # Format source if available
        source_str = ""
        source = memory.get("source", "")
        if source:
            source_str = f"[Source: {source}]"

        # Format the memory text
        memory_text = memory.get("text", "")
        formatted_parts = []

        if timestamp_str:
            formatted_parts.append(timestamp_str)
        if source_str:
            formatted_parts.append(source_str)

        # Join metadata parts
        metadata_str = " ".join(formatted_parts)
        if metadata_str:
            metadata_str = f"{metadata_str}: "

        return f"{metadata_str}{memory_text}"

    async def _check_and_summarize(
        self, formatted_memories: str, query: str
    ) -> Tuple[str, bool]:
        """
        Check token count and summarize if needed.

        Args:
            formatted_memories: Formatted memory text
            query: User query

        Returns:
            Tuple of (possibly summarized memories, was_summarized flag)
        """
        # Skip if no tokenizer or no memories
        if not self.tokenizer or not formatted_memories:
            return formatted_memories, False

        # Count tokens
        memory_tokens = len(self.tokenizer.encode(formatted_memories))
        query_tokens = len(self.tokenizer.encode(query))

        # Add buffer for system instruction and formatting
        estimated_total = memory_tokens + query_tokens + 200

        # Check if we need to summarize
        threshold_tokens = int(self.max_context_tokens * self.summarize_threshold)

        if estimated_total <= threshold_tokens:
            logger.debug(f"Context size OK: {estimated_total} tokens")
            return formatted_memories, False

        logger.info(f"Context too large ({estimated_total} tokens), summarizing...")

        # For now, implement a simple truncation-based summarization
        # In a real implementation, you would call a dedicated summarization model or service

        # Simple truncation as fallback
        if self.tokenizer:
            tokens = self.tokenizer.encode(formatted_memories)
            # Keep only up to threshold tokens
            truncated_tokens = tokens[: threshold_tokens - 200]  # Leave room for suffix
            truncated_memories = self.tokenizer.decode(truncated_tokens)

            # Add a note about truncation
            truncated_memories += "\n\n[Note: Some older memories were truncated due to length constraints]"

            return truncated_memories, True

        # If we couldn't tokenize, do a crude character-based truncation
        char_ratio = threshold_tokens / estimated_total
        char_limit = int(
            len(formatted_memories) * char_ratio * 0.8
        )  # 0.8 for safety margin
        truncated_memories = (
            formatted_memories[:char_limit]
            + "...\n\n[Note: Some memories were truncated]"
        )

        return truncated_memories, True

    def _create_system_instruction(
        self,
        user_metadata: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a system instruction with user context if available.

        Args:
            user_metadata: Optional user information
            session_context: Optional session context

        Returns:
            Formatted system instruction
        """
        instruction = "You are a helpful assistant with access to the user's memories. "

        # Add user-specific information if available
        if user_metadata:
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

            # Add any additional user metadata that might be helpful
            if "background" in user_metadata:
                instruction += f"Background context: {user_metadata['background']}. "

            if "goals" in user_metadata:
                if isinstance(user_metadata["goals"], list):
                    goals = ", ".join(user_metadata["goals"])
                else:
                    goals = user_metadata["goals"]
                instruction += f"The user's goals include: {goals}. "

        # Add session context if available
        if session_context:
            if "topic" in session_context:
                instruction += (
                    f"The current conversation is about: {session_context['topic']}. "
                )

            if "previous_interactions" in session_context:
                count = session_context["previous_interactions"]
                instruction += (
                    f"This is interaction #{count+1} in the current session. "
                )

        # Add instruction for using the memories
        instruction += (
            "Below are relevant memories that might help you respond to the user's query. "
            "Use these memories to provide a personalized and contextually appropriate response. "
            "The memories are arranged from most to least relevant. "
            "Incorporate relevant memories naturally without explicitly mentioning that you're using memories "
            "unless directly asked about past interactions."
        )

        return instruction

    def format_for_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the aggregated context for LLM input.

        Takes the context dictionary and formats it as a structured input
        suitable for sending to the LLM, either as a prompt string or
        a structured chat format depending on the model requirements.

        Args:
            context: The context dictionary from aggregate_context

        Returns:
            Dictionary with formatted inputs for different LLM types
        """
        # Create prompt components
        system_instruction = context["system_instruction"]
        memories = context["memories"]
        user_query = context["user_query"]

        # 1. Construct text prompt format (for completion models)
        text_prompt_parts = [
            f"{system_instruction}\n\n",
        ]

        # Add memories if available
        if memories:
            text_prompt_parts.append(f"RELEVANT MEMORIES:\n{memories}\n\n")

        # Add user query
        text_prompt_parts.append(f"USER QUERY: {user_query}\n\n")
        text_prompt_parts.append("YOUR RESPONSE:")

        text_prompt = "".join(text_prompt_parts)

        # 2. Construct chat format (for chat models)
        chat_messages = [{"role": "system", "content": system_instruction}]

        # Add memories as system message if available
        if memories:
            chat_messages.append(
                {"role": "system", "content": f"RELEVANT MEMORIES:\n{memories}"}
            )

        # Add user query
        chat_messages.append({"role": "user", "content": user_query})

        # Return both formats
        return {
            "text_prompt": text_prompt,
            "chat_messages": chat_messages,
            "token_estimate": (
                self._estimate_tokens(text_prompt) if self.tokenizer else None
            ),
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text using the loaded tokenizer."""
        if not self.tokenizer:
            return None
        return len(self.tokenizer.encode(text))
