#!/usr/bin/env python3
# db_utils.py

"""
Database utilities for the Deep Recall framework.

This module provides helper functions for common database operations,
including connecting to the database, storing memories, and retrieving
memories based on semantic similarity.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values, Json
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default database configuration
DEFAULT_DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'database': os.environ.get('DB_NAME', 'deep_recall'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres')
}

class DeepRecallDB:
    """Database access layer for the Deep Recall framework."""
    
    def __init__(self, db_config: Dict[str, Any] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the database connection.
        
        Args:
            db_config: Database connection parameters
            embedding_model: Name of the SentenceTransformer model to use for embeddings
        """
        self.db_config = db_config or DEFAULT_DB_CONFIG
        self.conn = None
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model {embedding_model}: {e}")
    
    def connect(self) -> None:
        """Establish a connection to the database."""
        try:
            self.conn = psycopg2.connect(
                **self.db_config, 
                cursor_factory=RealDictCursor
            )
            logger.info(f"Connected to database: {self.db_config['database']}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_embedding_model_id(self) -> str:
        """Get the ID of the current embedding model."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM embedding_models WHERE name = %s",
                (self.embedding_model_name,)
            )
            result = cur.fetchone()
            
            if not result:
                # Model not found, insert it
                cur.execute(
                    """
                    INSERT INTO embedding_models (name, dimensions, provider, version, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        self.embedding_model_name,
                        self.embedding_model.get_sentence_embedding_dimension(),
                        "sentence-transformers",
                        "1.0",
                        True
                    )
                )
                result = cur.fetchone()
                self.conn.commit()
                
            return result['id']
    
    def create_user(
        self, 
        username: str, 
        email: str, 
        password_hash: str,
        preferences: Dict[str, Any] = None,
        background: str = None,
        goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new user.
        
        Args:
            username: User's username
            email: User's email
            password_hash: Hashed password
            preferences: User preferences
            background: User background information
            goals: User goals
            
        Returns:
            User record
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, preferences, background, goals)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    username,
                    email,
                    password_hash,
                    Json(preferences) if preferences else Json({}),
                    background,
                    Json(goals) if goals else Json([])
                )
            )
            user = cur.fetchone()
            self.conn.commit()
            
            logger.info(f"Created user: {username}")
            return dict(user)
    
    def create_session(
        self, 
        user_id: str, 
        name: str = None, 
        topic: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            user_id: ID of the user
            name: Session name
            topic: Session topic
            metadata: Additional metadata
            
        Returns:
            Session record
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (user_id, name, topic, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (
                    user_id,
                    name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    topic,
                    Json(metadata) if metadata else Json({})
                )
            )
            session = cur.fetchone()
            self.conn.commit()
            
            logger.info(f"Created session for user {user_id}: {name}")
            return dict(session)
    
    def store_memory(
        self, 
        user_id: str, 
        text: str, 
        source: str = "user",
        importance: float = 0.5,
        category: str = None,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            user_id: ID of the user
            text: Memory text content
            source: Source of the memory
            importance: Importance score (0-1)
            category: Category of the memory
            session_id: Optional session ID
            metadata: Additional metadata
            
        Returns:
            Memory record
        """
        if not self.conn:
            self.connect()
            
        # Generate embedding for the text
        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text).tolist()
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
        
        # Get embedding model ID
        embedding_model_id = self.get_embedding_model_id() if embedding else None
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    user_id, text, embedding, source, importance, 
                    category, session_id, embedding_model_id, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    user_id,
                    text,
                    embedding,
                    source,
                    importance,
                    category,
                    session_id,
                    embedding_model_id,
                    Json(metadata) if metadata else Json({})
                )
            )
            memory = cur.fetchone()
            self.conn.commit()
            
            logger.info(f"Stored memory for user {user_id}: {text[:50]}...")
            return dict(memory)
    
    def get_memories_by_semantic_search(
        self, 
        user_id: str, 
        query: str, 
        k: int = 10, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on semantic similarity to a query.
        
        Args:
            user_id: ID of the user
            query: Query text to match against memories
            k: Number of memories to retrieve
            filters: Additional filters to apply
            
        Returns:
            List of memory records with similarity scores
        """
        if not self.conn:
            self.connect()
            
        # Generate embedding for the query
        query_embedding = None
        if self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode(query).tolist()
            except Exception as e:
                logger.warning(f"Failed to generate query embedding: {e}")
                return []
        
        if not query_embedding:
            logger.warning("No query embedding available for semantic search")
            return []
        
        # Build the filter conditions
        conditions = ["user_id = %s"]
        params = [user_id]
        
        if filters:
            if 'category' in filters:
                conditions.append("category = %s")
                params.append(filters['category'])
                
            if 'source' in filters:
                conditions.append("source = %s")
                params.append(filters['source'])
                
            if 'session_id' in filters:
                conditions.append("session_id = %s")
                params.append(filters['session_id'])
        
        # Construct the WHERE clause
        where_clause = " AND ".join(conditions)
        
        # Add the query embedding and limit
        params.extend([query_embedding, k])
        
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT 
                    m.*,
                    embedding <-> %s::vector AS similarity
                FROM 
                    memories m
                WHERE 
                    {where_clause}
                    AND embedding IS NOT NULL
                ORDER BY 
                    similarity ASC
                LIMIT %s
                """,
                params
            )
            memories = cur.fetchall()
            
            # Add relevance score based on similarity (lower distance = higher relevance)
            for memory in memories:
                similarity = memory.get('similarity', 1.0)
                # Convert distance to relevance score (0-1)
                memory['metadata'] = dict(memory['metadata'])
                memory['metadata']['relevance'] = max(0, min(1, 1 - similarity))
            
            logger.info(f"Retrieved {len(memories)} memories for user {user_id} matching query: {query[:50]}...")
            return [dict(m) for m in memories]
    
    def store_interaction(
        self, 
        session_id: str, 
        user_id: str, 
        prompt: str, 
        response: str, 
        memory_count: int = 0,
        memory_ids: List[str] = None,
        model_used: str = None,
        was_summarized: bool = False,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store a new interaction.
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            prompt: User prompt/query
            response: Generated response
            memory_count: Number of memories used
            memory_ids: IDs of memories used in this interaction
            model_used: Name of the model used
            was_summarized: Whether memory context was summarized
            metadata: Additional metadata
            
        Returns:
            Interaction record
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            # Insert the interaction
            cur.execute(
                """
                INSERT INTO interactions (
                    session_id, user_id, prompt, response, memory_count,
                    was_summarized, model_used, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    session_id,
                    user_id,
                    prompt,
                    response,
                    memory_count,
                    was_summarized,
                    model_used,
                    Json(metadata) if metadata else Json({})
                )
            )
            interaction = cur.fetchone()
            
            # If memory IDs are provided, link them to the interaction
            if memory_ids and interaction:
                interaction_id = interaction['id']
                # Default relevance score if not provided
                relevance_score = 0.75
                
                # Prepare data for batch insert
                memory_data = [(memory_id, interaction_id, relevance_score) for memory_id in memory_ids]
                
                execute_values(
                    cur,
                    """
                    INSERT INTO memory_interactions (memory_id, interaction_id, relevance_score)
                    VALUES %s
                    """,
                    memory_data
                )
            
            self.conn.commit()
            
            logger.info(f"Stored interaction for user {user_id} in session {session_id}")
            return dict(interaction)
    
    def store_feedback(
        self, 
        interaction_id: str, 
        user_id: str, 
        rating: int, 
        feedback_text: str = None
    ) -> Dict[str, Any]:
        """
        Store user feedback for an interaction.
        
        Args:
            interaction_id: ID of the interaction
            user_id: ID of the user
            rating: Numeric rating (1-5)
            feedback_text: Optional text feedback
            
        Returns:
            Feedback record
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (interaction_id, user_id, rating, feedback_text)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (interaction_id, user_id, rating, feedback_text)
            )
            feedback = cur.fetchone()
            self.conn.commit()
            
            logger.info(f"Stored feedback for interaction {interaction_id} from user {user_id}")
            return dict(feedback)
    
    def get_user_metadata(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user metadata.
        
        Args:
            user_id: ID of the user
            
        Returns:
            User metadata
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    id, username, email, preferences, background, goals, metadata,
                    created_at, updated_at, is_active
                FROM 
                    users
                WHERE 
                    id = %s
                """,
                (user_id,)
            )
            user = cur.fetchone()
            
            if not user:
                logger.warning(f"User not found: {user_id}")
                return {}
            
            # Convert to dict and ensure JSON fields are parsed
            user_dict = dict(user)
            for field in ['preferences', 'goals', 'metadata']:
                if field in user_dict and user_dict[field]:
                    user_dict[field] = dict(user_dict[field])
            
            return user_dict
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session metadata.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session metadata
        """
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    id, user_id, name, topic, previous_interactions, 
                    created_at, updated_at, is_active, metadata
                FROM 
                    sessions
                WHERE 
                    id = %s
                """,
                (session_id,)
            )
            session = cur.fetchone()
            
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return {}
            
            # Convert to dict and ensure JSON fields are parsed
            session_dict = dict(session)
            if 'metadata' in session_dict and session_dict['metadata']:
                session_dict['metadata'] = dict(session_dict['metadata'])
            
            return session_dict


# Example usage
if __name__ == "__main__":
    # This will use the default connection settings
    with DeepRecallDB() as db:
        # Print all users for demonstration
        db.conn.cursor().execute("SELECT * FROM users")
        users = db.conn.cursor().fetchall()
        print(f"Found {len(users)} users:")
        for user in users:
            print(f"  - {user['username']} ({user['email']})")
        
        # Example: Store a new memory
        if users:
            user_id = users[0]['id']
            memory = db.store_memory(
                user_id=user_id,
                text="This is a test memory created by the db_utils.py script.",
                importance=0.6,
                category="test",
                metadata={"tags": ["test", "example", "db_utils"]}
            )
            print(f"Stored new memory with ID: {memory['id']}")
            
            # Example: Retrieve similar memories
            similar_memories = db.get_memories_by_semantic_search(
                user_id=user_id,
                query="Tell me about test memories",
                k=5
            )
            print(f"Found {len(similar_memories)} similar memories")
            for i, mem in enumerate(similar_memories):
                relevance = mem['metadata'].get('relevance', 0)
                print(f"  {i+1}. [{relevance:.2f}] {mem['text'][:50]}...") 