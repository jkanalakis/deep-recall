#!/usr/bin/env python3
"""
Database utilities for interacting with PostgreSQL.

This module provides functions for connecting to the database and
working with the embeddings and memories tables.
"""

import os
import logging
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "recall_memories_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    
    Returns:
        Connection object if successful, None otherwise
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        logger.info(f"Connected to database: {DB_NAME} at {DB_HOST}:{DB_PORT}")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def store_embedding(conn, vector_str: str) -> Optional[int]:
    """
    Store a vector embedding in the database.
    
    Args:
        conn: Database connection
        vector_str: Vector string in pgvector format
        
    Returns:
        Embedding ID if successful, None otherwise
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO embeddings (vector) VALUES (%s::vector) RETURNING id",
                (vector_str,)
            )
            result = cur.fetchone()
            conn.commit()
            
            if result and 'id' in result:
                return result['id']
            return None
            
    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        if conn:
            conn.rollback()
        return None

def store_memory(conn, user_id: str, text: str, metadata: Dict, embedding_id: int) -> Optional[int]:
    """
    Store a memory in the database.
    
    Args:
        conn: Database connection
        user_id: User ID
        text: Memory text
        metadata: Memory metadata
        embedding_id: ID of the associated embedding
        
    Returns:
        Memory ID if successful, None otherwise
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (user_id, text, metadata, embedding_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, text, Json(metadata or {}), embedding_id)
            )
            result = cur.fetchone()
            conn.commit()
            
            if result and 'id' in result:
                return result['id']
            return None
            
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        if conn:
            conn.rollback()
        return None

def search_memories(conn, vector_str: str, user_id: str, limit: int = 10, threshold: float = 0.2) -> List[Dict]:
    """
    Search for memories by vector similarity.
    
    Args:
        conn: Database connection
        vector_str: Query vector string in pgvector format
        user_id: User ID to filter memories
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        
    Returns:
        List of memory dictionaries with similarity scores
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM search_memories(%s::vector, %s, %s, %s)",
                (vector_str, user_id, limit, threshold)
            )
            return cur.fetchall()
            
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return []

def get_memory(conn, memory_id: int) -> Optional[Dict]:
    """
    Get a memory by ID.
    
    Args:
        conn: Database connection
        memory_id: Memory ID
        
    Returns:
        Memory dictionary if found, None otherwise
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, m.text, m.metadata, m.created_at, e.vector
                FROM memories m
                JOIN embeddings e ON m.embedding_id = e.id
                WHERE m.id = %s
                """,
                (memory_id,)
            )
            return cur.fetchone()
            
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        return None

def delete_memory(conn, memory_id: int) -> bool:
    """
    Delete a memory by ID.
    
    Args:
        conn: Database connection
        memory_id: Memory ID
        
    Returns:
        True if deleted, False otherwise
    """
    try:
        with conn.cursor() as cur:
            # First get the embedding ID
            cur.execute(
                "SELECT embedding_id FROM memories WHERE id = %s",
                (memory_id,)
            )
            result = cur.fetchone()
            
            if not result or 'embedding_id' not in result:
                return False
                
            embedding_id = result['embedding_id']
            
            # Delete the memory
            cur.execute(
                "DELETE FROM memories WHERE id = %s",
                (memory_id,)
            )
            
            # Delete the embedding if no other memories reference it
            cur.execute(
                "DELETE FROM embeddings WHERE id = %s AND NOT EXISTS (SELECT 1 FROM memories WHERE embedding_id = %s)",
                (embedding_id, embedding_id)
            )
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        if conn:
            conn.rollback()
        return False

def get_all_memories_for_user(conn, user_id: str) -> List[Dict]:
    """
    Get all memories for a user.
    
    Args:
        conn: Database connection
        user_id: User ID
        
    Returns:
        List of memory dictionaries
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, m.text, m.metadata, m.created_at
                FROM memories m
                WHERE m.user_id = %s
                ORDER BY m.created_at DESC
                """,
                (user_id,)
            )
            return cur.fetchall()
            
    except Exception as e:
        logger.error(f"Error getting user memories: {e}")
        return [] 