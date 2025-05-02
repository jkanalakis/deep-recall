#!/usr/bin/env python3
"""
Memory store for Deep Recall.

This module provides a memory store for managing text data, embeddings, and metadata.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MemoryStore:
    def __init__(
        self,
        embedding_dim: int = 384,
        metadata_path: str = "memory_metadata.json",
        # Database connection defaults
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
    ):
        """
        Initialize a memory store.

        Args:
            embedding_dim: Dimension of the embedding vectors
            metadata_path: Path to save/load the metadata JSON for backup
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.last_backup_time = None
        self.backup_interval = 3600  # Default to hourly backups (in seconds)

        # Initialize empty data structures for local cache
        self.text_data = {}
        self.metadata = {}
        
        # Set up database connection parameters
        self.db_config = {
            "host": db_host or os.environ.get("DB_HOST", "localhost"),
            "port": db_port or int(os.environ.get("DB_PORT", 5432)),
            "database": db_name or os.environ.get("DB_NAME", "recall_memories_db"),
            "user": db_user or os.environ.get("DB_USER", "postgres"),
            "password": db_password or os.environ.get("DB_PASSWORD", "postgres"),
        }
        
        # Connection object
        self.conn = None
        
        # Try to connect to database
        self._connect_db()
        
        # Initialize database tables if needed
        self._init_db_tables()
        
        # Load existing data as a fallback
        self._load_existing_data()

    def _connect_db(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                **self.db_config, cursor_factory=RealDictCursor
            )
            logger.info(f"Connected to database: {self.db_config['database']} at {self.db_config['host']}:{self.db_config['port']}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self.conn = None

    def _ensure_connection(self):
        """Ensure we have a valid database connection."""
        if self.conn is None:
            self._connect_db()
        else:
            try:
                # Test connection with a simple query
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except psycopg2.OperationalError:
                logger.warning("Database connection lost, reconnecting...")
                self.conn = None
                self._connect_db()

    def _init_db_tables(self):
        """Initialize database tables if they don't exist."""
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot initialize database tables: No connection")
            return
            
        try:
            with self.conn.cursor() as cur:
                # Initialize pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create embeddings table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        vector vector({self.embedding_dim}) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create memories table - using TEXT type for id to avoid type mismatch issues
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}',
                        embedding_id INTEGER REFERENCES embeddings(id) ON DELETE CASCADE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding_id ON memories(embedding_id)")
                
                # Create search function if it doesn't exist - ensuring types match table definition
                cur.execute("""
                    CREATE OR REPLACE FUNCTION search_memories(
                        query_vector vector(384),
                        user_id_filter TEXT,
                        limit_count INTEGER DEFAULT 10,
                        similarity_threshold FLOAT DEFAULT 0.7
                    )
                    RETURNS TABLE (
                        id TEXT,
                        user_id TEXT,
                        text TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE,
                        similarity FLOAT
                    )
                    LANGUAGE SQL
                    AS $$
                        SELECT
                            m.id,
                            m.user_id,
                            m.text,
                            m.metadata,
                            m.created_at,
                            1 - (e.vector <-> query_vector) AS similarity
                        FROM
                            memories m
                        JOIN
                            embeddings e ON m.embedding_id = e.id
                        WHERE
                            m.user_id = user_id_filter
                            AND 1 - (e.vector <-> query_vector) >= similarity_threshold
                        ORDER BY
                            similarity DESC
                        LIMIT
                            limit_count;
                    $$
                """)
                
                self.conn.commit()
                logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            if self.conn:
                self.conn.rollback()

    def _load_existing_data(self):
        """Load metadata from JSON if it exists (as a fallback)."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                    self.text_data = data.get("text_data", {})
                    # Convert string keys back to integers
                    self.text_data = {int(k): v for k, v in self.text_data.items()}
                    self.metadata = data.get("metadata", {})
                    # Convert string keys back to integers
                    self.metadata = {int(k): v for k, v in self.metadata.items()}
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading metadata: {e}")
                # Initialize empty if loading fails
                self.text_data = {}
                self.metadata = {}

    def add_memory(self, memory):
        """
        Add a memory to the store.

        Args:
            memory: Memory object to store

        Returns:
            ID of the stored memory
        """
        # First try to store in database
        stored_in_db = self._store_memory_in_db(memory)
        
        # Also maintain in-memory cache for fast access
        numeric_id = hash(memory.id) % (2**31)
        self.text_data[numeric_id] = memory.text

        # Create metadata from memory object
        metadata = memory.metadata.copy() if memory.metadata else {}
        metadata["user_id"] = memory.user_id
        metadata["created_at"] = memory.created_at
        metadata["importance"] = memory.importance.value if hasattr(memory.importance, "value") else memory.importance

        self.metadata[numeric_id] = metadata

        # Consider auto-backup to JSON
        self._maybe_backup()
        
        logger.info(f"Memory {memory.id} stored{'in database' if stored_in_db else ' (in-memory only)'}")
        return memory.id
    
    def _store_memory_in_db(self, memory) -> bool:
        """
        Store a memory in the PostgreSQL database.
        
        Args:
            memory: Memory object to store
            
        Returns:
            True if successfully stored in database, False otherwise
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot store memory in database: No connection")
            return False
            
        try:
            with self.conn.cursor() as cur:
                # Check if memory already exists
                cur.execute(
                    "SELECT id, embedding_id FROM memories WHERE id = %s",
                    (memory.id,)
                )
                existing_memory = cur.fetchone()
                
                # Handle the embedding
                embedding_id = None
                if hasattr(memory, 'embedding') and memory.embedding is not None:
                    # Format the embedding vector for pgvector
                    vector_str = f"[{','.join(str(x) for x in memory.embedding)}]"
                    
                    if existing_memory and existing_memory['embedding_id'] is not None:
                        # Update existing embedding
                        embedding_id = existing_memory['embedding_id']
                        cur.execute(
                            "UPDATE embeddings SET vector = %s::vector WHERE id = %s",
                            (vector_str, embedding_id)
                        )
                    else:
                        # Create new embedding
                        cur.execute(
                            "INSERT INTO embeddings (vector) VALUES (%s::vector) RETURNING id",
                            (vector_str,)
                        )
                        embedding_id = cur.fetchone()['id']
                elif existing_memory:
                    # Keep existing embedding_id if there is one
                    embedding_id = existing_memory['embedding_id']
                
                # Store or update the memory
                if existing_memory:
                    # Update existing memory
                    cur.execute(
                        """
                        UPDATE memories 
                        SET text = %s, 
                            metadata = %s,
                            user_id = %s,
                            embedding_id = %s
                        WHERE id = %s
                        """,
                        (
                            memory.text, 
                            Json(memory.metadata or {}),
                            memory.user_id,
                            embedding_id,
                            memory.id
                        )
                    )
                else:
                    # Insert new memory
                    cur.execute(
                        """
                        INSERT INTO memories (id, user_id, text, metadata, embedding_id)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            memory.id,
                            memory.user_id,
                            memory.text,
                            Json(memory.metadata or {}),
                            embedding_id
                        )
                    )
                
                self.conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing memory in database: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            Dict with memory data or None if not found
        """
        # First try to get from database
        db_memory = self._get_memory_from_db(memory_id)
        if db_memory:
            return db_memory
            
        # Fall back to in-memory data
        numeric_id = hash(memory_id) % (2**31)
        if numeric_id not in self.text_data:
            return None

        metadata = self.metadata.get(numeric_id, {})
        
        # Skip deleted items
        if metadata.get("deleted", False):
            return None

        return {
            "id": memory_id,
            "text": self.text_data[numeric_id],
            "metadata": metadata
        }
    
    def _get_memory_from_db(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory directly from the database.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Memory dict if found, None otherwise
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot retrieve memory from database: No connection")
            return None
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT m.id, m.user_id, m.text, m.metadata, m.created_at, e.vector
                    FROM memories m
                    LEFT JOIN embeddings e ON m.embedding_id = e.id
                    WHERE m.id = %s AND m.metadata->>'deleted' IS NULL
                    """,
                    (memory_id,)
                )
                memory = cur.fetchone()
                
                if memory:
                    # Format as a memory dict
                    result = dict(memory)
                    result["metadata"] = dict(result["metadata"])
                    result["created_at"] = str(result["created_at"])
                    
                    # Update local cache for future fast access
                    numeric_id = hash(memory_id) % (2**31)
                    self.text_data[numeric_id] = result["text"]
                    self.metadata[numeric_id] = result["metadata"]
                    
                    return result
                    
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving memory from database: {e}")
            return None

    def get_memories_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of memory dictionaries
        """
        # First try to get from database
        db_memories = self._get_user_memories_from_db(user_id)
        if db_memories:
            return db_memories
            
        # Fall back to in-memory data if database retrieval failed
        results = []
        for numeric_id, text in self.text_data.items():
            metadata = self.metadata.get(numeric_id, {})
            
            # Skip deleted items
            if metadata.get("deleted", False):
                continue
                
            # Only include memories for this user
            if metadata.get("user_id") != user_id:
                continue
                
            results.append({
                "id": str(numeric_id),  # Convert back to string
                "text": text,
                "metadata": metadata
            })
            
        return results
    
    def _get_user_memories_from_db(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve all memories for a user from the database.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of memory dictionaries or None if retrieval failed
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot retrieve user memories from database: No connection")
            return None
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT m.id, m.user_id, m.text, m.metadata, m.created_at
                    FROM memories m
                    WHERE m.user_id = %s AND m.metadata->>'deleted' IS NULL
                    """,
                    (user_id,)
                )
                memories = cur.fetchall()
                
                results = []
                for memory in memories:
                    result = dict(memory)
                    result["metadata"] = dict(result["metadata"])
                    result["created_at"] = str(result["created_at"])
                    results.append(result)
                    
                    # Update local cache
                    numeric_id = hash(result["id"]) % (2**31)
                    self.text_data[numeric_id] = result["text"]
                    self.metadata[numeric_id] = result["metadata"]
                    
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving user memories from database: {e}")
            return None

    def search_memories(self, user_id: str, query_vector: np.ndarray, limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for memories using vector similarity.
        
        Args:
            user_id: ID of the user
            query_vector: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot search memories: No connection")
            return []
            
        try:
            with self.conn.cursor() as cur:
                # Format vector for pgvector
                vector_str = f"[{','.join(str(x) for x in query_vector)}]"
                
                # Use the search_memories function
                cur.execute(
                    "SELECT * FROM search_memories(%s::vector, %s, %s, %s)",
                    (vector_str, user_id, limit, threshold)
                )
                
                results = []
                for memory in cur.fetchall():
                    result = dict(memory)
                    result["metadata"] = dict(result["metadata"])
                    result["created_at"] = str(result["created_at"])
                    results.append(result)
                    
                    # Update local cache
                    numeric_id = hash(result["id"]) % (2**31)
                    self.text_data[numeric_id] = result["text"]
                    self.metadata[numeric_id] = result["metadata"]
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            bool: True if deleted successfully
        """
        # Try to delete in database
        db_deleted = self._delete_memory_in_db(memory_id)
        
        # Also update local cache
        numeric_id = hash(memory_id) % (2**31)
        if numeric_id not in self.text_data:
            return db_deleted or False  # Return db result if available

        # Mark as deleted in metadata
        if numeric_id in self.metadata:
            self.metadata[numeric_id]["deleted"] = True
            self.metadata[numeric_id]["deletion_timestamp"] = datetime.now().isoformat()

        # Save changes to backup
        self._save_metadata()
        return True
    
    def _delete_memory_in_db(self, memory_id: str) -> bool:
        """
        Delete a memory in the database (by marking as deleted).
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            True if successfully deleted in database, False otherwise
        """
        self._ensure_connection()
        
        if self.conn is None:
            logger.warning("Cannot delete memory in database: No connection")
            return False
            
        try:
            with self.conn.cursor() as cur:
                # We don't actually delete; just mark as deleted in metadata
                cur.execute(
                    """
                    UPDATE memories
                    SET metadata = metadata || '{"deleted": true, "deletion_timestamp": "' || NOW() || '"}'::jsonb
                    WHERE id = %s
                    """,
                    (memory_id,)
                )
                
                self.conn.commit()
                affected = cur.rowcount
                return affected > 0
                
        except Exception as e:
            logger.error(f"Error deleting memory in database: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def _save_metadata(self):
        """Save metadata to disk (as backup)."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.metadata_path)), exist_ok=True)

        # Convert int keys to strings for JSON serialization
        serializable_text_data = {str(k): v for k, v in self.text_data.items()}
        serializable_metadata = {str(k): v for k, v in self.metadata.items()}

        data = {
            "text_data": serializable_text_data,
            "metadata": serializable_metadata,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(data, f)

    def _maybe_backup(self):
        """Backup data if enough time has passed since the last backup."""
        current_time = time.time()

        if (
            self.last_backup_time is None
            or (current_time - self.last_backup_time) > self.backup_interval
        ):
            self._save_metadata()
            self.last_backup_time = current_time

    def set_backup_interval(self, interval_seconds: int):
        """Set the automatic backup interval."""
        self.backup_interval = interval_seconds

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        stats = {
            "active_memories": sum(
                1 for m in self.metadata.values() if not m.get("deleted", False)
            ),
            "deleted_memories": sum(
                1 for m in self.metadata.values() if m.get("deleted", False)
            ),
            "embedding_dimension": self.embedding_dim,
            "last_backup": self.last_backup_time,
            "db_connected": self.conn is not None
        }
        
        # Add database stats if connected
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as count FROM memories WHERE metadata->>'deleted' IS NULL")
                    db_count = cur.fetchone()
                    if db_count:
                        stats["db_active_memories"] = db_count["count"]
                        
                    cur.execute("SELECT COUNT(*) as count FROM memories WHERE metadata->>'deleted' = 'true'")
                    db_deleted = cur.fetchone()
                    if db_deleted:
                        stats["db_deleted_memories"] = db_deleted["count"]
                        
                    cur.execute("SELECT COUNT(*) as count FROM embeddings")
                    embeddings_count = cur.fetchone()
                    if embeddings_count:
                        stats["db_embeddings"] = embeddings_count["count"]
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
        
        return stats
