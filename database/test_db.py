#!/usr/bin/env python3
# test_db.py

"""
Test script for the PostgreSQL database with pgvector.
This script connects to the database and performs a simple vector embedding storage and query.
"""

import os
import logging
import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "deep_recall"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

def test_connection():
    """Test database connection and vector operations."""
    conn = None
    try:
        # Connect to the database
        logger.info(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        logger.info("Connected to database successfully!")
        
        # Check if pgvector extension is installed
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                logger.info("pgvector extension is installed")
            else:
                logger.error("pgvector extension is NOT installed!")
                return
        
        # Load embedding model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embedding for a test text
        test_text = "This is a test memory for vector embeddings"
        embedding = model.encode(test_text)
        logger.info(f"Generated embedding with shape: {embedding.shape}")
        
        # Store embedding in database
        with conn.cursor() as cur:
            # Format vector for pgvector
            vector_str = "[" + ",".join(str(x) for x in embedding) + "]"
            
            # Insert embedding
            cur.execute(
                "INSERT INTO embeddings (vector) VALUES (%s::vector) RETURNING id",
                (vector_str,)
            )
            embedding_id = cur.fetchone()["id"]
            logger.info(f"Inserted embedding with ID: {embedding_id}")
            
            # Store memory
            metadata = {"source": "test", "importance": 0.8}
            cur.execute(
                "INSERT INTO memories (user_id, text, metadata, embedding_id) VALUES (%s, %s, %s, %s) RETURNING id",
                ("test_user", test_text, Json(metadata), embedding_id)
            )
            memory_id = cur.fetchone()["id"]
            logger.info(f"Inserted memory with ID: {memory_id}")
            
            # Test search by similarity
            query_text = "Test vector embedding memory"
            query_embedding = model.encode(query_text)
            query_vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            cur.execute(
                "SELECT * FROM search_memories(%s::vector, %s, 5, 0.5)",
                (query_vector_str, "test_user")
            )
            results = cur.fetchall()
            logger.info(f"Found {len(results)} similar memories")
            
            # Print results
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['text']} (similarity: {result['similarity']:.4f})")
            
            conn.commit()
            logger.info("Test completed successfully!")
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    test_connection() 