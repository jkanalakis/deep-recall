#!/usr/bin/env python3
"""
Fix script for the search_memories PostgreSQL function.

This script fixes the return type mismatch issue in the search_memories function
where it tries to return text instead of integer for the ID column.
"""

import os
import sys
import logging
import psycopg2

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def connect_db():
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
            password=DB_PASSWORD
        )
        logger.info(f"Connected to database: {DB_NAME} at {DB_HOST}:{DB_PORT}")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def fix_search_function():
    """
    Fix the search_memories function in the database.
    
    Returns:
        True if fix was successful, False otherwise
    """
    try:
        # Connect to the database
        conn = connect_db()
        if not conn:
            logger.error("Could not connect to database")
            return False
            
        # SQL to drop and recreate the function with proper type casting
        sql = """
        -- Drop the existing function if it exists
        DROP FUNCTION IF EXISTS search_memories;

        -- Recreate the function with correct return type (ensuring ID is an INTEGER)
        CREATE OR REPLACE FUNCTION search_memories(
            query_vector vector(768),
            user_id_filter TEXT,
            limit_count INTEGER DEFAULT 10,
            similarity_threshold FLOAT DEFAULT 0.2
        )
        RETURNS TABLE (
            id INTEGER,
            user_id TEXT,
            text TEXT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE,
            similarity FLOAT
        )
        LANGUAGE SQL
        AS $$
            SELECT
                m.id::INTEGER,  -- Explicitly cast to INTEGER to ensure correct type
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
        $$;
        """
            
        # Execute the SQL
        with conn.cursor() as cur:
            cur.execute(sql)
            
        # Commit the changes
        conn.commit()
        
        logger.info("search_memories function fixed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing search_memories function: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    if fix_search_function():
        print("search_memories function fixed successfully.")
    else:
        print("Failed to fix search_memories function. See logs for details.")
        sys.exit(1) 