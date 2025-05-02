#!/usr/bin/env python3
"""
Setup script for the Deep Recall Memories database.

This script initializes the PostgreSQL database with the required tables
and functions for the memory system.
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

def setup_database():
    """
    Set up the database with required extensions, tables, and functions.
    
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Connect to the database
        conn = connect_db()
        if not conn:
            logger.error("Could not connect to database")
            return False
            
        # Execute SQL from the init_embeddings_tables.sql file
        sql_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../init_embeddings_tables.sql'))
        with open(sql_path, 'r') as f:
            sql = f.read()
            
        # Execute the SQL
        with conn.cursor() as cur:
            cur.execute(sql)
            
        # Commit the changes
        conn.commit()
        
        logger.info("Database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    if setup_database():
        print("Database setup completed successfully.")
    else:
        print("Database setup failed. See logs for details.")
        sys.exit(1) 