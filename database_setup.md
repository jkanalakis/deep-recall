# Deep Recall Database Setup

This document provides instructions for setting up the PostgreSQL database for the Deep Recall framework.

## Prerequisites

- PostgreSQL 13+ installed on your system
- `pgvector` extension installed ([Installation Guide](https://github.com/pgvector/pgvector))

## Setup Instructions

### 1. Install PostgreSQL and pgvector

If you haven't already installed PostgreSQL and the pgvector extension, follow these steps:

```bash
# For macOS with Homebrew
brew install postgresql
brew services start postgresql

# Install pgvector (requires compilation)
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

For other operating systems, please follow the appropriate installation instructions for PostgreSQL and pgvector.

### 2. Create the Database and Schema

Run the initialization script to create the database, schema, and test data:

```bash
# Option 1: Run directly from the command line
psql -U postgres -f init_db.sql

# Option 2: Run from the PostgreSQL client
psql -U postgres
\i init_db.sql
```

This script will:
1. Create a new database called `deep_recall`
2. Create all necessary tables, indexes, and triggers
3. Add a test user with sample memories and interactions

### 3. Verify the Installation

Confirm that the database was created correctly:

```bash
psql -U postgres -d deep_recall
```

Once connected, you can run the following queries to check the data:

```sql
-- Check the users table
SELECT * FROM users;

-- Check the memories table
SELECT id, user_id, text, source, importance, category FROM memories;

-- Check the vector index
EXPLAIN ANALYZE SELECT * FROM memories ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector LIMIT 5;
```

## Database Schema Overview

### Core Tables

- **users**: Stores user information, preferences, and metadata
- **memories**: Stores memory content and their vector embeddings
- **sessions**: Tracks conversation sessions
- **interactions**: Records conversation turns between users and the system
- **feedback**: Stores user feedback on interactions

### Relationship Tables

- **memory_interactions**: Many-to-many relationship between memories and interactions

### Configuration Tables

- **embedding_models**: Tracks different embedding models used in the system
- **api_keys**: Stores API keys for authentication

### Statistics and Monitoring

- **statistics**: Stores usage and performance metrics
- **recent_user_activity**: View for monitoring recent user activity

## Working with Vector Embeddings

The `memories` table includes a `vector(384)` column for storing embeddings. To perform semantic searches:

```sql
-- Find memories similar to a query vector (replace with actual embedding)
SELECT id, text, embedding <-> '[...]'::vector as distance
FROM memories
WHERE user_id = '...'
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
```

## Test Data

The initialization script creates a test user with the following credentials:

- **Username**: testuser
- **Email**: test@example.com
- **Password**: testpassword (BCrypt hashed in the database)

This user has several test memories and a sample interaction.

## Development Configuration

For local development, you may want to update your connection settings in your application:

```python
# Example database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'deep_recall',
    'user': 'postgres',
    'password': 'your_postgres_password'
}
```

## Production Considerations

For production environments:

1. Use strong passwords and restrict database access
2. Enable SSL connections to the database
3. Set up database backups and monitoring
4. Consider using connection pooling for better performance
5. Add appropriate indexes based on your query patterns 