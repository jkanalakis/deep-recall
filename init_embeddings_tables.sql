-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create embeddings table to store vector embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    vector vector(384) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create memories table to store text and metadata
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding_id INTEGER NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_embedding_id ON memories(embedding_id);

-- Create a function to search by vector similarity
CREATE OR REPLACE FUNCTION search_memories(
    query_vector vector(384),
    user_id_filter TEXT,
    limit_count INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
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
$$; 