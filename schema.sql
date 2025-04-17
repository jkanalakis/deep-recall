-- Deep Recall PostgreSQL Schema
-- This file contains the database schema for the Deep Recall framework

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUID generation
CREATE EXTENSION IF NOT EXISTS "pgvector";   -- For vector operations

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    preferences JSONB DEFAULT '{}',
    background TEXT,
    goals JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255),
    topic VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    previous_interactions INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_is_active ON sessions(is_active);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);

-- Memories table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding VECTOR(384),  -- Default dimension for all-MiniLM-L6-v2, adjust as needed
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    source VARCHAR(100) DEFAULT 'user',
    importance FLOAT DEFAULT 0.5,
    category VARCHAR(100),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_memories_user_id ON memories(user_id);
CREATE INDEX idx_memories_timestamp ON memories(timestamp);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_session_id ON memories(session_id);
CREATE INDEX idx_memories_source ON memories(source);
-- Create a vector index for efficient similarity search
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Interactions table (for storing conversation turns)
CREATE TABLE interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    memory_count INTEGER NOT NULL DEFAULT 0,
    was_summarized BOOLEAN NOT NULL DEFAULT FALSE,
    model_used VARCHAR(100),
    total_time_ms INTEGER,
    memory_retrieval_time_ms INTEGER,
    context_aggregation_time_ms INTEGER,
    inference_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_interactions_user_id ON interactions(user_id);
CREATE INDEX idx_interactions_session_id ON interactions(session_id);
CREATE INDEX idx_interactions_timestamp ON interactions(timestamp);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feedback_interaction_id ON feedback(interaction_id);
CREATE INDEX idx_feedback_user_id ON feedback(user_id);

-- Memory - Interaction relationship table (which memories were used in which interactions)
CREATE TABLE memory_interactions (
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    relevance_score FLOAT,
    PRIMARY KEY (memory_id, interaction_id)
);

CREATE INDEX idx_memory_interactions_memory_id ON memory_interactions(memory_id);
CREATE INDEX idx_memory_interactions_interaction_id ON memory_interactions(interaction_id);

-- User API keys for authentication
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    permissions JSONB DEFAULT '{"read": true, "write": true}'
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);

-- Embedding models table to track different embedding models
CREATE TABLE embedding_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    dimensions INTEGER NOT NULL,
    provider VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Add embedding_model_id to memories
ALTER TABLE memories ADD COLUMN embedding_model_id UUID REFERENCES embedding_models(id);
CREATE INDEX idx_memories_embedding_model_id ON memories(embedding_model_id);

-- Statistics table for tracking usage and performance metrics
CREATE TABLE statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    category VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_statistics_timestamp ON statistics(timestamp);
CREATE INDEX idx_statistics_user_id ON statistics(user_id);
CREATE INDEX idx_statistics_category ON statistics(category);
CREATE INDEX idx_statistics_name ON statistics(name);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update updated_at columns
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for recent user activity
CREATE VIEW recent_user_activity AS
SELECT 
    u.id AS user_id,
    u.username,
    COUNT(DISTINCT s.id) AS session_count,
    COUNT(DISTINCT i.id) AS interaction_count,
    COUNT(DISTINCT m.id) AS memory_count,
    MAX(i.timestamp) AS last_interaction_time
FROM 
    users u
LEFT JOIN 
    sessions s ON u.id = s.user_id
LEFT JOIN 
    interactions i ON u.id = i.user_id
LEFT JOIN 
    memories m ON u.id = m.user_id
WHERE
    u.is_active = TRUE
GROUP BY 
    u.id, u.username
ORDER BY 
    last_interaction_time DESC NULLS LAST;

-- Insert default embedding model
INSERT INTO embedding_models (
    name, dimensions, provider, version, is_active
) VALUES (
    'all-MiniLM-L6-v2', 384, 'sentence-transformers', '1.0', TRUE
); 