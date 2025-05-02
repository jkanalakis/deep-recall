-- Database Initialization Script for Deep Recall
-- This script creates the database and a test user for local development

-- Create database
CREATE DATABASE deep_recall;

-- Connect to the database
\c deep_recall

-- Run the schema.sql script
\i database/schema.sql

-- Create a test user (password: testpassword)
INSERT INTO users (
    username, 
    email, 
    password_hash, 
    preferences, 
    background, 
    goals
) VALUES (
    'testuser',
    'test@example.com',
    '$2b$12$K8HbT1XPAPh7yYyoT0i8meIpaeJJxGBKRH6HQn6I2kUCQ7YNDZdZm',  -- Hash for 'testpassword'
    '{"theme": "dark", "notification_preferences": {"email": true, "push": false}}',
    'This is a test user for development purposes.',
    '["Test the Deep Recall system", "Evaluate memory retrieval performance"]'
);

-- Create a test API key for the user
INSERT INTO api_keys (
    user_id,
    key_hash,
    name
) VALUES (
    (SELECT id FROM users WHERE username = 'testuser'),
    'sha256$abcdef1234567890',  -- Placeholder hash, real implementation would use proper hashing
    'Test API Key'
);

-- Create a test session
INSERT INTO sessions (
    user_id,
    name,
    topic
) VALUES (
    (SELECT id FROM users WHERE username = 'testuser'),
    'Test Session',
    'General Testing'
);

-- Insert some test memories
INSERT INTO memories (
    user_id,
    text,
    source,
    importance,
    category,
    session_id,
    embedding_model_id,
    metadata
) VALUES
(
    (SELECT id FROM users WHERE username = 'testuser'),
    'My favorite color is blue.',
    'user',
    0.7,
    'preferences',
    (SELECT id FROM sessions WHERE name = 'Test Session' LIMIT 1),
    (SELECT id FROM embedding_models WHERE name = 'all-MiniLM-L6-v2'),
    '{"tags": ["personal", "preferences", "color"]}'
),
(
    (SELECT id FROM users WHERE username = 'testuser'),
    'I enjoy hiking in the mountains during summer.',
    'user',
    0.8,
    'hobbies',
    (SELECT id FROM sessions WHERE name = 'Test Session' LIMIT 1),
    (SELECT id FROM embedding_models WHERE name = 'all-MiniLM-L6-v2'),
    '{"tags": ["personal", "hobbies", "outdoors", "summer"]}'
),
(
    (SELECT id FROM users WHERE username = 'testuser'),
    'Need to remember to buy groceries this weekend.',
    'user',
    0.5,
    'tasks',
    (SELECT id FROM sessions WHERE name = 'Test Session' LIMIT 1),
    (SELECT id FROM embedding_models WHERE name = 'all-MiniLM-L6-v2'),
    '{"tags": ["personal", "tasks", "shopping"]}'
);

-- Insert a test interaction
INSERT INTO interactions (
    session_id,
    user_id,
    prompt,
    response,
    memory_count,
    model_used
) VALUES (
    (SELECT id FROM sessions WHERE name = 'Test Session' LIMIT 1),
    (SELECT id FROM users WHERE username = 'testuser'),
    'What are my favorite hobbies?',
    'Based on your memories, you enjoy hiking in the mountains, particularly during the summer.',
    1,
    'DeepSeek-R1'
);

-- Associate the hobby memory with the interaction
INSERT INTO memory_interactions (
    memory_id,
    interaction_id,
    relevance_score
) VALUES (
    (SELECT id FROM memories WHERE text LIKE '%hiking%' LIMIT 1),
    (SELECT id FROM interactions WHERE prompt = 'What are my favorite hobbies?' LIMIT 1),
    0.92
);

-- Add some test statistics
INSERT INTO statistics (
    user_id,
    category,
    name,
    value,
    metadata
) VALUES
(
    (SELECT id FROM users WHERE username = 'testuser'),
    'performance',
    'avg_response_time_ms',
    243.5,
    '{"sample_size": 1}'
),
(
    (SELECT id FROM users WHERE username = 'testuser'),
    'usage',
    'memories_retrieved',
    3,
    '{"period": "day"}'
);

-- Print success message
SELECT 'Database initialized successfully with test data.' AS message; 