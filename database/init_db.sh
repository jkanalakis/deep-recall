#!/bin/bash

# Database initialization script for Deep Recall
set -e

# Print with colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deep Recall Database Initialization${NC}"
echo "==============================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running or you don't have permissions.${NC}"
  exit 1
fi

# Check for psycopg2 dependency
if ! python -c "import psycopg2" &> /dev/null; then
  echo -e "${YELLOW}Installing psycopg2-binary...${NC}"
  pip install psycopg2-binary
fi

# Start the database container
echo -e "${GREEN}Starting the database container...${NC}"
docker-compose -f deployments/docker/docker-compose.db.yml up -d

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 5
  
# Check if it's running
if ! docker ps | grep -q "deep-recall-postgres"; then
  echo -e "${RED}Error: Could not start the database container.${NC}"
  exit 1
fi

echo -e "${GREEN}Database container is running.${NC}"

# Initialize the pgvector tables and functions
echo "Initializing pgvector tables and functions..."
docker exec -i deep-recall-postgres psql -U postgres -d recall_memories_db < database/init_embeddings_tables.sql

echo -e "${GREEN}Database initialization completed successfully!${NC}"
echo "You can now run the application with: docker-compose up" 