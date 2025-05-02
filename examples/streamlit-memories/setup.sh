#!/bin/bash

# Setup script for Deep Recall Memories
set -e

# Print with colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deep Recall Memories Setup${NC}"
echo "==============================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running or you don't have permissions.${NC}"
  exit 1
fi

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Start the database container
echo -e "${GREEN}Starting the database container...${NC}"
docker-compose -f docker-compose.db.yml up -d

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 5
  
# Check if it's running
if ! docker ps | grep -q "recall-memories-postgres"; then
  echo -e "${RED}Error: Could not start the database container.${NC}"
  exit 1
fi

echo -e "${GREEN}Database container is running.${NC}"

# Initialize the database schema
echo "Initializing database schema..."
docker exec -i recall-memories-postgres psql -U postgres -d recall_memories_db < init_embeddings_tables.sql

echo -e "${GREEN}Setup completed successfully!${NC}"
echo "You can now run the Streamlit app with: streamlit run streamlit_app.py" 