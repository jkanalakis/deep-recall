#!/bin/bash
# Script to start all services for the Deep Recall Memories demo

set -e

# Check for API key
if [ -z "$1" ]; then
  echo "Please provide your OpenAI API key as the first argument"
  echo "Usage: ./run_all.sh YOUR_OPENAI_API_KEY"
  exit 1
fi

OPENAI_API_KEY=$1
API_PORT=8404
STREAMLIT_PORT=8501
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "Starting Deep Recall Memories demo..."
cd "$PROJECT_ROOT"

# Step 1: Start the PostgreSQL database
echo "Starting PostgreSQL database..."
docker-compose up -d db

# Wait for the database to be ready
echo "Waiting for database to be ready..."
sleep 5

# Step 2: Start the API server
echo "Starting API server on port $API_PORT..."
export OPENAI_API_KEY=$OPENAI_API_KEY
docker-compose up -d api

# Wait for the API to be ready
echo "Waiting for API to be ready..."
attempt=0
max_attempts=30
until $(curl --output /dev/null --silent --head --fail http://localhost:$API_PORT/health); do
  if [ ${attempt} -eq ${max_attempts} ]; then
    echo "API server failed to start after $max_attempts attempts."
    exit 1
  fi
  
  printf '.'
  attempt=$(($attempt+1))
  sleep 1
done
echo ""
echo "API server is ready!"

# Step 3: Start the Streamlit app
echo "Starting Streamlit app on port $STREAMLIT_PORT..."
# Explicitly set API_URL to the correct port
export API_URL="http://localhost:$API_PORT"
echo "Setting API_URL=$API_URL"
export OPENAI_API_KEY=$OPENAI_API_KEY

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
  echo "Streamlit is not installed. Installing now..."
  pip install streamlit
fi

# Run in the background and save the PID
streamlit run "$PROJECT_ROOT/streamlit_app.py" --server.port $STREAMLIT_PORT &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > "$PROJECT_ROOT/streamlit.pid"

echo "Streamlit app is starting..."
sleep 3

echo ""
echo "All services are now running!"
echo "- API server: http://localhost:$API_PORT"
echo "- API docs: http://localhost:$API_PORT/docs"
echo "- Streamlit UI: http://localhost:$STREAMLIT_PORT"
echo ""
echo "To stop all services: ./scripts/stop_all.sh"
echo "" 