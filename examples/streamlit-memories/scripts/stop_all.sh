#!/bin/bash
# Script to stop all services for the Deep Recall Memories demo

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "Stopping Deep Recall Memories demo..."
cd "$PROJECT_ROOT"

# Step 1: Stop the Streamlit app
if [ -f "$PROJECT_ROOT/streamlit.pid" ]; then
  STREAMLIT_PID=$(cat "$PROJECT_ROOT/streamlit.pid")
  if ps -p $STREAMLIT_PID > /dev/null; then
    echo "Stopping Streamlit app (PID: $STREAMLIT_PID)..."
    kill $STREAMLIT_PID
  else
    echo "Streamlit app is not running."
  fi
  rm "$PROJECT_ROOT/streamlit.pid"
else
  echo "No Streamlit PID file found."
fi

# Step 2: Stop the API server and database
echo "Stopping API server and database..."
docker-compose down

echo ""
echo "All services have been stopped."
echo "" 