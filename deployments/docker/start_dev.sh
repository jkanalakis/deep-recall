#!/bin/bash
set -e

echo "Starting Deep Recall development environment..."

# Change to the repository root directory
cd "$(dirname "$0")/../.."

# Check if .env file exists, create if not
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cat > .env << EOF
# Database settings
DATABASE_URL=postgresql://deep_recall:deep_recall_password@postgres/deep_recall
VECTOR_DB_URL=http://qdrant:6333

# Service URLs
MEMORY_SERVICE_URL=http://memory-service:8000
INFERENCE_SERVICE_URL=http://inference-service:8000

# Model settings
MODEL_CACHE_DIR=/app/model_cache
MODEL_TYPE=deepseek_r1
MODEL_CONFIG_PATH=/app/config/model_config.yaml

# Logging
LOG_LEVEL=DEBUG
EOF
    echo ".env file created."
fi

# Check for presence of NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - enabling GPU support"
    COMPOSE_ARGS="--profile gpu"
else
    echo "No NVIDIA GPU detected - running without GPU support"
    COMPOSE_ARGS=""
fi

# Start the services
docker-compose -f deployments/docker/docker-compose.dev.yaml $COMPOSE_ARGS up --build

# Exit handler
function cleanup {
    echo "Shutting down services..."
    docker-compose -f deployments/docker/docker-compose.dev.yaml down
}

# Register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

# Keep the script running
echo "Development environment is running. Press Ctrl+C to stop."
wait 