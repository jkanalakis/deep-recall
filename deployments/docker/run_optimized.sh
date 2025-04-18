#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/../.."

# Default values
MULTI_MODEL=false
MONITORING=false
BUILD=false
DETACH=true
GPU_INFO=false

# Function to display script usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  -b, --build       Build the Docker images"
  echo "  -f, --foreground  Run containers in foreground (not detached)"
  echo "  -m, --multi-model Enable multiple model servers"
  echo "  -o, --monitoring  Enable monitoring with Prometheus and Grafana"
  echo "  -g, --gpu-info    Show GPU information before starting"
  echo "  -h, --help        Display this help message"
  exit 1
}

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -b|--build) BUILD=true; shift 1;;
    -f|--foreground) DETACH=false; shift 1;;
    -m|--multi-model) MULTI_MODEL=true; shift 1;;
    -o|--monitoring) MONITORING=true; shift 1;;
    -g|--gpu-info) GPU_INFO=true; shift 1;;
    -h|--help) usage;;
    *) echo "Unknown parameter: $1"; usage;;
  esac
done

# Display GPU information if requested
if [ "$GPU_INFO" = true ]; then
  echo "===== GPU Information ====="
  if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
  else
    echo "nvidia-smi not found. NVIDIA drivers may not be installed or accessible."
  fi
  echo "=========================="
  echo
fi

# Build the Docker images if requested
if [ "$BUILD" = true ]; then
  echo "Building Docker images..."
  docker-compose -f deployments/docker/docker-compose.optimized.yaml build
  echo "Docker images built successfully."
fi

# Prepare the docker-compose command
DOCKER_COMPOSE_CMD="docker-compose -f deployments/docker/docker-compose.optimized.yaml"
PROFILES=""

# Add profiles
if [ "$MULTI_MODEL" = true ]; then
  PROFILES="$PROFILES multi-model"
fi

if [ "$MONITORING" = true ]; then
  PROFILES="$PROFILES monitoring"
fi

# Add profiles to the command if any were set
if [ -n "$PROFILES" ]; then
  DOCKER_COMPOSE_CMD="$DOCKER_COMPOSE_CMD --profile $PROFILES"
fi

# Run the containers
echo "Starting containers..."
if [ "$DETACH" = true ]; then
  $DOCKER_COMPOSE_CMD up -d
  echo "Containers started in detached mode."
  echo "To view logs, use: docker-compose -f deployments/docker/docker-compose.optimized.yaml logs -f"
else
  $DOCKER_COMPOSE_CMD up
fi

# If detached, print some helpful information
if [ "$DETACH" = true ]; then
  echo
  echo "===== Service URLs ====="
  echo "Main Inference API: http://localhost:8000"
  
  if [ "$MULTI_MODEL" = true ]; then
    echo "Quantized Model API: http://localhost:8001"
  fi
  
  if [ "$MONITORING" = true ]; then
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin)"
  fi
  
  echo "======================="
fi

echo "Done!" 