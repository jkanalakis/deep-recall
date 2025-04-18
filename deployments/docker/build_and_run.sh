#!/bin/bash
set -e

# Change to the project root directory
cd "$(dirname "$0")/../.."

# Function to display script usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  -b, --build    Build the Docker images"
  echo "  -r, --run      Run the Docker containers"
  echo "  -h, --help     Display this help message"
  echo "Example: $0 -b -r"
  exit 1
}

# Parse command line arguments
BUILD=false
RUN=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    -b|--build) BUILD=true; shift 1;;
    -r|--run) RUN=true; shift 1;;
    -h|--help) usage;;
    *) echo "Unknown parameter: $1"; usage;;
  esac
done

# If no options provided, show usage
if [ "$BUILD" = false ] && [ "$RUN" = false ]; then
  usage
fi

# Build the Docker images
if [ "$BUILD" = true ]; then
  echo "Building Docker images..."
  docker-compose -f deployments/docker/docker-compose.yaml build
  echo "Docker images built successfully."
fi

# Run the Docker containers
if [ "$RUN" = true ]; then
  echo "Running Docker containers..."
  docker-compose -f deployments/docker/docker-compose.yaml up -d
  echo "Docker containers started in detached mode."
  echo "To view logs, use: docker-compose -f deployments/docker/docker-compose.yaml logs -f"
fi

echo "Done!" 