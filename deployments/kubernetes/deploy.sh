#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

# Default values
ENVIRONMENT="dev"
REGISTRY=""
TAG="latest"
BUILD_IMAGES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--environment)
      ENVIRONMENT="$2"
      shift
      shift
      ;;
    -r|--registry)
      REGISTRY="$2"
      shift
      shift
      ;;
    -t|--tag)
      TAG="$2"
      shift
      shift
      ;;
    -b|--build)
      BUILD_IMAGES=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate environment
if [[ "${ENVIRONMENT}" != "dev" && "${ENVIRONMENT}" != "prod" ]]; then
  echo "Invalid environment. Must be 'dev' or 'prod'."
  exit 1
fi

echo "🚀 Deploying Deep Recall to ${ENVIRONMENT} environment with tag ${TAG}"

# Build and push images if requested
if [[ "${BUILD_IMAGES}" == "true" ]]; then
  echo "🏗️  Building and pushing Docker images..."
  
  # Navigate to repository root
  cd "${SCRIPT_DIR}/../.."
  
  # Build Memory Service
  docker build -t "${REGISTRY}deep-recall-memory:${TAG}" -f deployments/docker/Dockerfile.memory .
  
  # Build Orchestrator Service
  docker build -t "${REGISTRY}deep-recall-orchestrator:${TAG}" -f deployments/docker/Dockerfile.orchestrator .
  
  # Build Inference Service (GPU)
  docker build -t "${REGISTRY}deep-recall-inference:${TAG}" -f deployments/docker/Dockerfile.inference .
  
  # Build Inference Service (CPU)
  docker build -t "${REGISTRY}deep-recall-inference-cpu:${TAG}" -f deployments/docker/Dockerfile.inference-cpu .
  
  # Push images
  if [[ -n "${REGISTRY}" ]]; then
    echo "📦 Pushing images to registry ${REGISTRY}..."
    docker push "${REGISTRY}deep-recall-memory:${TAG}"
    docker push "${REGISTRY}deep-recall-orchestrator:${TAG}"
    docker push "${REGISTRY}deep-recall-inference:${TAG}"
    docker push "${REGISTRY}deep-recall-inference-cpu:${TAG}"
  fi
  
  # Return to script directory
  cd "${SCRIPT_DIR}"
fi

# Deploy using kustomize
echo "📦 Deploying Kubernetes resources for ${ENVIRONMENT}..."

# Create namespace if it doesn't exist
if [[ "${ENVIRONMENT}" == "dev" ]]; then
  kubectl create namespace deep-recall-dev --dry-run=client -o yaml | kubectl apply -f -
else
  kubectl create namespace deep-recall --dry-run=client -o yaml | kubectl apply -f -
fi

# Apply kustomize overlay
kubectl apply -k "overlays/${ENVIRONMENT}" --prune -l app=deep-recall,environment=${ENVIRONMENT}

echo "✅ Deployment completed successfully!"
echo "📝 To check the status of the deployment, run:"
if [[ "${ENVIRONMENT}" == "dev" ]]; then
  echo "  kubectl get pods -n deep-recall-dev"
else
  echo "  kubectl get pods -n deep-recall"
fi 