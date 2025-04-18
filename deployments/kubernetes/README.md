# Deep Recall Kubernetes Deployment

This directory contains Kubernetes manifests and Helm charts for deploying the Deep Recall system to Kubernetes clusters.

## Deployment Methods

There are two deployment options available:

1. **Kustomize-based deployment**: Traditional Kubernetes manifests with Kustomize for environment-specific configuration
2. **Helm-based deployment**: Helm charts for more complex deployments with dependencies

## Kustomize Deployment

The Kustomize-based deployment uses a base set of manifests and overlays for different environments.

### Directory Structure

- `base/`: Base Kubernetes manifests
- `overlays/dev/`: Development environment overlay
- `overlays/prod/`: Production environment overlay

### Deployment

Use the provided deployment script to deploy to a Kubernetes cluster:

```bash
# Deploy to development environment
./deploy.sh --environment dev

# Deploy to production environment
./deploy.sh --environment prod

# Build and push Docker images before deployment
./deploy.sh --environment dev --build --registry your-registry.example.com/

# Specify a custom tag for Docker images
./deploy.sh --environment prod --tag v1.0.0
```

## Helm Deployment

The Helm-based deployment provides a more flexible and comprehensive deployment solution.

### Directory Structure

- `helm/deep-recall/`: Main Helm chart for Deep Recall
  - `templates/`: Kubernetes manifest templates
  - `values/`: Environment-specific values files
  - `charts/`: Dependent charts (automatically downloaded during deployment)

### Deployment

To deploy using Helm:

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update

# Deploy to development environment
helm install deep-recall ./helm/deep-recall -f ./helm/deep-recall/values/dev.yaml --namespace deep-recall-dev --create-namespace

# Deploy to production environment
helm install deep-recall ./helm/deep-recall -f ./helm/deep-recall/values/prod.yaml --namespace deep-recall --create-namespace
```

### Custom Values

You can override specific values with a custom values file:

```bash
helm install deep-recall ./helm/deep-recall -f ./helm/deep-recall/values/prod.yaml -f ./custom-values.yaml --namespace deep-recall --create-namespace
```

## Architecture

The deployment includes the following components:

1. **Memory Service**: Manages conversation memory and vector embeddings
2. **Inference Service**: Handles LLM inference (with GPU support in production)
3. **Orchestrator Service**: Coordinates between services and serves as API gateway
4. **PostgreSQL**: Relational database for storing conversation data
5. **Qdrant**: Vector database for storing and retrieving embeddings

## Configuration

### Resource Requirements

The resource requirements vary by environment:

- **Development**: Minimal resources for local or development clusters
- **Production**: Optimized for performance with GPU support for inference

### Scaling

Both deployment methods support horizontal pod autoscaling:

- Memory Service: Based on CPU and memory usage
- Inference Service: Based on CPU, memory, and GPU usage (in production)
- Orchestrator Service: Based on CPU and memory usage

## Customization

For customized deployments, you can:

1. Create new environment overlays with Kustomize
2. Create new values files for Helm
3. Modify resource limits and requests
4. Configure ingress for different domain names
5. Adjust autoscaling parameters 