# Deep Recall Docker Development Environment

This directory contains Docker configurations for local development of the Deep Recall system.

## Quick Start

The easiest way to start the development environment is to use the provided script:

```bash
./start_dev.sh
```

This script:
1. Creates a default `.env` file if one doesn't exist
2. Automatically detects if you have an NVIDIA GPU
3. Starts the appropriate services based on your hardware

## Manual Setup

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support)

### Available Docker Configurations

- `Dockerfile.memory`: Container for the Memory Service
- `Dockerfile.inference`: GPU-enabled container for the Inference Service
- `Dockerfile.inference-cpu`: CPU-only container for the Inference Service
- `Dockerfile.inference-optimized`: Optimized container for production inference
- `Dockerfile.orchestrator`: Container for the Orchestrator/API Gateway

### Docker Compose Files

- `docker-compose.dev.yaml`: Development environment with mounted source code for hot reloading
- `docker-compose.yaml`: Basic deployment
- `docker-compose.optimized.yaml`: Production-optimized deployment

### Running with Docker Compose

For CPU-only development:

```bash
docker-compose -f docker-compose.dev.yaml --profile cpu up
```

For GPU-accelerated development:

```bash
docker-compose -f docker-compose.dev.yaml --profile gpu up
```

## Service Access

Once running, the services are available at:

- Memory Service: http://localhost:8000
- Inference Service: http://localhost:8080
- Orchestrator/API Gateway: http://localhost:8001
- PostgreSQL: localhost:5432
- Qdrant Vector DB: http://localhost:6333

## Development Workflow

1. The source code is mounted into the containers, so any changes you make will be reflected immediately due to the `--reload` flag in the development Dockerfiles.

2. Database migrations and schema changes need to be applied manually. See `../../database_setup.md` for more information.

3. For GPU usage, the containers are configured to use NVIDIA GPUs automatically when available.

## Troubleshooting

- If you encounter issues with GPU support, ensure the NVIDIA Container Toolkit is properly installed and configured.
- For database connection issues, check that PostgreSQL is healthy with `docker-compose exec postgres pg_isready`.
- For vector database issues, check Qdrant's status with `curl http://localhost:6333/health`. 