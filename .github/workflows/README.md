# Deep Recall CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment of the Deep Recall system.

## Available Workflows

### CI - Test (`ci-test.yaml`)

Triggered on:
- Push to `main` branch
- Pull requests to `main` branch

This workflow:
1. Sets up test infrastructure (PostgreSQL and Qdrant)
2. Runs unit tests with code coverage reporting
3. Performs code linting and style checks

### CD - Build Images (`cd-build-images.yaml`)

Triggered on:
- Push to `main` branch
- Release tags (`v*.*.*`)
- Manual trigger via GitHub Actions UI

This workflow:
1. Builds all service Docker images:
   - Memory Service
   - Inference Service (GPU version)
   - Inference Service (CPU version) 
   - Orchestrator Service
2. Pushes images to GitHub Container Registry (ghcr.io)
3. Tags images with git SHA, release tag, or custom tag

### CD - Deploy to Dev (`cd-deploy-dev.yaml`)

Triggered on:
- Completion of "CD - Build Images" workflow 
- Manual trigger via GitHub Actions UI

This workflow:
1. Deploys to the development environment using Kustomize
2. Updates image tags in the deployment
3. Verifies deployment success
4. Provides a deployment summary

### CD - Deploy to Production (`cd-deploy-prod.yaml`)

Triggered on:
- Release publication
- Manual trigger via GitHub Actions UI

This workflow:
1. Requires manual approval in the production environment
2. Deploys to production using Helm
3. Uses release tags for image versions
4. Verifies deployment success
5. Provides a deployment summary

## Required Secrets

The following secrets need to be configured in GitHub repository settings:

- `DEV_KUBECONFIG`: Kubernetes config file for the development environment
- `PROD_KUBECONFIG`: Kubernetes config file for the production environment
- `CODECOV_TOKEN`: Token for Codecov.io coverage reporting (optional)

## Usage

### Manual Deployments

#### Deploying to Development

1. Go to "Actions" in the GitHub repository
2. Select "CD - Deploy to Dev" workflow
3. Click "Run workflow"
4. Choose the image tag (defaults to "latest")
5. Click "Run workflow"

#### Deploying to Production

1. Go to "Actions" in the GitHub repository 
2. Select "CD - Deploy to Production" workflow
3. Click "Run workflow"
4. Choose the image tag (defaults to latest release)
5. Click "Run workflow"
6. Approve the deployment in the "Environments" section

### Release Process

1. Create and publish a new release with tag version (e.g., `v1.2.3`)
2. The "CD - Build Images" workflow will automatically build and tag images
3. The "CD - Deploy to Production" workflow will be triggered, requiring approval

## Environments

The workflows use the following GitHub Environments:
- `production`: Used for production deployments with required approval

Make sure to configure these environments in the repository settings. 