# deep-recall
A hyper-personalized agent memory framework for open-source LLMs to store, retrieve, and seamlessly integrate past user interactions. This enables LLMs to tailor responses with relevant personal context. It's lightweight, extensible, and easy to deploy on any cloud or local environment.

A **hyper-personalized agent memory framework** for open-source Large Language Models (LLMs). RecallChain stores, retrieves, and seamlessly integrates past user interactions—allowing LLMs to tailor responses with relevant personal context. It's lightweight, extensible, and easy to deploy on any cloud or local environment.

## Key Features

- **Vector Memory**: Semantic embeddings for quick, accurate retrieval of user history.  
- **Modular Design**: Swap out storage backends (FAISS, Milvus, Qdrant, etc.) or LLMs with minimal changes.  
- **Privacy & Control**: Provide APIs to view, update, or delete stored user data.  
- **Scalable & Cloud-Native**: Containerized microservices for long-running deployments on any platform.

## Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/jkanalakis/deep-recall.git
   cd deep-recall
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the example**  
   ```bash
   python examples/user_history_example.py
   ```
   This adds sample user messages, retrieves relevant memory, and demonstrates how to integrate an open-source LLM.

## Testing

The project includes comprehensive unit tests, integration tests, and API tests.

### Running Tests Locally

We provide a convenient script to run tests:

```bash
# Run all tests
./scripts/run_tests.sh

# Run only unit tests
./scripts/run_tests.sh --unit

# Run integration tests with HTML coverage report
./scripts/run_tests.sh --integration --cov

# Get help with all test options
./scripts/run_tests.sh --help
```

### Test Structure

Tests are organized by type:
- **Unit Tests**: Tests for individual components (memory store, embeddings, vector DB)
- **Integration Tests**: Tests for the complete memory pipeline
- **API Tests**: Tests for the API endpoints

### Continuous Integration

All tests run automatically on GitHub Actions for every pull request and push to main. The CI pipeline includes:
- Unit tests across multiple Python versions
- Integration tests
- API tests
- Code linting and formatting checks
- Code coverage reporting

For more details on CI, see the [CI/CD workflow documentation](./.github/workflows/README.md).

## Docker & Kubernetes Deployment

### Local Development with Docker

For local development using Docker, use the provided script:

```bash
./deployments/docker/start_dev.sh
```

This detects your hardware capabilities and starts the appropriate containers.

### Kubernetes Deployment

Deploy to Kubernetes using either Kustomize or Helm:

```bash
# Using Kustomize for dev environment
./deployments/kubernetes/deploy.sh --environment dev

# Using Helm for prod environment
helm install deep-recall ./deployments/kubernetes/helm/deep-recall -f ./deployments/kubernetes/helm/deep-recall/values/prod.yaml
```

See the [Kubernetes deployment README](./deployments/kubernetes/README.md) for more details.

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **CI**: Automated testing and linting on pull requests
- **CD**: Automatic image building and deployment to development environment
- **Production**: Manual approval required for production deployments

For more details, see the [CI/CD workflow documentation](./.github/workflows/README.md).

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with enhancements and bug fixes. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our workflow and code style.

## License

Distributed under the [Apache 2.0 License](LICENSE). Make it yours, and help build the future of AI-driven personalized experiences!
