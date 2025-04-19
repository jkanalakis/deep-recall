# Deep Recall

A hyper-personalized agent memory framework for open-source LLMs to store, retrieve, and seamlessly integrate past user interactions. This enables LLMs to tailor responses with relevant personal context. It's lightweight, extensible, and easy to deploy on any cloud or local environment.

## 🔍 Overview

Deep Recall is a sophisticated memory framework designed to enhance the capabilities of open-source Large Language Models (LLMs) by providing:

- **Contextual Awareness**: Enables LLMs to remember and reference past conversations with specific users
- **Personalized Responses**: Tailors responses based on user history, preferences, and past interactions
- **Scalable Architecture**: Designed for high-performance in both cloud and local deployments

## 🚀 Key Features

- **Vector Memory**: Semantic embeddings for quick, accurate retrieval of user history  
- **Modular Design**: Swap out storage backends (FAISS, Milvus, Qdrant, Chroma) or LLMs with minimal changes
- **Privacy & Control**: APIs to view, update, or delete stored user data
- **Scalable & Cloud-Native**: Containerized microservices for long-running deployments on any platform
- **Multi-modal Support**: Store and retrieve text, structured data, and metadata
- **GPU Acceleration**: Optimized for GPU inference with support for quantization
- **Comprehensive Monitoring**: Built-in metrics and logging for performance tracking

## 🏗️ System Architecture

Deep Recall consists of three primary components:

### 1. Memory Service
- **Storage**: PostgreSQL for structured data storage with pgvector extension
- **Vector Embeddings**: Generates embeddings using SentenceTransformers or Hugging Face models
- **Vector Databases**: Supports FAISS, Qdrant, Milvus, and Chroma for efficient similarity search
- **Semantic Search**: Fast top-k retrieval with configurable similarity thresholds

### 2. Inference Service
- **Model Hosting**: Support for open-source LLMs including DeepSeek R1 and LLaMA variants
- **GPU Optimization**: CUDA-enabled inference with mixed precision (FP16)
- **Model Deployment**: Containerized deployment optimized for Kubernetes

### 3. Orchestrator/API Gateway
- **Request Routing**: Efficient management of requests between services
- **Context Aggregation**: Combines retrieved memories with current context
- **API Management**: RESTful and gRPC interfaces for external integration

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 13+ with pgvector extension
- CUDA-compatible GPU (optional but recommended)

### 1. Clone the repo  
```bash
git clone https://github.com/jkanalakis/deep-recall.git
cd deep-recall
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Set up the database
```bash
# Install PostgreSQL and pgvector (see database_setup.md for details)
# Initialize the database
psql -U postgres -f init_db.sql
```

### 4. Run the example  
```bash
python examples/user_history_example.py
```

This adds sample user messages, retrieves relevant memory, and demonstrates how to integrate with an open-source LLM.

## 📊 Advanced Examples

### Semantic Search

```python
from memory.semantic_search import SemanticSearch
from memory.vector_db.faiss_store import FAISSVectorStore

# Initialize vector store and semantic search
vector_store = FAISSVectorStore(dimension=384)
semantic_search = SemanticSearch(vector_store=vector_store)

# Search for similar memories
query = "What was our discussion about machine learning?"
results = semantic_search.search(
    query=query,
    user_id="user123",
    limit=5,
    threshold=0.75
)

for result in results:
    print(f"Similarity: {result.similarity}, Memory: {result.text}")
```

### Personalized Response Generation

```python
from memory.memory_retriever import MemoryRetriever
from inference_service.models.llm import LLM

# Initialize components
memory_retriever = MemoryRetriever()
llm = LLM(model_name="deepseek-r1")

# Retrieve relevant memories
memories = memory_retriever.get_relevant_memories(
    user_id="user123",
    query="What did we discuss about my project last time?",
    limit=3
)

# Format context with memories
context = "Previous relevant conversations:\n"
for memory in memories:
    context += f"- {memory.text}\n"

# Generate personalized response
response = llm.generate(
    prompt=f"{context}\nUser: What did we discuss about my project last time?",
    max_tokens=150
)

print(response)
```

## 🗄️ Database Configuration

Deep Recall uses PostgreSQL with the pgvector extension for efficient vector storage and retrieval.

### Core Tables

- **users**: User information and preferences
- **memories**: Memory content and vector embeddings
- **sessions**: Conversation sessions
- **interactions**: Conversation turns between users and the system
- **feedback**: User feedback on interactions

See [database_setup.md](database_setup.md) for detailed setup instructions.

## 🧪 Testing

The project includes comprehensive unit tests, integration tests, and API tests.

### Running Tests Locally

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

## 🐳 Docker & Kubernetes Deployment

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

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **CI**: Automated testing and linting on pull requests
- **CD**: Automatic image building and deployment to development environment
- **Production**: Manual approval required for production deployments

## 📝 API Reference

Deep Recall provides RESTful APIs for memory management and retrieval:

### Memory Management
- `POST /api/v1/memories` - Store a new memory
- `GET /api/v1/memories?query=<text>&user_id=<id>` - Retrieve relevant memories
- `DELETE /api/v1/memories/{memory_id}` - Delete a specific memory
- `DELETE /api/v1/users/{user_id}/memories` - Delete all memories for a user

### User Management
- `POST /api/v1/users` - Create a new user
- `GET /api/v1/users/{user_id}` - Get user information
- `DELETE /api/v1/users/{user_id}` - Delete a user and all associated data

See the [API documentation](./docs/api.md) for complete details.

## 📊 Performance Considerations

### Memory Optimization
- Use chunking strategies for large text inputs
- Configure embedding dimensions based on your accuracy/performance needs
- Implement caching for frequently accessed memories

### Scaling Options
- Horizontal scaling of memory and inference services
- Database sharding for large-scale deployments
- GPU-accelerated inference for production workloads

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with enhancements and bug fixes. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our workflow and code style.

### Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

## 📌 Roadmap

1. Backup & Disaster Recovery:

* Implement automated backup schedules.

* Clearly document restoration procedures and test regularly.

2. Security Automation:

* Integrate regular automated security scans (tools like OWASP ZAP, Dependabot, or GitHub Advanced Security).

* Document vulnerability management and patching procedures.

3. Rollback and Versioning:

* Add clear version control of model artifacts and database schemas.

* Implement rollback scripts for deployments.

4. Robustness & Edge Case Tests:

* Develop more comprehensive stress-tests and edge case scenarios.

* Add explicit error handling and recovery strategies.

5. Data Migration:

* Clearly define data migration pathways for evolving schemas or new embeddings.

6. Front-End Integration:

* Provide API client examples or SDKs (JavaScript, Python client libraries).

* Simple front-end example (e.g., React, Vue) that demonstrates API integration.

## 📄 License

Distributed under the [Apache 2.0 License](LICENSE). Make it yours, and help build the future of AI-driven personalized experiences!

## 🙏 Acknowledgements

- This project builds upon research from the field of personalized AI assistants
- Special thanks to the open-source ML community for providing accessible models and tools

## 🔒 Security Scanning

Deep Recall includes a comprehensive security scanning system to ensure code quality and identify potential vulnerabilities:

### Security Scanner Features

- **Dependency Scanning**: Checks for known vulnerabilities in Python dependencies using Safety
- **Code Security Analysis**: Identifies security issues in Python code using Bandit
- **Secrets Detection**: Finds hardcoded secrets and sensitive information using detect-secrets
- **Configurable Scanning**: Customize scan settings via `config/security_config.json`
- **Detailed Reporting**: Generates both JSON and Markdown reports with findings

### Running Security Scans

```bash
# Run all security scans
python scripts/security_scan.py

# Run specific scan types
python scripts/security_scan.py --scan-type dependencies
python scripts/security_scan.py --scan-type code
python scripts/security_scan.py --scan-type secrets

# Specify project root and config file
python scripts/security_scan.py --project-root /path/to/project --config /path/to/config.json
```

### Security Configuration

The security scanner can be configured via `config/security_config.json`:

```json
{
    "scan_settings": {
        "dependencies": {
            "enabled": true,
            "check_updates": true,
            "ignore_patterns": ["test/*", "tests/*", "docs/*"]
        },
        "code": {
            "enabled": true,
            "ignore_patterns": ["test/*", "tests/*", "docs/*"],
            "severity_levels": ["LOW", "MEDIUM", "HIGH"],
            "confidence_levels": ["LOW", "MEDIUM", "HIGH"]
        },
        "secrets": {
            "enabled": true,
            "ignore_patterns": ["test/*", "tests/*", "docs/*"],
            "detectors": ["AWSKeyDetector", "BasicAuthDetector", "PrivateKeyDetector"]
        }
    },
    "reporting": {
        "output_dir": "security_reports",
        "formats": ["json", "markdown"],
        "include_source": true,
        "max_file_size": 1048576,
        "retention_days": 30
    }
}
```

### CI Integration

The security scanner can be integrated into your CI/CD pipeline to automatically check for vulnerabilities:

```yaml
# Example GitHub Actions workflow
jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Run security scan
        run: python scripts/security_scan.py
```
