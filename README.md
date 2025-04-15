# deep-recall
A hyper-personalized agent memory framework for open-source LLMs to store, retrieve, and seamlessly integrate past user interactions. This enables LLMs to tailor responses with relevant personal context. It’s lightweight, extensible, and easy to deploy on any cloud or local environment.

A **hyper-personalized agent memory framework** for open-source Large Language Models (LLMs). RecallChain stores, retrieves, and seamlessly integrates past user interactions—allowing LLMs to tailor responses with relevant personal context. It’s lightweight, extensible, and easy to deploy on any cloud or local environment.

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

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with enhancements and bug fixes. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our workflow and code style.

## License

Distributed under the [Apache 2.0 License](LICENSE). Make it yours, and help build the future of AI-driven personalized experiences!
