# Deep Recall Streamlit Memories

This example demonstrates how to use Deep Recall for semantic memory within a Streamlit application. It uses a two-table database design with separate tables for embeddings and memories.

## Database Design

The memory system uses two main database tables:

1. `embeddings` - For storing vector embeddings:
   - `id` - Primary key
   - `vector` - The vector representation (384 dimensions)
   - `created_at` - Timestamp

2. `memories` - For storing text and metadata:
   - `id` - Primary key
   - `user_id` - ID of the user who owns the memory
   - `text` - The text content of the memory
   - `metadata` - Additional information in JSON format
   - `embedding_id` - Foreign key to the embeddings table
   - `created_at` - Timestamp

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- pip

## Setup

1. Clone the Deep Recall repository:

```bash
git clone https://github.com/your-repo/deep-recall.git
cd deep-recall/examples/streamlit-memories
```

2. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install Python dependencies
- Start the PostgreSQL database with pgvector extension
- Initialize the database schema
- Fix any potential issues with the search function

## Running the Application

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This will start the web interface, which you can access at http://localhost:8501.

## Manual Setup (if needed)

If you prefer to set up manually or the setup script doesn't work, follow these steps:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the PostgreSQL database:

```bash
docker-compose -f docker-compose.db.yml up -d
```

3. Initialize the database:

```bash
python scripts/setup_db.py
python scripts/fix_search_function.py
```

4. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## How It Works

The application uses the following components:

- **PostgreSQL with pgvector**: For storing text data and vector embeddings
- **Sentence Transformers**: For generating embeddings from text
- **FAISS**: As a fallback vector search engine
- **Streamlit**: For the web user interface

## Features

- Store memories with metadata
- Retrieve memories based on semantic similarity
- Test memory retention and retrieval
- Visualize memory relevance

## Customization

You can customize the application by modifying:

- `streamlit_app.py`: The main Streamlit interface
- `memory/models.py`: Memory data models
- `memory/memory_store.py`: Memory storage implementation
- `memory/semantic_search.py`: Semantic search implementation

## Environment Variables

The following environment variables can be used to configure the application:

- `DB_HOST`: Database host (default: db)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: recall_memories_db)
- `DB_USER`: Database user (default: postgres)
- `DB_PASSWORD`: Database password (default: postgres)
- `API_URL`: API URL for the Streamlit app (default: http://localhost:8404)

## Troubleshooting

- **Database Connection Issues**: Ensure the PostgreSQL container is running with `docker ps`
- **Vector Search Errors**: Check that the pgvector extension is properly installed
- **Import Errors**: Verify that all dependencies are installed

## License

This project is licensed under the MIT License - see the LICENSE file for details. 