import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memory.memory_retriever import MemoryRetriever
from memory.memory_store import MemoryStore
from memory.semantic_search import SemanticSearch


class MockEmbeddingModel:
    """Mock embedding model for testing purposes."""

    def __init__(self, dimension=64):
        self.dimension = dimension

    def embed_text(self, text):
        """Generate a deterministic embedding based on text content."""
        # Create a simple hash of the text
        if isinstance(text, list):
            # For batched text
            embeddings = np.zeros((len(text), self.dimension), dtype=np.float32)
            for i, t in enumerate(text):
                embeddings[i] = self._generate_embedding(t)
            return embeddings
        else:
            # For single text
            return self._generate_embedding(text)

    def _generate_embedding(self, text):
        """Generate a single embedding."""
        hash_val = hash(text) % 10000

        # Create a deterministic embedding vector
        np.random.seed(hash_val)
        emb = np.random.rand(self.dimension).astype(np.float32)

        # Normalize the vector to unit length for cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    async def embed_text_async(self, text):
        """Async version of embed_text."""
        return self.embed_text(text)

    def get_embedding_dim(self):
        """Return the embedding dimension."""
        return self.dimension


class TestMemoryPipeline(unittest.TestCase):
    """Integration tests for the full memory pipeline."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for vector database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_vector_db")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")

        # Create mock embedding model with a smaller dimension for testing
        self.embedding_model = MockEmbeddingModel(
            dimension=64
        )  # Using smaller dimension for tests
        self.embedding_dim = self.embedding_model.get_embedding_dim()

        # Initialize memory store with FAISS vector database using the correct dimension
        self.memory_store = MemoryStore(
            embedding_dim=self.embedding_dim,
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path,
        )

        # Initialize semantic search
        self.semantic_search = SemanticSearch(
            memory_store=self.memory_store, embedding_model=self.embedding_model
        )

        # Initialize memory retriever
        self.memory_retriever = MemoryRetriever(
            memory_store=self.memory_store, embedding_model=self.embedding_model
        )

    def tearDown(self):
        """Clean up resources after each test method."""
        self.temp_dir.cleanup()

    def test_full_pipeline(self):
        """Test the full memory pipeline from adding to retrieval."""
        # Prepare test data
        test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning models need lots of training data",
            "Python is a popular programming language for data science",
            "Deep learning revolutionized the field of artificial intelligence",
            "Natural language processing helps computers understand human language",
        ]

        # Test metadata
        test_metadata = [
            {"user_id": "user1", "importance": "high", "category": "animals"},
            {
                "user_id": "user2",
                "importance": "medium",
                "category": "machine_learning",
            },
            {"user_id": "user1", "importance": "medium", "category": "programming"},
            {"user_id": "user2", "importance": "high", "category": "machine_learning"},
            {"user_id": "user3", "importance": "high", "category": "nlp"},
        ]

        # 1. Add texts to memory through the retriever
        for text, metadata in zip(test_texts, test_metadata):
            self.memory_retriever.add_to_memory(text, metadata)

        # 2. Test semantic search directly
        search_results = self.semantic_search.search(
            query="Tell me about artificial intelligence and machine learning", k=2
        )

        # Verify search results
        self.assertIn("results", search_results)
        self.assertEqual(len(search_results["results"]), 2)

        # 3. Test memory retrieval through the retriever
        retrieval_results = self.memory_retriever.search_memory(
            query="artificial intelligence and machine learning", k=2
        )

        # Verify retrieval results
        self.assertIn("results", retrieval_results)
        self.assertEqual(len(retrieval_results["results"]), 2)

        # 4. Test filtering by metadata
        filtered_results = self.memory_retriever.search_memory(
            query="programming languages", filter_metadata={"user_id": "user1"}, k=3
        )

        # Verify all results belong to user1
        for result in filtered_results["results"]:
            self.assertEqual(result["metadata"]["user_id"], "user1")

        # 5. Test hybrid search
        hybrid_results = self.memory_retriever.hybrid_search_memory(
            query="python programming", k=2
        )

        # Verify hybrid search results
        self.assertGreaterEqual(len(hybrid_results), 1)
        self.assertIn("combined_score", hybrid_results[0])

        # 6. Test pure metadata search
        metadata_results = self.memory_retriever.search_by_metadata(
            filter_metadata={"category": "machine_learning", "importance": "high"}
        )

        # Verify metadata search results
        self.assertEqual(len(metadata_results), 1)
        self.assertEqual(
            metadata_results[0]["metadata"]["category"], "machine_learning"
        )
        self.assertEqual(metadata_results[0]["metadata"]["importance"], "high")

    def test_pipeline_with_different_vector_dbs(self):
        """Test the memory pipeline with different vector database backends."""
        test_db_types = [
            "faiss"
        ]  # We could add "qdrant", "milvus", "chroma" if available

        test_text = "This is a test text for different vector databases"
        test_metadata = {"source": "test", "importance": "high"}
        query = "test different databases"

        for db_type in test_db_types:
            with self.subTest(db_type=db_type):
                # Create a new memory store with the current database type
                temp_dir = tempfile.TemporaryDirectory()
                db_path = os.path.join(temp_dir.name, f"test_{db_type}")
                metadata_path = os.path.join(
                    temp_dir.name, f"test_{db_type}_metadata.json"
                )

                memory_store = MemoryStore(
                    embedding_dim=self.embedding_dim,
                    db_type=db_type,
                    db_path=db_path,
                    metadata_path=metadata_path,
                )

                # Create retriever with the current store
                retriever = MemoryRetriever(
                    memory_store=memory_store, embedding_model=self.embedding_model
                )

                # Add text to memory
                retriever.add_to_memory(test_text, test_metadata)

                # Search memory
                results = retriever.search_memory(query=query, k=1)

                # Verify results
                self.assertEqual(len(results["results"]), 1)
                self.assertEqual(results["results"][0]["text"], test_text)
                self.assertEqual(results["results"][0]["metadata"]["source"], "test")

                # Clean up
                temp_dir.cleanup()

    def test_persistence(self):
        """Test that memory persists when saved and loaded."""
        # Add test data
        test_text = "This is a text that should persist across save and load"
        test_metadata = {"persistence": "test"}

        self.memory_retriever.add_to_memory(test_text, test_metadata)

        # Save the index
        self.memory_store.save_index()

        # Create a new memory store and retriever that will load from the saved files
        new_memory_store = MemoryStore(
            embedding_dim=self.embedding_dim,
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path,
        )

        new_retriever = MemoryRetriever(
            memory_store=new_memory_store, embedding_model=self.embedding_model
        )

        # Search for the text
        results = new_retriever.search_memory(query="text that should persist", k=1)

        # Verify the text was retrieved
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0]["text"], test_text)
        self.assertEqual(results["results"][0]["metadata"]["persistence"], "test")


class TestMemoryPipelineWithMocks(unittest.TestCase):
    """Integration tests using mocks for external dependencies."""

    def setUp(self):
        """Set up test environment with mocks."""
        # Patch the EmbeddingModelFactory
        self.embedding_model_patch = patch(
            "memory.embeddings.EmbeddingModelFactory.create_model",
            return_value=MockEmbeddingModel(
                dimension=64
            ),  # Use same dimension as other tests
        )
        self.embedding_model_patch.start()

        # Create a temporary directory for vector database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_vector_db")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")

    def tearDown(self):
        """Clean up resources after each test method."""
        self.embedding_model_patch.stop()
        self.temp_dir.cleanup()

    def test_memory_retriever_initialization(self):
        """Test that memory retriever can be initialized with factory-created embedding model."""
        # Initialize memory store with the same dimension as the mock model
        memory_store = MemoryStore(
            embedding_dim=64,  # Match the mock model's dimension
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path,
        )

        # Initialize memory retriever with model factory
        memory_retriever = MemoryRetriever(
            memory_store=memory_store,
            model_type="sentence_transformer",
            model_name="mock-model",
        )

        # Test basic functionality
        test_text = "Test text for mocked embedding model"
        memory_retriever.add_to_memory(test_text, {"source": "mock_test"})

        results = memory_retriever.search_memory(query="test mocked embedding", k=1)

        # Verify results
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0]["text"], test_text)


if __name__ == "__main__":
    unittest.main()
