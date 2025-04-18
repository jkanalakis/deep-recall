import unittest
import numpy as np
from typing import Dict, List, Any
import os
import tempfile
import json
from datetime import datetime

# Import the components to test
from memory.semantic_search import SemanticSearch
from memory.memory_retriever import MemoryRetriever
from memory.memory_store import MemoryStore
from memory.embeddings import EmbeddingModelFactory


class MockEmbeddingModel:
    """Mock embedding model for testing purposes."""
    
    def __init__(self, dimension=768):
        self.dimension = dimension
        
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding based on text content."""
        # Create a simple hash of the text
        hash_val = hash(text) % 10000
        
        # Create a deterministic embedding vector
        np.random.seed(hash_val)
        emb = np.random.rand(self.dimension).astype(np.float32)
        
        # Normalize the vector
        emb = emb / np.linalg.norm(emb)
        return emb
        
    async def embed_text_async(self, text: str) -> np.ndarray:
        """Async version of embed_text."""
        return self.embed_text(text)


class TestSemanticSearch(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for vector database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_vector_db")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")
        
        # Create mock embedding model
        self.embedding_dim = 768
        self.embedding_model = MockEmbeddingModel(dimension=self.embedding_dim)
        
        # Initialize memory store with FAISS vector database
        self.memory_store = MemoryStore(
            embedding_dim=self.embedding_dim,
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path
        )
        
        # Initialize semantic search
        self.semantic_search = SemanticSearch(
            memory_store=self.memory_store,
            embedding_model=self.embedding_model,
            default_similarity_threshold=0.5
        )
        
        # Add some test data
        self._add_test_data()
        
    def tearDown(self):
        """Clean up resources after each test method."""
        self.temp_dir.cleanup()
        
    def _add_test_data(self):
        """Add test data to the memory store."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning models need lots of training data",
            "Python is a popular programming language for data science",
            "Deep learning revolutionized the field of artificial intelligence",
            "Natural language processing helps computers understand human language",
            "The cat sat on the mat and took a nap",
            "Dogs are loyal companions and make great pets",
            "The Eiffel Tower is a famous landmark in Paris, France",
            "Climate change is a pressing global challenge",
            "Renewable energy sources include solar and wind power"
        ]
        
        # Generate metadata for each text
        for i, text in enumerate(texts):
            # Create embedding
            embedding = self.embedding_model.embed_text(text)
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "user_id": f"user_{i % 3}",  # Assign to 3 different users
                "tags": ["test", f"topic_{i % 5}"]  # 5 different topics
            }
            
            # Add to memory store
            self.memory_store.add_text(text, embedding, metadata)
    
    def test_basic_search(self):
        """Test basic search functionality."""
        # Search for something related to animals
        results = self.semantic_search.search(
            query="Animals like foxes and dogs",
            k=3
        )
        
        # Check structure of results
        self.assertIn("results", results)
        self.assertIn("pagination", results)
        self.assertIn("search_metadata", results)
        
        # Check pagination info
        self.assertEqual(results["pagination"]["page"], 1)
        self.assertEqual(results["pagination"]["items_per_page"], 50)
        
        # There should be results
        self.assertGreater(len(results["results"]), 0)
        
        # The first result should have text, similarity and metadata
        first_result = results["results"][0]
        self.assertIn("text", first_result)
        self.assertIn("similarity", first_result)
        self.assertIn("metadata", first_result)
        
        # Similarity should be a float between 0 and 1
        self.assertIsInstance(first_result["similarity"], float)
        self.assertGreaterEqual(first_result["similarity"], 0.0)
        self.assertLessEqual(first_result["similarity"], 1.0)
        
    def test_pagination(self):
        """Test pagination functionality."""
        # Search with pagination
        page1 = self.semantic_search.search(
            query="programming and technology",
            k=10,
            page=1,
            items_per_page=2
        )
        
        page2 = self.semantic_search.search(
            query="programming and technology",
            k=10,
            page=2,
            items_per_page=2
        )
        
        # Check pagination info
        self.assertEqual(page1["pagination"]["page"], 1)
        self.assertEqual(page1["pagination"]["items_per_page"], 2)
        self.assertEqual(page2["pagination"]["page"], 2)
        
        # Check that results are different
        if len(page1["results"]) > 0 and len(page2["results"]) > 0:
            self.assertNotEqual(
                page1["results"][0]["id"],
                page2["results"][0]["id"]
            )
        
    def test_filtering(self):
        """Test metadata filtering functionality."""
        # Search with metadata filter
        results = self.semantic_search.search(
            query="technology and science",
            filter_metadata={"user_id": "user_1"}
        )
        
        # All results should have user_id = user_1
        for result in results["results"]:
            self.assertEqual(result["metadata"]["user_id"], "user_1")
            
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        results = self.semantic_search.hybrid_search(
            query="cat mat",
            k=3
        )
        
        # Should have results
        self.assertGreater(len(results), 0)
        
        # Results should have combined score
        self.assertIn("combined_score", results[0])
        
        # The top result should contain the words "cat" and "mat"
        top_result_text = results[0]["text"].lower()
        self.assertTrue("cat" in top_result_text and "mat" in top_result_text)
        
    def test_different_metrics(self):
        """Test different similarity metrics."""
        # Search with different metrics
        cosine_results = self.semantic_search.search(
            query="artificial intelligence",
            metric="cosine"
        )
        
        euclidean_results = self.semantic_search.search(
            query="artificial intelligence",
            metric="euclidean"
        )
        
        dot_results = self.semantic_search.search(
            query="artificial intelligence",
            metric="dot"
        )
        
        # All should return results
        self.assertGreater(len(cosine_results["results"]), 0)
        self.assertGreater(len(euclidean_results["results"]), 0)
        self.assertGreater(len(dot_results["results"]), 0)
        
        # The metrics should be recorded correctly
        self.assertEqual(cosine_results["search_metadata"]["metric"], "cosine")
        self.assertEqual(euclidean_results["search_metadata"]["metric"], "euclidean")
        self.assertEqual(dot_results["search_metadata"]["metric"], "dot")


class TestMemoryRetrieverWithSemanticSearch(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for vector database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_vector_db")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")
        
        # Initialize memory store with FAISS vector database
        self.memory_store = MemoryStore(
            embedding_dim=768,
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path
        )
        
        # Mock the EmbeddingModelFactory.create_model function
        self.original_create_model = EmbeddingModelFactory.create_model
        EmbeddingModelFactory.create_model = lambda model_type, **kwargs: MockEmbeddingModel()
        
        # Initialize memory retriever
        self.memory_retriever = MemoryRetriever(
            memory_store=self.memory_store,
            model_type="sentence_transformer",
            model_name="mock-model"
        )
        
        # Add some test data
        self._add_test_data()
        
    def tearDown(self):
        """Clean up resources after each test method."""
        EmbeddingModelFactory.create_model = self.original_create_model
        self.temp_dir.cleanup()
        
    def _add_test_data(self):
        """Add test data to the memory retriever."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning models need lots of training data",
            "Python is a popular programming language for data science",
            "Deep learning revolutionized the field of artificial intelligence",
            "Natural language processing helps computers understand human language"
        ]
        
        # Add texts with metadata to memory
        for i, text in enumerate(texts):
            metadata = {
                "user_id": f"user_{i % 2}",
                "tags": ["test", f"topic_{i}"]
            }
            self.memory_retriever.add_to_memory(text, metadata)
    
    def test_search_memory(self):
        """Test search_memory method."""
        results = self.memory_retriever.search_memory(
            query="artificial intelligence and machine learning",
            k=3,
            metric="cosine"
        )
        
        # Check structure of results
        self.assertIn("results", results)
        self.assertIn("pagination", results)
        self.assertIn("search_metadata", results)
        
        # There should be results
        self.assertGreater(len(results["results"]), 0)
        
    def test_hybrid_search_memory(self):
        """Test hybrid_search_memory method."""
        results = self.memory_retriever.hybrid_search_memory(
            query="programming language Python",
            k=2
        )
        
        # Should have results
        self.assertGreater(len(results), 0)
        
        # Results should have combined score
        self.assertIn("combined_score", results[0])
        
    def test_search_by_metadata(self):
        """Test search_by_metadata method."""
        # Search by metadata only
        metadata_results = self.memory_retriever.search_by_metadata(
            filter_metadata={"user_id": "user_1"}
        )
        
        # All results should have user_id = user_1
        for result in metadata_results:
            self.assertEqual(result["metadata"]["user_id"], "user_1")
            
        # Search by metadata with semantic query
        hybrid_results = self.memory_retriever.search_by_metadata(
            filter_metadata={"user_id": "user_0"},
            query="artificial intelligence"
        )
        
        # All results should have user_id = user_0
        for result in hybrid_results:
            self.assertEqual(result["metadata"]["user_id"], "user_0")


if __name__ == "__main__":
    unittest.main() 