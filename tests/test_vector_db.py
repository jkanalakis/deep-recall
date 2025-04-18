import unittest
import numpy as np
import os
import tempfile
import faiss

from memory.vector_db import VectorDBFactory
from memory.vector_db.faiss_db import FaissVectorDB


class TestVectorDBFactory(unittest.TestCase):
    """Tests for the VectorDBFactory class."""
    
    def test_create_db(self):
        """Test creating different vector database types."""
        # Test FAISS DB creation
        db = VectorDBFactory.create_db("faiss", dimension=10)
        self.assertIsInstance(db, FaissVectorDB)
        
        # Test with invalid DB type
        with self.assertRaises(ValueError):
            VectorDBFactory.create_db("invalid_db_type", dimension=10)


class TestFaissVectorDB(unittest.TestCase):
    """Tests for the FaissVectorDB implementation."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.dimension = 10
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test embeddings
        self.num_vectors = 10
        self.test_vectors = np.random.rand(self.num_vectors, self.dimension).astype(np.float32)
        # Normalize for inner product similarity
        for i in range(self.num_vectors):
            self.test_vectors[i] = self.test_vectors[i] / np.linalg.norm(self.test_vectors[i])
        
        # Create query vectors
        self.num_queries = 3
        self.query_vectors = np.random.rand(self.num_queries, self.dimension).astype(np.float32)
        # Normalize for inner product similarity
        for i in range(self.num_queries):
            self.query_vectors[i] = self.query_vectors[i] / np.linalg.norm(self.query_vectors[i])
        
    def tearDown(self):
        """Clean up resources after each test method."""
        self.temp_dir.cleanup()
        
    def test_add_and_search_flat_ip(self):
        """Test adding vectors and searching with flat inner product index."""
        db = FaissVectorDB(dimension=self.dimension, index_type="flat", metric="ip")
        
        # Add vectors with auto-generated IDs
        ids = db.add(self.test_vectors)
        
        # Verify IDs were returned
        self.assertEqual(len(ids), self.num_vectors)
        
        # Verify vector count
        self.assertEqual(db.get_vector_count(), self.num_vectors)
        
        # Search
        k = 3
        similarities, result_ids = db.search(self.query_vectors, k=k)
        
        # Verify search results shape
        self.assertEqual(similarities.shape, (self.num_queries, k))
        self.assertEqual(result_ids.shape, (self.num_queries, k))
        
        # Verify similarities are between -1 and 1 for inner product
        self.assertTrue(np.all(similarities >= -1))
        self.assertTrue(np.all(similarities <= 1))
        
        # Verify IDs are valid
        self.assertTrue(np.all((result_ids >= 0) | (result_ids == -1)))
        
    def test_add_and_search_flat_l2(self):
        """Test adding vectors and searching with flat L2 index."""
        db = FaissVectorDB(dimension=self.dimension, index_type="flat", metric="l2")
        
        # Add vectors with custom IDs
        custom_ids = list(range(100, 100 + self.num_vectors))
        ids = db.add(self.test_vectors, custom_ids)
        
        # Verify returned IDs match custom IDs
        self.assertEqual(ids, custom_ids)
        
        # Search
        k = 3
        distances, result_ids = db.search(self.query_vectors, k=k)
        
        # Verify distances are non-negative for L2
        self.assertTrue(np.all(distances >= 0))
        
        # Verify some returned IDs match custom IDs
        for query_results in result_ids:
            for result_id in query_results:
                if result_id != -1:
                    self.assertIn(result_id, custom_ids)
                    
    def test_add_and_search_ivf(self):
        """Test adding vectors and searching with IVF index."""
        # For IVF, we need more vectors for training
        num_vectors = 100
        vectors = np.random.rand(num_vectors, self.dimension).astype(np.float32)
        
        db = FaissVectorDB(
            dimension=self.dimension, 
            index_type="ivf", 
            metric="ip",
            nlist=10,  # Number of clusters
            nprobe=3   # Number of clusters to visit during search
        )
        
        # Add vectors
        ids = db.add(vectors)
        
        # Search
        k = 5
        similarities, result_ids = db.search(self.query_vectors, k=k)
        
        # Basic validation of results
        self.assertEqual(similarities.shape, (self.num_queries, k))
        self.assertEqual(result_ids.shape, (self.num_queries, k))
        
    def test_save_and_load(self):
        """Test saving and loading the index."""
        db = FaissVectorDB(dimension=self.dimension)
        
        # Add vectors
        ids = db.add(self.test_vectors)
        
        # Save index
        save_path = os.path.join(self.temp_dir.name, "test_faiss")
        success = db.save(save_path)
        self.assertTrue(success)
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(save_path, "faiss_index.bin")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "faiss_metadata.json")))
        
        # Create new DB and load
        new_db = FaissVectorDB(dimension=self.dimension)
        success = new_db.load(save_path)
        self.assertTrue(success)
        
        # Verify vector count
        self.assertEqual(new_db.get_vector_count(), self.num_vectors)
        
        # Verify search results are similar between original and loaded DB
        k = 3
        orig_similarities, orig_ids = db.search(self.query_vectors, k=k)
        new_similarities, new_ids = new_db.search(self.query_vectors, k=k)
        
        # Results should be the same
        np.testing.assert_array_almost_equal(orig_similarities, new_similarities, decimal=5)
        np.testing.assert_array_equal(orig_ids, new_ids)
        
    def test_delete(self):
        """Test deleting vectors."""
        db = FaissVectorDB(dimension=self.dimension)
        
        # Add vectors with custom IDs
        custom_ids = list(range(100, 100 + self.num_vectors))
        db.add(self.test_vectors, custom_ids)
        
        # Delete half of the vectors
        ids_to_delete = custom_ids[:self.num_vectors // 2]
        success = db.delete(ids_to_delete)
        self.assertTrue(success)
        
        # Search
        k = 3
        similarities, result_ids = db.search(self.query_vectors, k=k)
        
        # Verify deleted IDs are not in results
        for query_results in result_ids:
            for result_id in query_results:
                if result_id != -1:
                    self.assertNotIn(result_id, ids_to_delete)
                    
    def test_optimize_index(self):
        """Test index optimization."""
        # Create a flat index with enough vectors to trigger optimization
        db = FaissVectorDB(dimension=self.dimension, index_type="flat")
        
        # Add more vectors to potentially trigger optimization
        many_vectors = np.random.rand(20000, self.dimension).astype(np.float32)
        db.add(many_vectors)
        
        # Call optimize
        success = db.optimize_index()
        # Note: May return True or False depending on whether optimization was needed
        
        # Verify we can still search
        k = 3
        similarities, result_ids = db.search(self.query_vectors, k=k)
        self.assertEqual(similarities.shape, (self.num_queries, k))


# Optional: Tests for other vector database implementations
# These would follow a similar pattern to the FAISS tests

class TestQdrantVectorDB(unittest.TestCase):
    """Tests for the QdrantVectorDB implementation."""
    
    @unittest.skip("Requires Qdrant server to be running")
    def test_basic_qdrant_operations(self):
        """Basic test for Qdrant vector DB."""
        from memory.vector_db.qdrant_db import QdrantVectorDB
        
        dimension = 10
        db = QdrantVectorDB(
            dimension=dimension,
            collection_name="test_collection",
            location=":memory:"  # Use in-memory storage for testing
        )
        
        # Add vectors
        vectors = np.random.rand(5, dimension).astype(np.float32)
        ids = db.add(vectors)
        
        # Search
        query = np.random.rand(1, dimension).astype(np.float32)
        similarities, result_ids = db.search(query, k=3)
        
        # Basic validation
        self.assertEqual(similarities.shape, (1, 3))
        self.assertEqual(result_ids.shape, (1, 3))


if __name__ == "__main__":
    unittest.main() 