import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from memory.memory_store import MemoryStore


class TestMemoryStore(unittest.TestCase):
    """Unit tests for the MemoryStore class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for vector database
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_vector_db")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")

        # Initialize memory store with FAISS vector database
        self.embedding_dim = 10
        self.memory_store = MemoryStore(
            embedding_dim=self.embedding_dim,
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path,
        )

    def tearDown(self):
        """Clean up resources after each test method."""
        self.temp_dir.cleanup()

    def test_add_text(self):
        """Test adding text to the memory store."""
        # Create a test embedding
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        # Create test metadata
        metadata = {"user_id": "test_user", "tags": ["test"]}

        # Add to memory store
        memory_id = self.memory_store.add_text("Test text", embedding, metadata)

        # Verify ID was returned
        self.assertIsInstance(memory_id, int)

        # Verify text was stored
        self.assertEqual(self.memory_store.text_data[memory_id], "Test text")

        # Verify metadata was stored
        self.assertEqual(self.memory_store.metadata[memory_id]["user_id"], "test_user")
        self.assertEqual(self.memory_store.metadata[memory_id]["tags"], ["test"])

        # Verify timestamp was added
        self.assertIn("timestamp", self.memory_store.metadata[memory_id])

    def test_search(self):
        """Test searching for similar texts."""
        # Add multiple texts with embeddings
        embeddings = []
        for i in range(5):
            # Create embeddings with increasing similarity to the search query
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            embedding[0] = 0.1 * i  # This will affect similarity
            embedding = (
                embedding / np.linalg.norm(embedding)
                if np.linalg.norm(embedding) > 0
                else embedding
            )
            embeddings.append(embedding)

            self.memory_store.add_text(f"Test text {i}", embedding, {"index": i})

        # Create search query embedding - should be most similar to the last embedding
        query_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        query_embedding[0] = 0.5
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        results = self.memory_store.search(query_embedding, k=3)

        # Verify structure of results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Verify each result has expected fields
        for result in results:
            self.assertIn("id", result)
            self.assertIn("text", result)
            self.assertIn("similarity", result)
            self.assertIn("metadata", result)

        # Results should be sorted by decreasing similarity
        for i in range(1, len(results)):
            self.assertGreaterEqual(
                results[i - 1]["similarity"], results[i]["similarity"]
            )

    def test_filter_metadata(self):
        """Test filtering search results by metadata."""
        # Add texts with different user_ids
        for i in range(10):
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            # Alternate between user_1 and user_2
            user_id = f"user_{i % 2 + 1}"

            self.memory_store.add_text(
                f"Text by {user_id}", embedding, {"user_id": user_id, "index": i}
            )

        # Create search query
        query_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search with filter for user_1
        results_user1 = self.memory_store.search(
            query_embedding, k=10, filter_metadata={"user_id": "user_1"}
        )

        # Verify all results are from user_1
        for result in results_user1:
            self.assertEqual(result["metadata"]["user_id"], "user_1")

        # Search with filter for user_2
        results_user2 = self.memory_store.search(
            query_embedding, k=10, filter_metadata={"user_id": "user_2"}
        )

        # Verify all results are from user_2
        for result in results_user2:
            self.assertEqual(result["metadata"]["user_id"], "user_2")

    def test_timestamp_range_filter(self):
        """Test filtering by timestamp range."""
        # Add texts with timestamps spaced one day apart
        now = datetime.now()
        for i in range(5):
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            # Create timestamp i days ago
            timestamp = (now - timedelta(days=i)).isoformat()

            self.memory_store.add_text(
                f"Text from {timestamp}",
                embedding,
                {"timestamp": timestamp, "index": i},
            )

        # Create search query
        query_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search for entries between 2 and 3 days ago
        start_timestamp = (now - timedelta(days=3)).isoformat()
        end_timestamp = (now - timedelta(days=1)).isoformat()

        results = self.memory_store.search(
            query_embedding,
            k=10,
            filter_metadata={"timestamp_range": [start_timestamp, end_timestamp]},
        )

        # Should only have entries from day 1 and 2
        for result in results:
            timestamp = result["metadata"]["timestamp"]
            self.assertGreaterEqual(timestamp, start_timestamp)
            self.assertLessEqual(timestamp, end_timestamp)

    def test_delete_memory(self):
        """Test deleting memories."""
        # Add a text
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        memory_id = self.memory_store.add_text("Text to delete", embedding)

        # Verify it exists
        self.assertIn(memory_id, self.memory_store.text_data)

        # Delete it
        success = self.memory_store.delete_memory(memory_id)

        # Verify deletion was successful
        self.assertTrue(success)

        # Verify metadata is marked as deleted
        self.assertTrue(self.memory_store.metadata[memory_id]["deleted"])
        self.assertIn("deletion_timestamp", self.memory_store.metadata[memory_id])

        # Test deleting non-existent memory
        non_existent_id = 9999
        success = self.memory_store.delete_memory(non_existent_id)
        self.assertFalse(success)

    def test_save_and_load_index(self):
        """Test saving and loading the index."""
        # Add some texts
        for i in range(5):
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            self.memory_store.add_text(f"Test text {i}", embedding, {"index": i})

        # Save the index
        self.memory_store.save_index()

        # Verify files were created
        self.assertTrue(os.path.exists(self.db_path))
        self.assertTrue(os.path.exists(self.metadata_path))

        # Create a new memory store that will load from the saved files
        new_memory_store = MemoryStore(
            embedding_dim=self.embedding_dim,
            db_type="faiss",
            db_path=self.db_path,
            metadata_path=self.metadata_path,
        )

        # Verify the data was loaded correctly
        self.assertEqual(len(new_memory_store.text_data), 5)
        self.assertEqual(len(new_memory_store.metadata), 5)

        # Verify texts are the same
        for i in range(5):
            found = False
            for memory_id, text in new_memory_store.text_data.items():
                if text == f"Test text {i}":
                    found = True
                    self.assertEqual(new_memory_store.metadata[memory_id]["index"], i)
                    break
            self.assertTrue(found, f"Text {i} not found in loaded data")

    def test_set_backup_interval(self):
        """Test setting the backup interval."""
        # Default interval
        default_interval = self.memory_store.backup_interval

        # Set new interval
        new_interval = 7200  # 2 hours
        self.memory_store.set_backup_interval(new_interval)

        # Verify interval was updated
        self.assertEqual(self.memory_store.backup_interval, new_interval)

    def test_get_stats(self):
        """Test getting store statistics."""
        # Add some texts
        for i in range(3):
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            self.memory_store.add_text(f"Test text {i}", embedding)

        # Delete one
        self.memory_store.delete_memory(0)

        # Get stats
        stats = self.memory_store.get_stats()

        # Verify stats
        self.assertEqual(stats["total_vectors"], 3)
        self.assertEqual(stats["active_memories"], 2)
        self.assertEqual(stats["deleted_memories"], 1)
        self.assertEqual(stats["embedding_dimension"], self.embedding_dim)
        self.assertEqual(stats["vector_db_type"], "faiss")


if __name__ == "__main__":
    unittest.main()
