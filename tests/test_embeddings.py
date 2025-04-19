import asyncio
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from memory.embeddings import EmbeddingModelFactory
from memory.embeddings.base import EmbeddingModel
from memory.embeddings.sentence_transformer import SentenceTransformerModel
from memory.embeddings.transformer import TransformerEmbeddingModel


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""

    def __init__(self, *args, **kwargs):
        self.dimension = 768

    def get_sentence_embedding_dimension(self):
        """Return the embedding dimension."""
        return self.dimension

    def encode(self, texts, **kwargs):
        """Return mock embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)
        # Generate deterministic but different embedding for each text
        embeddings = np.zeros((batch_size, self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            # Generate a deterministic embedding using text hash
            hash_val = hash(text) % 10000
            np.random.seed(hash_val)
            embedding = np.random.rand(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[i] = embedding

        return embeddings


class TestEmbeddingModels(unittest.TestCase):
    """Unit tests for embedding models."""

    def test_embedding_model_factory(self):
        """Test the EmbeddingModelFactory class."""
        # Test creating a sentence transformer model
        with patch(
            "memory.embeddings.sentence_transformer.SentenceTransformer",
            MockSentenceTransformer,
        ):
            model = EmbeddingModelFactory.create_model(
                "sentence_transformer", model_name="all-MiniLM-L6-v2"
            )
            self.assertIsInstance(model, SentenceTransformerModel)

        # Test creating a transformer model
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        # Create a dictionary-like object with a 'to' method
        class MockTokenizerOutput(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self["input_ids"] = torch.randint(0, 1000, (1, 10))
                self["attention_mask"] = torch.ones(1, 10)

            def to(self, device):
                # Move tensors to device
                return {
                    k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()
                }

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MockTokenizerOutput()

        # Mock the model outputs
        class MockOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(1, 10, 768)
                self.attention_mask = torch.ones(1, 10)

            def get(self, key, default):
                if key == "attention_mask":
                    return self.attention_mask
                return default

        mock_output = MockOutput()
        mock_model.return_value = mock_output

        with patch(
            "memory.embeddings.transformer.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            with patch(
                "memory.embeddings.transformer.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                model = EmbeddingModelFactory.create_model(
                    "transformer", model_name="bert-base-uncased"
                )
                self.assertIsInstance(model, TransformerEmbeddingModel)

        # Test invalid model type
        with self.assertRaises(ValueError):
            EmbeddingModelFactory.create_model("invalid_model_type")

    def test_sentence_transformer_model(self):
        """Test the SentenceTransformerModel class."""
        with patch(
            "memory.embeddings.sentence_transformer.SentenceTransformer",
            MockSentenceTransformer,
        ):
            model = SentenceTransformerModel(model_name="all-MiniLM-L6-v2")

            # Test embedding a single text
            text = "This is a test sentence."
            embedding = model.embed_text(text)

            # Check embedding shape and type
            self.assertEqual(embedding.shape, (1, 768))
            self.assertEqual(embedding.dtype, np.float32)

            # Test embedding multiple texts
            texts = ["First sentence.", "Second sentence.", "Third sentence."]
            embeddings = model.embed_text(texts)

            # Check embeddings shape and type
            self.assertEqual(embeddings.shape, (3, 768))
            self.assertEqual(embeddings.dtype, np.float32)

            # Test async embedding
            async def test_async():
                embedding = await model.embed_text_async("Async test sentence.")
                return embedding

            embedding = asyncio.run(test_async())
            self.assertEqual(embedding.shape, (1, 768))

            # Test get_embedding_dim
            self.assertEqual(model.get_embedding_dim(), 768)

    def test_transformer_model(self):
        """Test the TransformerEmbeddingModel class."""
        # Create mock auto model and tokenizer
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        # Create a dictionary-like object with a 'to' method
        class MockTokenizerOutput(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self["input_ids"] = torch.randint(0, 1000, (1, 10))
                self["attention_mask"] = torch.ones(1, 10)

            def to(self, device):
                # Move tensors to device
                return {
                    k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()
                }

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MockTokenizerOutput()

        # Mock the model outputs
        class MockOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(1, 10, 768)
                self.attention_mask = torch.ones(1, 10)

            def get(self, key, default):
                if key == "attention_mask":
                    return self.attention_mask
                return default

        mock_output = MockOutput()
        mock_model.return_value = mock_output

        with patch(
            "memory.embeddings.transformer.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            with patch(
                "memory.embeddings.transformer.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                model = TransformerEmbeddingModel(model_name="bert-base-uncased")

                # Test embedding a single text
                text = "This is a test sentence."
                embedding = model.embed_text(text)

                # Check embedding shape and type
                self.assertEqual(embedding.shape, (1, 768))

                # Test async embedding
                async def test_async():
                    embedding = await model.embed_text_async("Async test sentence.")
                    return embedding

                embedding = asyncio.run(test_async())
                self.assertEqual(embedding.shape, (1, 768))

                # Test get_embedding_dim
                self.assertEqual(model.get_embedding_dim(), 768)


class TestEmbeddingModelWithActualModel(unittest.TestCase):
    """
    Tests that attempt to load the actual models.
    These may be skipped if the models aren't available.
    """

    @unittest.skip("Requires downloading the actual model")
    def test_sentence_transformer_with_actual_model(self):
        """Test with actual SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformerModel(model_name="all-MiniLM-L6-v2")

            # Test on a single sentence
            embedding = model.embed_text("This is a test sentence.")
            self.assertEqual(
                embedding.shape, (384,)
            )  # all-MiniLM-L6-v2 has 384 dimensions

        except (ImportError, OSError):
            self.skipTest("SentenceTransformer model not available")


if __name__ == "__main__":
    unittest.main()
