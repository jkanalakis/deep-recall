"""
Vector database module for deep-recall.
This module provides integrations with various vector databases.
"""

from memory.vector_db.base import VectorDB, VectorDBFactory
from memory.vector_db.faiss_db import FaissVectorDB

# Import other vector databases conditionally
try:
    from memory.vector_db.qdrant_db import QdrantVectorDB
except ImportError:
    pass

__all__ = [
    'VectorDB',
    'VectorDBFactory',
    'FaissVectorDB',
]

# Add other database classes to __all__ if available
try:
    __all__.append('QdrantVectorDB')
except NameError:
    pass 