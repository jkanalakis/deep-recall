#!/usr/bin/env python3
# sqlalchemy_db_utils.py

"""
SQLAlchemy-based database utilities for the Deep Recall framework.

This module provides CRUD operations for the Deep Recall database using SQLAlchemy ORM.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text, create_engine)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.sql import func

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default database configuration
DEFAULT_DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "deep_recall"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

# Create the base class for SQLAlchemy models
Base = declarative_base()


# Define SQLAlchemy models
class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    preferences = Column(JSONB, default={})
    background = Column(Text)
    goals = Column(JSONB, default=[])
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    sessions = relationship("UserSession", back_populates="user")
    memories = relationship("Memory", back_populates="user")


class UserSession(Base):
    """Session model."""

    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String)
    topic = Column(String)
    previous_interactions = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default={})

    # Relationships
    user = relationship("User", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")
    memories = relationship("Memory", back_populates="session")


class EmbeddingModel(Base):
    """Embedding model metadata."""

    __tablename__ = "embedding_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    dimensions = Column(Integer)
    provider = Column(String)
    version = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    memories = relationship("Memory", back_populates="embedding_model")


class Memory(Base):
    """Memory model."""

    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float))
    source = Column(String, default="user")
    importance = Column(Float, default=0.5)
    category = Column(String)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    embedding_model_id = Column(UUID(as_uuid=True), ForeignKey("embedding_models.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    metadata = Column(JSONB, default={})

    # Relationships
    user = relationship("User", back_populates="memories")
    session = relationship("UserSession", back_populates="memories")
    embedding_model = relationship("EmbeddingModel", back_populates="memories")
    interactions = relationship("MemoryInteraction", back_populates="memory")


class Interaction(Base):
    """Interaction model."""

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    memory_count = Column(Integer, default=0)
    was_summarized = Column(Boolean, default=False)
    model_used = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})

    # Relationships
    session = relationship("UserSession", back_populates="interactions")
    feedback = relationship("Feedback", back_populates="interaction")
    memory_interactions = relationship(
        "MemoryInteraction", back_populates="interaction"
    )


class MemoryInteraction(Base):
    """Memory-Interaction mapping model."""

    __tablename__ = "memory_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id"), nullable=False)
    interaction_id = Column(
        UUID(as_uuid=True), ForeignKey("interactions.id"), nullable=False
    )
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    memory = relationship("Memory", back_populates="interactions")
    interaction = relationship("Interaction", back_populates="memory_interactions")


class Feedback(Base):
    """Feedback model."""

    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interaction_id = Column(
        UUID(as_uuid=True), ForeignKey("interactions.id"), nullable=False
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    feedback_text = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    interaction = relationship("Interaction", back_populates="feedback")


class DeepRecallORM:
    """SQLAlchemy ORM access layer for the Deep Recall framework."""

    def __init__(
        self,
        db_config: Dict[str, Any] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the SQLAlchemy engine and session.

        Args:
            db_config: Database connection parameters
            embedding_model: Name of the SentenceTransformer model to use for embeddings
        """
        self.db_config = db_config or DEFAULT_DB_CONFIG
        self.embedding_model_name = embedding_model
        self.embedding_model = None

        # Create database URL
        db_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"

        # Create engine and session
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model {embedding_model}: {e}")

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        logger.info("Created database tables")

    def get_db_session(self) -> Session:
        """Get a new SQLAlchemy session."""
        return self.SessionLocal()

    def __enter__(self):
        """Context manager entry."""
        self.db_session = self.get_db_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.db_session.close()

    # User CRUD operations
    def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        preferences: Dict[str, Any] = None,
        background: str = None,
        goals: List[str] = None,
    ) -> User:
        """
        Create a new user.

        Args:
            username: User's username
            email: User's email
            password_hash: Hashed password
            preferences: User preferences
            background: User background information
            goals: User goals

        Returns:
            User object
        """
        with self.get_db_session() as session:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                preferences=preferences or {},
                background=background,
                goals=goals or [],
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"Created user: {username}")
            return user

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.

        Args:
            user_id: ID of the user

        Returns:
            User object or None if not found
        """
        with self.get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by username.

        Args:
            username: Username to look up

        Returns:
            User object or None if not found
        """
        with self.get_db_session() as session:
            user = session.query(User).filter(User.username == username).first()
            return user

    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """
        Update a user.

        Args:
            user_id: ID of the user
            **kwargs: Fields to update

        Returns:
            Updated User object or None if not found
        """
        with self.get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User not found: {user_id}")
                return None

            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)

            session.commit()
            session.refresh(user)
            logger.info(f"Updated user: {user.username}")
            return user

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.

        Args:
            user_id: ID of the user

        Returns:
            True if deleted, False if not found
        """
        with self.get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User not found: {user_id}")
                return False

            session.delete(user)
            session.commit()
            logger.info(f"Deleted user: {user.username}")
            return True

    # Session CRUD operations
    def create_session(
        self,
        user_id: str,
        name: str = None,
        topic: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[UserSession]:
        """
        Create a new session.

        Args:
            user_id: ID of the user
            name: Session name
            topic: Session topic
            metadata: Additional metadata

        Returns:
            Session object or None if user not found
        """
        with self.get_db_session() as session:
            # Verify user exists
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User not found: {user_id}")
                return None

            new_session = UserSession(
                user_id=user_id,
                name=name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                topic=topic,
                metadata=metadata or {},
            )
            session.add(new_session)
            session.commit()
            session.refresh(new_session)
            logger.info(f"Created session for user {user_id}: {name}")
            return new_session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a session by ID.

        Args:
            session_id: ID of the session

        Returns:
            Session object or None if not found
        """
        with self.get_db_session() as session:
            result = (
                session.query(UserSession).filter(UserSession.id == session_id).first()
            )
            return result

    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """
        Get all sessions for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of Session objects
        """
        with self.get_db_session() as session:
            sessions = (
                session.query(UserSession).filter(UserSession.user_id == user_id).all()
            )
            return sessions

    def update_session(self, session_id: str, **kwargs) -> Optional[UserSession]:
        """
        Update a session.

        Args:
            session_id: ID of the session
            **kwargs: Fields to update

        Returns:
            Updated Session object or None if not found
        """
        with self.get_db_session() as session:
            result = (
                session.query(UserSession).filter(UserSession.id == session_id).first()
            )
            if not result:
                logger.warning(f"Session not found: {session_id}")
                return None

            for key, value in kwargs.items():
                if hasattr(result, key):
                    setattr(result, key, value)

            session.commit()
            session.refresh(result)
            logger.info(f"Updated session: {result.name}")
            return result

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: ID of the session

        Returns:
            True if deleted, False if not found
        """
        with self.get_db_session() as session:
            result = (
                session.query(UserSession).filter(UserSession.id == session_id).first()
            )
            if not result:
                logger.warning(f"Session not found: {session_id}")
                return False

            session.delete(result)
            session.commit()
            logger.info(f"Deleted session: {result.name}")
            return True

    # Memory CRUD operations
    def get_embedding_model(self) -> Optional[EmbeddingModel]:
        """
        Get or create the embedding model record.

        Returns:
            EmbeddingModel object
        """
        with self.get_db_session() as session:
            model = (
                session.query(EmbeddingModel)
                .filter(EmbeddingModel.name == self.embedding_model_name)
                .first()
            )

            if not model and self.embedding_model:
                # Model not found, create it
                model = EmbeddingModel(
                    name=self.embedding_model_name,
                    dimensions=self.embedding_model.get_sentence_embedding_dimension(),
                    provider="sentence-transformers",
                    version="1.0",
                    is_active=True,
                )
                session.add(model)
                session.commit()
                session.refresh(model)

            return model

    def create_memory(
        self,
        user_id: str,
        text: str,
        source: str = "user",
        importance: float = 0.5,
        category: str = None,
        session_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[Memory]:
        """
        Create a new memory.

        Args:
            user_id: ID of the user
            text: Memory text content
            source: Source of the memory
            importance: Importance score (0-1)
            category: Category of the memory
            session_id: Optional session ID
            metadata: Additional metadata

        Returns:
            Memory object or None if user not found
        """
        with self.get_db_session() as session:
            # Verify user exists
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.warning(f"User not found: {user_id}")
                return None

            # Generate embedding for the text
            embedding = None
            embedding_model_id = None

            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(text).tolist()
                    # Get embedding model ID
                    model = self.get_embedding_model()
                    if model:
                        embedding_model_id = model.id
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")

            memory = Memory(
                user_id=user_id,
                text=text,
                embedding=embedding,
                source=source,
                importance=importance,
                category=category,
                session_id=session_id,
                embedding_model_id=embedding_model_id,
                metadata=metadata or {},
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            logger.info(f"Stored memory for user {user_id}: {text[:50]}...")
            return memory

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            Memory object or None if not found
        """
        with self.get_db_session() as session:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            return memory

    def get_user_memories(
        self, user_id: str, category: str = None, session_id: str = None
    ) -> List[Memory]:
        """
        Get memories for a user with optional filters.

        Args:
            user_id: ID of the user
            category: Filter by category
            session_id: Filter by session ID

        Returns:
            List of Memory objects
        """
        with self.get_db_session() as session:
            query = session.query(Memory).filter(Memory.user_id == user_id)

            if category:
                query = query.filter(Memory.category == category)

            if session_id:
                query = query.filter(Memory.session_id == session_id)

            memories = query.all()
            return memories

    def update_memory(self, memory_id: str, **kwargs) -> Optional[Memory]:
        """
        Update a memory.

        Args:
            memory_id: ID of the memory
            **kwargs: Fields to update

        Returns:
            Updated Memory object or None if not found
        """
        with self.get_db_session() as session:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            if not memory:
                logger.warning(f"Memory not found: {memory_id}")
                return None

            # If updating text, regenerate embedding
            if "text" in kwargs and self.embedding_model:
                try:
                    kwargs["embedding"] = self.embedding_model.encode(
                        kwargs["text"]
                    ).tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")

            for key, value in kwargs.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)

            session.commit()
            session.refresh(memory)
            logger.info(f"Updated memory: {memory.id}")
            return memory

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            True if deleted, False if not found
        """
        with self.get_db_session() as session:
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            if not memory:
                logger.warning(f"Memory not found: {memory_id}")
                return False

            session.delete(memory)
            session.commit()
            logger.info(f"Deleted memory: {memory_id}")
            return True

    def get_memories_by_semantic_search(
        self, user_id: str, query: str, k: int = 10, filters: Dict[str, Any] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories based on semantic similarity to a query.

        Args:
            user_id: ID of the user
            query: Query text to match against memories
            k: Number of memories to retrieve
            filters: Additional filters to apply

        Returns:
            List of (Memory, similarity_score) tuples
        """
        if not self.embedding_model:
            logger.warning("No embedding model available for semantic search")
            return []

        # Generate embedding for the query
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return []

        # This is a simplistic approach - in a real application, you would use
        # PostgreSQL's vector similarity features or an external vector search service
        with self.get_db_session() as session:
            memories = session.query(Memory).filter(Memory.user_id == user_id)

            if filters:
                if "category" in filters:
                    memories = memories.filter(Memory.category == filters["category"])

                if "source" in filters:
                    memories = memories.filter(Memory.source == filters["source"])

                if "session_id" in filters:
                    memories = memories.filter(
                        Memory.session_id == filters["session_id"]
                    )

            # Filter out memories without embeddings
            memories = memories.filter(Memory.embedding.isnot(None)).all()

            # Calculate similarity for each memory (this is inefficient - in production use vector DB)
            memory_with_scores = []
            for memory in memories:
                if memory.embedding:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(
                        query_embedding, memory.embedding
                    )
                    memory_with_scores.append((memory, similarity))

            # Sort by similarity (highest first) and take top k
            memory_with_scores.sort(key=lambda x: x[1], reverse=True)
            return memory_with_scores[:k]

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Interaction CRUD operations
    def create_interaction(
        self,
        session_id: str,
        user_id: str,
        prompt: str,
        response: str,
        memory_ids: List[str] = None,
        model_used: str = None,
        was_summarized: bool = False,
        metadata: Dict[str, Any] = None,
    ) -> Optional[Interaction]:
        """
        Create a new interaction.

        Args:
            session_id: ID of the session
            user_id: ID of the user
            prompt: User prompt/query
            response: Generated response
            memory_ids: IDs of memories used in this interaction
            model_used: Name of the model used
            was_summarized: Whether memory context was summarized
            metadata: Additional metadata

        Returns:
            Interaction object or None if session not found
        """
        with self.get_db_session() as session:
            # Verify session exists
            chat_session = (
                session.query(UserSession).filter(UserSession.id == session_id).first()
            )
            if not chat_session:
                logger.warning(f"Session not found: {session_id}")
                return None

            memory_count = len(memory_ids) if memory_ids else 0

            interaction = Interaction(
                session_id=session_id,
                user_id=user_id,
                prompt=prompt,
                response=response,
                memory_count=memory_count,
                was_summarized=was_summarized,
                model_used=model_used,
                metadata=metadata or {},
            )
            session.add(interaction)
            session.commit()
            session.refresh(interaction)

            # If memory IDs are provided, link them to the interaction
            if memory_ids:
                for memory_id in memory_ids:
                    # Default relevance score
                    relevance_score = 0.75
                    memory_interaction = MemoryInteraction(
                        memory_id=memory_id,
                        interaction_id=interaction.id,
                        relevance_score=relevance_score,
                    )
                    session.add(memory_interaction)

                session.commit()

            logger.info(
                f"Stored interaction for user {user_id} in session {session_id}"
            )
            return interaction

    def get_interaction(self, interaction_id: str) -> Optional[Interaction]:
        """
        Get an interaction by ID.

        Args:
            interaction_id: ID of the interaction

        Returns:
            Interaction object or None if not found
        """
        with self.get_db_session() as session:
            interaction = (
                session.query(Interaction)
                .filter(Interaction.id == interaction_id)
                .first()
            )
            return interaction

    def get_session_interactions(self, session_id: str) -> List[Interaction]:
        """
        Get all interactions for a session.

        Args:
            session_id: ID of the session

        Returns:
            List of Interaction objects
        """
        with self.get_db_session() as session:
            interactions = (
                session.query(Interaction)
                .filter(Interaction.session_id == session_id)
                .order_by(Interaction.created_at)
                .all()
            )
            return interactions

    def update_interaction(
        self, interaction_id: str, **kwargs
    ) -> Optional[Interaction]:
        """
        Update an interaction.

        Args:
            interaction_id: ID of the interaction
            **kwargs: Fields to update

        Returns:
            Updated Interaction object or None if not found
        """
        with self.get_db_session() as session:
            interaction = (
                session.query(Interaction)
                .filter(Interaction.id == interaction_id)
                .first()
            )
            if not interaction:
                logger.warning(f"Interaction not found: {interaction_id}")
                return None

            for key, value in kwargs.items():
                if hasattr(interaction, key):
                    setattr(interaction, key, value)

            session.commit()
            session.refresh(interaction)
            logger.info(f"Updated interaction: {interaction.id}")
            return interaction

    def delete_interaction(self, interaction_id: str) -> bool:
        """
        Delete an interaction.

        Args:
            interaction_id: ID of the interaction

        Returns:
            True if deleted, False if not found
        """
        with self.get_db_session() as session:
            interaction = (
                session.query(Interaction)
                .filter(Interaction.id == interaction_id)
                .first()
            )
            if not interaction:
                logger.warning(f"Interaction not found: {interaction_id}")
                return False

            session.delete(interaction)
            session.commit()
            logger.info(f"Deleted interaction: {interaction_id}")
            return True

    # Feedback CRUD operations
    def create_feedback(
        self, interaction_id: str, user_id: str, rating: int, feedback_text: str = None
    ) -> Optional[Feedback]:
        """
        Create feedback for an interaction.

        Args:
            interaction_id: ID of the interaction
            user_id: ID of the user
            rating: Numeric rating (1-5)
            feedback_text: Optional text feedback

        Returns:
            Feedback object or None if interaction not found
        """
        with self.get_db_session() as session:
            # Verify interaction exists
            interaction = (
                session.query(Interaction)
                .filter(Interaction.id == interaction_id)
                .first()
            )
            if not interaction:
                logger.warning(f"Interaction not found: {interaction_id}")
                return None

            feedback = Feedback(
                interaction_id=interaction_id,
                user_id=user_id,
                rating=rating,
                feedback_text=feedback_text,
            )
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            logger.info(
                f"Stored feedback for interaction {interaction_id} from user {user_id}"
            )
            return feedback

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """
        Get feedback by ID.

        Args:
            feedback_id: ID of the feedback

        Returns:
            Feedback object or None if not found
        """
        with self.get_db_session() as session:
            feedback = (
                session.query(Feedback).filter(Feedback.id == feedback_id).first()
            )
            return feedback

    def get_interaction_feedback(self, interaction_id: str) -> List[Feedback]:
        """
        Get all feedback for an interaction.

        Args:
            interaction_id: ID of the interaction

        Returns:
            List of Feedback objects
        """
        with self.get_db_session() as session:
            feedback = (
                session.query(Feedback)
                .filter(Feedback.interaction_id == interaction_id)
                .all()
            )
            return feedback

    def update_feedback(self, feedback_id: str, **kwargs) -> Optional[Feedback]:
        """
        Update feedback.

        Args:
            feedback_id: ID of the feedback
            **kwargs: Fields to update

        Returns:
            Updated Feedback object or None if not found
        """
        with self.get_db_session() as session:
            feedback = (
                session.query(Feedback).filter(Feedback.id == feedback_id).first()
            )
            if not feedback:
                logger.warning(f"Feedback not found: {feedback_id}")
                return None

            for key, value in kwargs.items():
                if hasattr(feedback, key):
                    setattr(feedback, key, value)

            session.commit()
            session.refresh(feedback)
            logger.info(f"Updated feedback: {feedback.id}")
            return feedback

    def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete feedback.

        Args:
            feedback_id: ID of the feedback

        Returns:
            True if deleted, False if not found
        """
        with self.get_db_session() as session:
            feedback = (
                session.query(Feedback).filter(Feedback.id == feedback_id).first()
            )
            if not feedback:
                logger.warning(f"Feedback not found: {feedback_id}")
                return False

            session.delete(feedback)
            session.commit()
            logger.info(f"Deleted feedback: {feedback_id}")
            return True


# Example usage
if __name__ == "__main__":
    # Initialize the ORM
    db = DeepRecallORM()

    # Create tables if they don't exist
    db.create_tables()

    # Example: Create a user
    user = db.create_user(
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        preferences={"theme": "dark"},
        background="Test user for SQLAlchemy example",
        goals=["Test the ORM", "Implement CRUD operations"],
    )

    if user:
        print(f"Created user: {user.username} ({user.id})")

        # Example: Create a session
        session = db.create_session(
            user_id=str(user.id),
            name="Test Session",
            topic="SQLAlchemy Testing",
            metadata={"source": "example code"},
        )

        if session:
            print(f"Created session: {session.name} ({session.id})")

            # Example: Store a memory
            memory = db.create_memory(
                user_id=str(user.id),
                text="This is a test memory created by the sqlalchemy_db_utils.py script.",
                importance=0.6,
                category="test",
                session_id=str(session.id),
                metadata={"tags": ["test", "example", "sqlalchemy"]},
            )

            if memory:
                print(f"Created memory: {memory.id}")

                # Example: Store an interaction
                interaction = db.create_interaction(
                    session_id=str(session.id),
                    user_id=str(user.id),
                    prompt="Test prompt",
                    response="Test response",
                    memory_ids=[str(memory.id)],
                    model_used="test-model",
                    metadata={"test": True},
                )

                if interaction:
                    print(f"Created interaction: {interaction.id}")

                    # Example: Add feedback
                    feedback = db.create_feedback(
                        interaction_id=str(interaction.id),
                        user_id=str(user.id),
                        rating=5,
                        feedback_text="Great response!",
                    )

                    if feedback:
                        print(f"Created feedback: {feedback.id}")
