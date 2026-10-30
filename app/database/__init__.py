"""Database package for FaceCheck enterprise PostgreSQL data store."""

from __future__ import annotations

from typing import Optional
from app.database.connection import DatabaseConnectionManager
from app.database.models import HumanReviewFeedback, LearningStats, StoredImageRecord
from app.database.repository import PostgresLearningStore

# Alias PostgresLearningStore as LearningStore for standard access
LearningStore = PostgresLearningStore

_store_instance: Optional[PostgresLearningStore] = None


def get_learning_store() -> PostgresLearningStore:
    """Retrieve or initialize the global PostgreSQL LearningStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = PostgresLearningStore()
    return _store_instance


__all__ = [
    "DatabaseConnectionManager",
    "HumanReviewFeedback",
    "LearningStats",
    "StoredImageRecord",
    "PostgresLearningStore",
    "LearningStore",
    "get_learning_store",
]
