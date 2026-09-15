"""The two adapters behind `KnowledgeRepository`.

`SQLiteKnowledgeRepository` ships with the submission and runs offline on the standard
library. `PostgresRepository` is the deployment target. They implement the same
Protocol against the same eight tables, so the engine cannot tell them apart —
which is what makes the demonstration path and the production path one system.
"""

from .embedded import EmbeddedRepository, SQLiteKnowledgeRepository

__all__ = ["SQLiteKnowledgeRepository", "EmbeddedRepository"]
