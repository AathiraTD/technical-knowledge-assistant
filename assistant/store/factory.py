"""Select storage once at application entry points."""
import os

def open_repository(db="data/index/knowledge.db", dsn=None, *, apply_schema=True):
    connection = os.environ.get("ASSISTANT_POSTGRES_DSN", "") if dsn is None else dsn
    if connection:
        from .postgres import PostgresKnowledgeRepository
        return PostgresKnowledgeRepository(connection, apply_schema=apply_schema)
    from . import SQLiteKnowledgeRepository
    return SQLiteKnowledgeRepository(db)
