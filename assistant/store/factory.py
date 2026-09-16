"""Select storage once at application entry points."""
import os


def open_repository(db="data/index/knowledge.db", dsn=None, *,
                    apply_schema=True, thread_safe=False):
    """The one place that decides which store an entry point talks to.

    `thread_safe` is for callers that serve more than one request at a time.
    It is off by default because the cost is real — every call serialises — and
    a CLI or an indexer has nothing to serialise. The threaded web server turns
    it on; see assistant/store/locking.py for why a lock rather than a pool.
    """
    connection = os.environ.get("ASSISTANT_POSTGRES_DSN", "") if dsn is None else dsn
    if connection:
        from .postgres import PostgresKnowledgeRepository
        repository = PostgresKnowledgeRepository(connection, apply_schema=apply_schema)
    else:
        from . import SQLiteKnowledgeRepository
        repository = SQLiteKnowledgeRepository(
            db, check_same_thread=not thread_safe)

    if thread_safe:
        from .locking import LockedRepository
        return LockedRepository(repository)
    return repository
