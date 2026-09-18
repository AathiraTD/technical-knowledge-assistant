"""Select storage once at application entry points."""
import os
import sqlite3


def open_repository(db="data/index/knowledge.db", dsn=None, *,
                    apply_schema=True, thread_safe=False):
    """The one place that decides which store an entry point talks to.

    Selection is one variable and one rule: `ASSISTANT_POSTGRES_DSN` unset or
    empty means SQLite at `db`, anything else means PostgreSQL + pgvector.
    There is deliberately no second switch — no `STORAGE_BACKEND` naming a
    backend that the DSN then has to agree with, because two settings that can
    disagree are a way to select a backend nobody asked for.

    Unset is the local/interview default, and it is a default rather than a
    fallback: nothing here probes for a database and quietly gives up. The
    local path needs no PostgreSQL, no pgvector and no server running, and
    `assistant/infrastructure/health.py` only checks a database when this same variable
    selects one.

    Conversation state is a separate decision on a separate variable
    (`ASSISTANT_CHECKPOINT_DSN`, read in `assistant/answering/engine.py`), so selecting
    PostgreSQL for the knowledge store does not imply a durable checkpointer —
    which matters, because the PostgreSQL checkpointer is dependency-blocked
    and raises. See `assistant/turn/graph.py:checkpointer_for`.

    `thread_safe` is for callers that serve more than one request at a time.
    It is off by default because the cost is real — every call serialises — and
    a CLI or an indexer has nothing to serialise. The threaded web server turns
    it on; see assistant/knowledge/store/locking.py for why a lock rather than a pool.
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


def open_persisted_session_store(db="data/index/knowledge.db", dsn=None):
    """Create a session store backed by the same database as the knowledge store.

    Opens a separate connection to SQLite or PostgreSQL for session persistence,
    independent of the knowledge repository. Sessions survive server restarts
    and can be shared across instances.

    Returns a PersistedSessionStore if a connection can be established,
    otherwise a regular (in-memory) SessionStore.
    """
    from ...turn.session import SessionStore
    from ...turn.session_storage import PersistedSessionStore

    connection_string = os.environ.get("ASSISTANT_POSTGRES_DSN", "") if dsn is None else dsn

    try:
        if connection_string:
            # PostgreSQL connection
            import psycopg
            connection = psycopg.connect(connection_string)
            return PersistedSessionStore(SessionStore(), connection, is_postgres=True)
        else:
            # SQLite connection
            connection = sqlite3.connect(db, check_same_thread=False)
            return PersistedSessionStore(SessionStore(), connection, is_postgres=False)
    except Exception:
        # If connection fails, fall back to in-memory store
        return SessionStore()
