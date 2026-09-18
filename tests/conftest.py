"""Shared fixtures.

The repository contract is one behavioural suite that must run against every
adapter, so `repo` is parametrised over them rather than bound to one. SQLite
runs always; PostgreSQL runs when `ASSISTANT_POSTGRES_DSN` points at a
reachable pgvector instance and is skipped, visibly, when it does not — a
skipped adapter is reported as skipped rather than quietly passing.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.infrastructure import ollama
from assistant.indexing.index import CHUNKING_VERSION           # noqa: E402
from assistant.knowledge.model import Snapshot                   # noqa: E402
from assistant.knowledge.store import SQLiteKnowledgeRepository  # noqa: E402

TEST_POSTGRES_DSN = os.environ.get("ASSISTANT_POSTGRES_DSN")


@pytest.fixture(autouse=True)
def isolated_backend_environment(monkeypatch):
    # Contract tests use the explicitly configured disposable service. Other
    # tests must not silently switch their temporary SQLite stores to it.
    monkeypatch.delenv("ASSISTANT_POSTGRES_DSN", raising=False)


def _sqlite():
    return SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "contract.db")


def _postgres():
    dsn = TEST_POSTGRES_DSN
    if not dsn:
        pytest.skip("ASSISTANT_POSTGRES_DSN is not set; PostgreSQL contract not run")
    import uuid
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from assistant.knowledge.store.postgres import PostgresKnowledgeRepository
    admin = psycopg.connect(dsn, autocommit=True)
    schema = "tka_test_" + uuid.uuid4().hex
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    made = PostgresKnowledgeRepository(make_conninfo(dsn, options=f"-c search_path={schema},public"))
    close = made.close
    def cleanup():
        close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()
    made.close = cleanup
    return made


ADAPTERS = {"sqlite": _sqlite, "postgres": _postgres}


@pytest.fixture(params=list(ADAPTERS), ids=list(ADAPTERS))
def repo(request):
    """A fresh, empty repository for each test, one per adapter."""
    made = ADAPTERS[request.param]()
    try:
        yield made
    finally:
        if hasattr(made, "close"):
            made.close()


@pytest.fixture
def indexed_repo(repo):
    """`repo`, with an empty release published so an `Assistant` can be built.

    `repo` is deliberately empty, which is right for the storage tests that own
    it. It is wrong for any test that constructs an `Assistant`, because
    `Retriever._verify()` refuses to run against a store with no active
    snapshot -- correctly: decision 6 makes an index built by one embedding
    model and queried by another a refusal rather than a warning, and "no index
    at all" is the same class of fault.

    So this publishes a release with no documents in it. Retrieval returns
    nothing, which suits a test about slot detection or session state and would
    not suit a test about answers.
    """
    repo.publish([], [], [], Snapshot(
        snapshot_id="snap-empty", created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION,
        document_count=0, chunk_count=0, notes={}), [])
    return repo
