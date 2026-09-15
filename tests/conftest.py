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

from assistant.store import SQLiteKnowledgeRepository  # noqa: E402

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
    from assistant.store.postgres import PostgresKnowledgeRepository
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
