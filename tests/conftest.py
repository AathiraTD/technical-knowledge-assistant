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


def _sqlite():
    return SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "contract.db")


def _postgres():
    dsn = os.environ.get("ASSISTANT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("ASSISTANT_POSTGRES_DSN is not set; PostgreSQL contract not run")
    try:
        from assistant.store.postgres import PostgresKnowledgeRepository
    except ImportError as exc:
        pytest.skip(f"psycopg unavailable: {exc}")
    try:
        return PostgresKnowledgeRepository(dsn)
    except Exception as exc:                     # unreachable server
        pytest.skip(f"PostgreSQL unreachable: {exc}")


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
