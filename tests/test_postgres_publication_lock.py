"""What happens to the second indexer while the first one is publishing.

`apply_delta` serialises publication on a PostgreSQL advisory lock. That is the
right mechanism — two indexers publishing into one release would interleave
versions — but `pg_advisory_xact_lock` waits forever, and forever is a bad
answer for a scheduled job. An indexer killed without its connection being
reaped holds the lock until the server notices the socket is gone, and every
later run then blocks with no output and nothing in the log to explain it. That
failure reads as "indexing is slow tonight", for as long as nobody looks.

So the wait is bounded and the timeout is turned into a named repository error.
These tests cover both halves: the bound fires under real contention, and a
failure that is *not* the timeout is still raised as itself rather than being
relabelled — because a mis-set server parameter is a fault to read, not a busy
indexer to retry.

PostgreSQL only, and deliberately not part of the shared contract suite: SQLite
reaches the same situation through its own busy timeout by a different route,
and a test reaching into `pg_locks` would have nothing to assert there.

**The lock is database-wide.** `pg_advisory_xact_lock(8675309)` is not scoped to
the temporary schema each test gets, so a blocker left holding it does not fail
one test — it fails every later test that publishes, each with a misleading
"another indexer is publishing". Every acquisition in this file therefore goes
through `held_lock`, which releases in a `finally`. Do not take the lock any
other way here.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.indexing.index import CHUNKING_VERSION                      # noqa: E402
from assistant.knowledge.model import (
    Chunk,
    Document,
    DocumentUpdate,
    DocumentVersion,
    Snapshot,
)
from assistant.knowledge.repository import PublicationBusy                  # noqa: E402

DSN = os.environ.get("ASSISTANT_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="ASSISTANT_POSTGRES_DSN is not set; PostgreSQL not run")

DIMS = 1024
URL = "https://example.invalid/solo"
LOCK = 8675309          # the publication lock, as `apply_delta` names it


def unit() -> list[float]:
    v = [0.0] * DIMS
    v[0] = 1.0
    return v


@pytest.fixture
def store():
    """A repository in a schema of its own, torn down afterwards."""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from assistant.knowledge.store.postgres import PostgresKnowledgeRepository

    admin = psycopg.connect(DSN, autocommit=True)
    schema = "tka_lock_" + uuid.uuid4().hex
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    repo = PostgresKnowledgeRepository(
        make_conninfo(DSN, options=f"-c search_path={schema},public"))
    try:
        yield repo
    finally:
        repo.close()
        admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        admin.close()


@contextlib.contextmanager
def held_lock():
    """Hold the publication lock on a connection of its own, and always let go.

    This is what a crashed indexer's abandoned transaction looks like to the
    server. The `finally` is the important part — see the module docstring.
    """
    import psycopg

    blocker = psycopg.connect(DSN, autocommit=False)
    try:
        blocker.execute(f"SELECT pg_advisory_xact_lock({LOCK})")
        yield blocker
    finally:
        blocker.rollback()          # releases the transaction-scoped lock
        blocker.close()


def waiters() -> int:
    """How many connections are queued on *our* publication lock right now.

    Every clause after `locktype` is load-bearing, and each was shown to matter
    by a waiter that the looser query counted wrongly:

      `classid = 0`    a bigint key is stored as classid 0 with the key in
                       objid; `pg_advisory_xact_lock(4303642605)` is a different
                       lock that shares our low 32 bits and was counted.
      `objsubid = 1`   distinguishes the one-argument bigint form from
                       `pg_advisory_xact_lock(0, 8675309)`, which is a different
                       lock again.
      `database = …`   `pg_locks` is cluster-wide. Without this, a second suite
                       running against another database on the same server makes
                       this one report contention it did not cause — the same
                       database-wide hazard the module docstring warns about,
                       arriving through the observability query instead.
    """
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as watcher:
        return watcher.execute(
            "SELECT count(*) FROM pg_locks "
            " WHERE locktype = 'advisory' AND classid = 0 AND objid = %s "
            "   AND objsubid = 1 AND NOT granted "
            "   AND database = (SELECT oid FROM pg_database "
            "                    WHERE datname = current_database())",
            (LOCK,)).fetchone()[0]


def a_delta() -> tuple[list[DocumentUpdate], Snapshot]:
    document = Document(canonical_url=URL, title="Solo datasheet",
                        document_type="datasheet", authority=1, product="Solo",
                        link_text="Solo Datasheet")
    version = DocumentVersion(canonical_url=URL, version=1, content_hash="h1",
                              source_path="cache/solo.pdf",
                              fetched_at="2026-01-01T00:00:00Z")
    chunk = Chunk(canonical_url=URL, version=1, chunk_index=0, section="Mixing",
                  content="Mix with 5-6 litres per 25 kg sack.", product="Solo",
                  document_type="datasheet", authority=1,
                  source_date="2024-07-01", embedding=unit())
    snapshot = Snapshot(snapshot_id="snap-lock", created_at="2026-01-01T00:00:00Z",
                        embedding_model="qwen3-embedding:0.6b",
                        embedding_dimensions=DIMS,
                        chunking_version=CHUNKING_VERSION,
                        document_count=1, chunk_count=1,
                        notes={"products": ["Solo"], "colours": [],
                               "merchants": [], "contact": {}})
    return [DocumentUpdate(document, version, [chunk], [])], snapshot


def test_a_publisher_waiting_on_a_held_lock_gives_up_with_a_diagnosis(
        store, monkeypatch):
    """The whole point: a hung publisher must fail loudly, not wait forever."""
    from assistant.knowledge.store import postgres

    # One second, not thirty: what is under test is that a bound exists and is
    # enforced, not the value chosen for production.
    monkeypatch.setattr(postgres, "PUBLISH_LOCK_TIMEOUT", 1)

    # Assert the patch took, rather than inferring it from the clock. This is
    # what an upper bound on the elapsed time was really for: if the monkeypatch
    # silently failed, the wait would be the unpatched 30 s and a ceiling would
    # notice. Checking the value directly notices the same thing and cannot be
    # confused by a slow host, which a timing assertion provably can be — a
    # paused container produced readings of 26 s and 51 s here, on a database
    # that behaved perfectly. Timing it on the server's clock does not help:
    # when the backend is frozen its `lock_timeout` timer freezes with it, so
    # the wait really was that long.
    assert postgres.PUBLISH_LOCK_TIMEOUT == 1, "the timeout patch did not apply"

    updates, snapshot = a_delta()
    with held_lock():
        started = time.perf_counter()
        with pytest.raises(PublicationBusy) as raised:
            store.apply_delta(updates, [], snapshot)
        waited = time.perf_counter() - started

    assert "still holds the publication lock" in str(raised.value)
    assert "Nothing was changed" in str(raised.value)
    # Only a lower bound, and only to prove it waited for the lock rather than
    # failing on something else instantly. It is server-enforced and has never
    # been observed below 1.001 s across more than a hundred runs. There is no
    # upper bound, because giving up at all is what this test is about and the
    # raised exception is the proof of it — an adapter that hung would never
    # reach this line.
    assert waited > 0.5, f"gave up after only {waited:.2f}s"

    # And nothing was published: the store is exactly as it was.
    assert store.snapshot() is None
    assert store.counts()["documents"] == 0


def test_the_release_of_the_lock_lets_the_next_publisher_through(store):
    """A bounded wait must still be a wait, or publication would never queue."""
    updates, snapshot = a_delta()
    outcome: dict[str, object] = {}

    def publish() -> None:
        try:
            outcome["id"] = store.apply_delta(updates, [], snapshot)
        except Exception as error:                      # pragma: no cover - diagnostic
            outcome["error"] = error

    publisher = threading.Thread(target=publish)
    with held_lock():
        publisher.start()
        # Wait for the publisher to actually queue on the lock rather than
        # sleeping and assuming it did. `assert not outcome` after a fixed sleep
        # passes just as well when the thread has not started yet, which would
        # make this test prove nothing on a loaded machine.
        deadline = time.perf_counter() + 15
        while waiters() < 1 and time.perf_counter() < deadline:
            time.sleep(0.05)

        assert waiters() == 1, "the publisher never queued on the lock"
        assert not outcome, "it published while the lock was held"

    publisher.join(timeout=30)
    assert not publisher.is_alive(), "the publisher never returned"
    assert "error" not in outcome, outcome.get("error")
    assert store.snapshot().snapshot_id == "snap-lock"


def test_a_failure_that_is_not_the_timeout_is_raised_as_itself(store):
    """Relabelling every error as 'busy' would hide the ones worth reading.

    The error is a real one from the server rather than a stubbed cursor: a
    DBA-imposed `statement_timeout` shorter than the publication wait cancels
    the very same statement, with sqlstate 57014 instead of 55P03. That is
    exactly what the sqlstate check must let through unrelabelled — a mis-set
    server parameter is a fault to read, not a busy indexer to retry.

    Stubbing was tried first and cannot work: `psycopg.Cursor` defines
    `__slots__`, so assigning to `cursor.execute` raises `AttributeError`
    before the adapter is reached, on every platform.
    """
    import psycopg
    from psycopg.conninfo import make_conninfo
    from assistant.knowledge.store.postgres import PostgresKnowledgeRepository

    # Spaces are stripped because libpq splits the `options` string on them: a
    # `search_path` echoed back as "schema, public" would silently truncate the
    # options that follow it, and `statement_timeout` would never be set — the
    # test would then pass or fail for a reason unrelated to what it checks.
    search_path = store.conn.execute("SHOW search_path").fetchone()[0]
    impatient = PostgresKnowledgeRepository(make_conninfo(
        DSN, options=f"-c search_path={search_path.replace(' ', '')} "
                     f"-c statement_timeout=400ms"))

    updates, snapshot = a_delta()
    try:
        with held_lock():
            with pytest.raises(psycopg.errors.QueryCanceled) as raised:
                impatient.apply_delta(updates, [], snapshot)

        assert raised.value.sqlstate == "57014"
        assert not isinstance(raised.value, PublicationBusy)
    finally:
        impatient.close()
