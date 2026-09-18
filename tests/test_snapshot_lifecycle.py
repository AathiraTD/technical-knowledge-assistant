"""One answer reads one release, even while the next release is being published.

Decision 19 pins a single read transaction per answer, and the reason is not
performance. An answer is assembled from several reads — passages, then
caveats, then the snapshot header printed beside it — and an indexer that
publishes between two of those reads would produce a reply whose citations and
whose header describe different states of the corpus. The retrieval rule that
active and superseded versions never blend would hold on each individual query
and still be violated by the answer as a whole.

So `read_snapshot()` is a context manager over a real database transaction, and
these tests exercise it with **two live connections to the same store**: a
reader inside the context, a writer publishing a new version underneath it. The
reader must continue to see `before` — both its snapshot id and its passage
text — and must see `after` only once the context has closed. The second test
covers the path that would quietly break the first: an exception inside the
context has to end the transaction rather than leave it open, or the connection
serves that release forever.

The third test is about what a snapshot has to carry to be auditable at all.
Publication records per-document membership — version number and content hash
in `notes["active_versions"]` — and the crawl outcome that produced it,
including the removals, which are the one category with no surviving row to
point at. A crawl whose counts were written outside the publishing transaction
could disagree with the release it describes.

These are lifecycle tests, not concurrency stress tests: they prove the
isolation boundary exists and is scoped correctly, not that it holds under
load. Contention between competing publishers is covered by the publication
lock tests.
"""
from dataclasses import replace

import pytest

from assistant.knowledge.model import DocumentUpdate
from assistant.knowledge.store import SQLiteKnowledgeRepository
from test_repository_contract import A, doc, ver, chunk, snap


def test_read_snapshot_pins_passages_and_caveats_until_request_ends(tmp_path):
    """A reader holding the context keeps `before` until it exits, then sees `after`."""
    path = tmp_path / "read.db"
    with SQLiteKnowledgeRepository(path) as writer, SQLiteKnowledgeRepository(path) as reader:
        first = DocumentUpdate(doc("one"), ver("one"), [chunk("one", 0, "old", A)])
        writer.apply_delta([first], [], snap("before"))
        with reader.read_snapshot() as release:
            assert release.snapshot_id == "before"
            assert reader.retrieve(A)[0].chunk.content == "old"
            writer.apply_delta([replace(first, chunks=[chunk("one", 0, "new", A)])], [], snap("after"))
            assert reader.snapshot().snapshot_id == "before"
            assert reader.retrieve(A)[0].chunk.content == "old"
        assert reader.snapshot().snapshot_id == "after"
        assert reader.retrieve(A)[0].chunk.content == "new"


def test_read_snapshot_releases_transaction_on_error(tmp_path):
    """An exception inside the pinned read must not strand an open transaction."""
    with SQLiteKnowledgeRepository(tmp_path / "read.db") as repo:
        with pytest.raises(ValueError):
            with repo.read_snapshot():
                raise ValueError("cancelled")
        assert not repo.db.in_transaction


def test_snapshot_records_membership_and_complete_crawl_outcome(repo):
    """Version, content hash, removals and snapshot id are written inside publication."""
    from assistant.knowledge.model import CrawlRun
    repo.apply_delta([DocumentUpdate(doc("one"), ver("one"), [chunk("one", 0, "text", A)])], [],
                     snap("release"), crawl_run=CrawlRun("2026-01-01", documents_removed=2, snapshot_id="release"))
    assert repo.snapshot().notes["active_versions"]["one"]["version"] == 1
    assert repo.snapshot().notes["active_versions"]["one"]["content_hash"] == "h1"
    assert repo.crawl_runs()[0].documents_removed == 2
    assert repo.crawl_runs()[0].snapshot_id == "release"
