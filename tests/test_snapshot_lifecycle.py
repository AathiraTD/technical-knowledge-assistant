"""A request must read one release while the next release is published."""
from dataclasses import replace

import pytest

from assistant.knowledge.model import DocumentUpdate
from assistant.knowledge.store import SQLiteKnowledgeRepository
from test_repository_contract import A, doc, ver, chunk, snap


def test_read_snapshot_pins_passages_and_caveats_until_request_ends(tmp_path):
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
    with SQLiteKnowledgeRepository(tmp_path / "read.db") as repo:
        with pytest.raises(ValueError):
            with repo.read_snapshot():
                raise ValueError("cancelled")
        assert not repo.db.in_transaction


def test_snapshot_records_membership_and_complete_crawl_outcome(repo):
    from assistant.knowledge.model import CrawlRun
    repo.apply_delta([DocumentUpdate(doc("one"), ver("one"), [chunk("one", 0, "text", A)])], [],
                     snap("release"), crawl_run=CrawlRun("2026-01-01", documents_removed=2, snapshot_id="release"))
    assert repo.snapshot().notes["active_versions"]["one"]["version"] == 1
    assert repo.snapshot().notes["active_versions"]["one"]["content_hash"] == "h1"
    assert repo.crawl_runs()[0].documents_removed == 2
    assert repo.crawl_runs()[0].snapshot_id == "release"
