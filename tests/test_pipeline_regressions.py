"""Knowledge lifecycle failures reproduced with a disposable corpus and real SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from assistant import index
from assistant.embedcache import EmbeddingCache
from assistant.extract import Extracted
from assistant.store import SQLiteKnowledgeRepository


URL = "https://example.test/products/plaster"


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Use real extraction, caching, indexing and storage; replace only the model."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(index, "ROOT", tmp_path)
    monkeypatch.setattr(index, "CACHE", cache_dir)
    monkeypatch.setattr(index, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(index, "EmbeddingCache",
                        lambda: EmbeddingCache(tmp_path / "embeddings.db"))
    monkeypatch.setattr(index.ollama, "EMBED_DIMENSIONS", 4)
    monkeypatch.setattr(index.ollama, "require", lambda *models: None)
    monkeypatch.setattr(index.ollama, "embed", lambda texts, progress=None: [
        [1.0] + [0.0] * (index.ollama.EMBED_DIMENSIONS - 1) for text in texts
    ])

    def stage(water="5 to 6", *, failed=False):
        html = (
            "<html><head><title>Plaster</title></head><body><main>"
            "<h1>Plaster</h1><h2>Mixing</h2>"
            f"<p>Add {water} litres of water to each 25 kg bag. "
            "Mix thoroughly until the plaster has a smooth consistency. "
            "Use clean tools and apply to a suitably prepared background.</p>"
            "</main></body></html>"
        )
        source = cache_dir / "plaster.html"
        source.write_text(html, encoding="utf-8")
        digest = "sha256:" + hashlib.sha256(html.encode()).hexdigest()
        entry = {"url": URL, "path": str(source), "kind": "page",
                 "doc_type": "product_page", "title": "Plaster"}
        log = {"fetched": [] if failed else [entry], "skipped": [],
               "errors": [{"url": URL, "status": 503}] if failed else []}
        (cache_dir / "crawl-log.json").write_text(json.dumps(log), encoding="utf-8")
        (cache_dir / "versions.json").write_text(json.dumps({
            URL: {"content_hash": digest, "version": 1,
                  "fetched_at": "2026-01-01T00:00:00+00:00"}
        }), encoding="utf-8")

    with SQLiteKnowledgeRepository(tmp_path / "knowledge.db") as repo:
        yield stage, repo


def served_text(repo):
    query = [1.0] + [0.0] * (index.ollama.EMBED_DIMENSIONS - 1)
    return " ".join(hit.chunk.content for hit in repo.retrieve(query, top_k=50))


def test_separate_reader_refreshes_vectors_after_publication(pipeline, tmp_path):
    stage, writer = pipeline
    stage()
    index.build(repo=writer, verbose=False)
    with SQLiteKnowledgeRepository(tmp_path / "knowledge.db") as reader:
        assert "5 to 6 litres" in served_text(reader)
        stage(water="7 to 8")
        index.build(repo=writer, verbose=False)
        assert reader.active_version(URL).version == 2
        current = served_text(reader)
        assert "7 to 8 litres" in current
        assert "5 to 6 litres" not in current


def test_transient_crawl_error_preserves_last_published_document(pipeline):
    stage, repo = pipeline
    stage()
    index.build(repo=repo, verbose=False)
    previous_hash = repo.active_content_hashes()[URL]

    stage(failed=True)
    report = index.build(repo=repo, verbose=False)

    assert repo.active_content_hashes().get(URL) == previous_hash
    assert "5 to 6 litres" in served_text(repo)
    assert report["delta"]["removed"] == 0
    assert report["delta"]["failed"] == 1


@pytest.mark.parametrize("quality", ["failed", "flat"])
def test_unusable_extraction_cannot_supersede_published_evidence(
    pipeline, monkeypatch, quality
):
    stage, repo = pipeline
    stage()
    index.build(repo=repo, verbose=False)
    previous_hash = repo.active_content_hashes()[URL]
    stage(water="7 to 8")
    monkeypatch.setattr(index, "extract_html", lambda *args: (
        Extracted(quality=quality, note="No readable content"), {}
    ))

    report = index.build(repo=repo, verbose=False)

    assert repo.active_content_hashes()[URL] == previous_hash
    assert len(repo.versions(URL)) == 1
    assert "5 to 6 litres" in served_text(repo)
    assert report["delta"]["failed"] == 1


def test_failed_rebuild_rolls_back_to_the_complete_previous_index(pipeline):
    stage, repo = pipeline
    stage()
    index.build(repo=repo, verbose=False)
    previous = repo.snapshot()
    previous_hashes = repo.active_content_hashes()
    # A storage fault must roll back the whole replacement, including any clear.
    repo.db.execute("""CREATE TRIGGER reject_new_chunks BEFORE INSERT ON chunks
                       BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END""")
    repo.db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="simulated storage failure"):
        index.build(repo=repo, verbose=False, rebuild=True)

    assert repo.active_content_hashes() == previous_hashes
    assert repo.snapshot() == previous
    assert "5 to 6 litres" in served_text(repo)


@pytest.mark.parametrize("setting", ["embedding_model", "embedding_dimensions",
                                     "chunking_version"])
def test_processing_configuration_change_reprocesses_unchanged_sources(
    pipeline, monkeypatch, setting
):
    stage, repo = pipeline
    stage()
    index.build(repo=repo, verbose=False)
    original_hash = repo.active_content_hashes()[URL]
    if setting == "embedding_model":
        monkeypatch.setattr(index.ollama, "EMBED_MODEL", "replacement-embedding-model")
    elif setting == "embedding_dimensions":
        monkeypatch.setattr(index.ollama, "EMBED_DIMENSIONS", 6)
    else:
        monkeypatch.setattr(index, "CHUNKING_VERSION", "structure-aware/next")

    report = index.build(repo=repo, verbose=False)

    assert report["delta"]["reprocessed"] == 1
    assert repo.active_content_hashes()[URL] == original_hash
    assert len(repo.versions(URL)) == 2, "reprocessing must preserve source history"
    assert "5 to 6 litres" in served_text(repo)
