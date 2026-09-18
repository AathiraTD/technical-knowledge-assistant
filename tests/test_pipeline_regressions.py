"""Every way a bad run could replace a good release, reproduced on a one-page corpus.

The claim decision 18 and decision 19 make is that a failed crawl or a failed
indexing attempt never corrupts what is currently being served. That claim is
about the *unhappy* paths, so it cannot be demonstrated by a build that works —
it has to be demonstrated by builds that break in each of the ways a real one
breaks: the source could not be fetched, the source was fetched but extracts to
nothing, storage rejects a write halfway through a rebuild, or the processing
configuration changed underneath an unchanged corpus.

The fixture is the design of this file. Extraction, chunking, the embedding
cache, the delta computation and a real `SQLiteKnowledgeRepository` all run for
real; the **only** thing replaced is the model, which returns a fixed
four-element vector. Substituting the repository with a fake would move the
tests off the invariant they exist for, because "the previous version is still
active" is a database fact. The corpus is a single hand-written page whose
mixing figure reads `5 to 6 litres` before and `7 to 8 litres` after, so every
assertion about what is served can be made by looking for a figure rather than
by counting rows.

`served_text` is what makes those assertions honest: it retrieves rather than
querying the tables directly, so a superseded version that remained reachable
would show up as text in an answer, which is the way a reader would actually
meet it.

Two details are worth pointing out because they are easy to write weaker. The
rollback test installs a SQLite trigger that aborts inserts into `chunks`
during a `--rebuild`, which reaches the one path where the store is cleared
before it is refilled; the assertion is that both the snapshot and the active
hashes are *identical* to before, not merely non-empty. And the configuration
tests assert `len(repo.versions(URL)) == 2` — reprocessing for a new embedding
model or chunking version must create an auditable new version of an unchanged
source, not quietly rewrite the old one.

This file does not test the crawler, the HTTP layer or answer quality. The
crawl log and version ledger it stages are files on disk, written by the
fixture, standing in for a crawl that has already happened.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from assistant.indexing import index
from assistant.indexing.embedcache import EmbeddingCache
from assistant.indexing.extract import Extracted
from assistant.knowledge.store import SQLiteKnowledgeRepository


URL = "https://example.test/products/plaster"


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Use real extraction, caching, indexing and storage; replace only the model.

    Yields `(stage, repo)`. Calling `stage()` writes the one-page corpus, its
    crawl log and its version ledger into a temporary cache; `stage(water=...)`
    changes the published figure so the next build sees a changed document, and
    `stage(failed=True)` writes a log in which the fetch errored instead.
    """
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
    """What a reader would actually be shown: retrieved passages, not raw rows."""
    query = [1.0] + [0.0] * (index.ollama.EMBED_DIMENSIONS - 1)
    return " ".join(hit.chunk.content for hit in repo.retrieve(query, top_k=50))


def test_separate_reader_refreshes_vectors_after_publication(pipeline, tmp_path):
    """A second connection picks up the new release and stops serving the old figure.

    The SQLite adapter holds vectors in memory, so a reader that never
    reloaded them would keep answering from the superseded version — visibly,
    and with a citation.
    """
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
    """A 503 is a failure, not a withdrawal: the document stays active and counts as failed."""
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
    """A changed source that extracts to nothing must not create a version.

    Both unusable qualities are covered — `failed` and `flat` — because a
    scanned PDF and a page whose text layer collapsed are the same hazard: a
    real change arrives, the bytes hash differently, and the replacement
    carries no evidence. The published version count stays at one.
    """
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
    """A storage fault during `--rebuild` rolls back the clear as well as the write.

    Rebuild is the only path that empties the store before refilling it, so it
    is the only path where a partial failure could leave nothing being served.
    """
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
    """A new model, dimension or chunking version reprocesses a source that did not change.

    Vectors from two embedding models in one index return confident nonsense,
    so the delta cannot be computed from the content hash alone. The source
    hash is unchanged and must stay unchanged; what changes is that a second,
    auditable version is published under the new configuration.
    """
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
