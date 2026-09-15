"""The four-run proof that ingestion is incremental rather than a rebuild.

This is the test the architecture's claim rests on. Everything else shows that
a snapshot can be built; this shows that a second crawl of an unchanged site
does no work, that a changed datasheet keeps the version it replaced, and that a
withdrawn one stops being quoted without being erased.

    run 1   empty store          -> 3 new, everything processed
    run 2   nothing changed      -> 0 reprocessed, no new versions
    run 3   one datasheet edited -> 1 changed, v1 retained, only v2 served
    run 4   one document removed -> deactivated, history kept, absent from answers

It runs against a small staged corpus rather than the real 94 documents, so it
is fast and its arithmetic is checkable by eye. Ollama is replaced: what is
under test is the lifecycle, not the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import index                                    # noqa: E402
from assistant.embedcache import EmbeddingCache                # noqa: E402
from assistant.store import SQLiteKnowledgeRepository          # noqa: E402

PAGE = """<!doctype html><html><head><title>{title} | Lime Green</title></head>
<body><main>
<h1>{title}</h1>
<h2>Description</h2><p>{body}</p>
<h2>Mixing</h2><p>Add {water} litres of clean water per 25kg sack and mix until
a smooth creamy consistency is reached. Do not apply below 5 degrees C.</p>
<h2>Coverage</h2><p>Approximately {coverage} m2 per 25 kg sack at 10 mm.</p>
</main></body></html>"""


def a_page(title: str, body: str, water: str = "5 to 6",
           coverage: str = "2") -> str:
    return PAGE.format(title=title, body=body, water=water, coverage=coverage)


def stage(tmp_path: Path, pages: dict[str, str]) -> Path:
    """Write a cache directory shaped exactly like the real one."""
    cache = tmp_path / "cache"
    (cache / "pages").mkdir(parents=True, exist_ok=True)
    fetched, ledger = [], {}
    for slug, html in pages.items():
        path = cache / "pages" / f"{slug}.html"
        path.write_text(html, encoding="utf-8")
        url = f"https://example.test/products/{slug}"
        digest = index.hashlib.sha256(path.read_bytes()).hexdigest()
        fetched.append({
            "url": url, "path": str(path).replace("\\", "/"), "kind": "page",
            "doc_type": "product_page", "title": f"{slug} | Lime Green",
            "link_text": "", "product": slug, "status": 200,
            "content_hash": f"sha256:{digest}", "version": 1,
        })
        ledger[url] = {"version": 1, "content_hash": f"sha256:{digest}",
                       "fetched_at": "2026-01-01T00:00:00+00:00",
                       "first_seen_at": "2026-01-01T00:00:00+00:00",
                       "checked_at": "2026-01-01T00:00:00+00:00",
                       "is_active": True}
    (cache / "crawl-log.json").write_text(json.dumps(
        {"site": "https://example.test", "fetched": fetched,
         "skipped": [{"url": "https://example.test/sds.pdf",
                      "reason": "safety data sheets are read whole"}]}),
        encoding="utf-8")
    (cache / "versions.json").write_text(json.dumps(ledger), encoding="utf-8")
    return cache


def fake_vector(text: str) -> list[float]:
    """Deterministic and cheap. Retrieval quality is not what this file tests."""
    v = [0.0] * index.ollama.EMBED_DIMENSIONS
    for i, ch in enumerate(text[:64]):
        v[ord(ch) % index.ollama.EMBED_DIMENSIONS] += 1.0 + i * 0.001
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A real repository and a real indexer, with only Ollama replaced."""
    monkeypatch.setattr(index, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(index, "EmbeddingCache",
                        lambda: EmbeddingCache(tmp_path / "embeddings.db"))
    monkeypatch.setattr(index.ollama, "require", lambda *m: None)
    monkeypatch.setattr(index.ollama, "embed",
                        lambda texts, progress=None: [fake_vector(t) for t in texts])
    # No fixtures: this file is about the delta, and the staff fixture has its
    # own coverage elsewhere.
    monkeypatch.setattr(index, "ROOT", tmp_path)
    (tmp_path / "eval" / "fixtures").mkdir(parents=True, exist_ok=True)

    repo = SQLiteKnowledgeRepository(tmp_path / "knowledge.db")

    def run(pages: dict[str, str]) -> dict:
        monkeypatch.setattr(index, "CACHE", stage(tmp_path, pages))
        return index.build(repo=repo, verbose=False)

    yield run, repo
    repo.close()


THREE = {
    "solo": a_page("Solo Onecoat", "A one-coat lime plaster for interior use."),
    "duro": a_page("Duro Basecoat", "A general purpose lime undercoat plaster.",
                   water="4.5 to 5", coverage="2.5"),
    "forte": a_page("Forte Basecoat", "A render base coat for masonry.",
                    water="4 to 5", coverage="1.8"),
}

SOLO_URL = "https://example.test/products/solo"
FORTE_URL = "https://example.test/products/forte"


# ------------------------------------------------------------------- run 1


def test_run_one_indexes_everything_because_nothing_is_indexed_yet(harness):
    """A first build is a delta against an empty store, not a separate code path."""
    run, repo = harness
    report = run(THREE)

    assert report["delta"]["new"] == 3
    assert report["delta"]["changed"] == 0
    assert report["delta"]["unchanged"] == 0
    assert report["delta"]["reprocessed"] == 3
    assert report["documents"] == 3
    assert report["chunks"] > 0
    assert len(repo.active_content_hashes()) == 3


# ------------------------------------------------------------------- run 2


def test_run_two_reprocesses_nothing_when_the_site_has_not_changed(harness):
    """This is the whole claim. If anything is reprocessed here, it is a rebuild."""
    run, repo = harness
    first = run(THREE)
    second = run(THREE)

    assert second["delta"]["unchanged"] == 3
    assert second["delta"]["new"] == 0
    assert second["delta"]["changed"] == 0
    assert second["delta"]["reprocessed"] == 0
    assert second["embeddings_computed"] == 0
    assert second["embeddings_cached"] == 0, "nothing should have been embedded at all"

    # No new versions, and the same passages still served.
    for url in repo.active_content_hashes():
        assert len(repo.versions(url)) == 1
    assert second["chunks"] == first["chunks"]
    assert second["documents"] == first["documents"]


def test_run_two_still_records_a_crawl_run(harness):
    """'94 unchanged, nothing reprocessed' is the evidence, and it has to persist."""
    run, repo = harness
    run(THREE)
    run(THREE)

    runs = repo.crawl_runs()
    assert len(runs) == 2
    latest = runs[0]
    assert latest.documents_unchanged == 3
    assert latest.reprocessed == 0
    assert latest.documents_checked == 3


# ------------------------------------------------------------------- run 3


def test_run_three_creates_a_new_version_and_keeps_the_old_one(harness):
    """A changed datasheet must not take its own history with it."""
    run, repo = harness
    run(THREE)

    edited = dict(THREE)
    edited["solo"] = a_page("Solo Onecoat",
                            "A one-coat lime plaster for interior use.",
                            water="6 to 7", coverage="1.8")
    report = run(edited)

    assert report["delta"]["changed"] == 1
    assert report["delta"]["unchanged"] == 2
    assert report["delta"]["reprocessed"] == 1

    versions = repo.versions(SOLO_URL)
    assert len(versions) == 2, "the superseded version was discarded"
    assert versions[0].is_active is True
    assert versions[1].is_active is False

    # The untouched documents did not gain versions.
    assert len(repo.versions(FORTE_URL)) == 1


def test_run_three_serves_only_the_new_figures(harness):
    """Two coverage figures retrievable at once is the failure this prevents."""
    run, repo = harness
    run(THREE)
    edited = dict(THREE)
    edited["solo"] = a_page("Solo Onecoat",
                            "A one-coat lime plaster for interior use.",
                            water="6 to 7", coverage="1.8")
    run(edited)

    served = " ".join(
        h.chunk.content for h in repo.retrieve(fake_vector("Mixing water"), top_k=50)
    )
    assert "6 to 7 litres" in served
    assert "5 to 6 litres" not in served, "a superseded figure was still retrievable"


# ------------------------------------------------------------------- run 4


def test_run_four_deactivates_a_removed_document_without_erasing_it(harness):
    """A withdrawn sheet must stop being quoted, and must stay auditable."""
    run, repo = harness
    run(THREE)

    remaining = {k: v for k, v in THREE.items() if k != "forte"}
    report = run(remaining)

    assert report["delta"]["removed"] == 1
    assert report["delta"]["removed_urls"] == [FORTE_URL]
    assert report["documents"] == 2

    assert FORTE_URL not in repo.active_content_hashes()
    history = repo.versions(FORTE_URL)
    assert len(history) == 1, "history was deleted rather than deactivated"
    assert history[0].is_active is False
    assert repo.document(FORTE_URL) is not None, "the document row was deleted"

    served = " ".join(
        h.chunk.content for h in repo.retrieve(fake_vector("render base coat"),
                                               top_k=50))
    assert "render base coat for masonry" not in served.lower()


def test_a_document_that_returns_to_the_site_is_reactivated_as_a_new_version(harness):
    """Withdrawal is not deletion, so republication is a version rather than a rebirth."""
    run, repo = harness
    run(THREE)
    run({k: v for k, v in THREE.items() if k != "forte"})
    run(THREE)

    history = repo.versions(FORTE_URL)
    assert len(history) == 2
    assert history[0].is_active is True
    assert FORTE_URL in repo.active_content_hashes()


# ---------------------------------------------------------------- the escape hatch


def test_a_forced_rebuild_reprocesses_everything_even_when_nothing_changed(harness):
    """Chunking can change without any content hash moving, and then a delta sees nothing."""
    run, repo = harness
    run(THREE)

    import functools
    rebuild = functools.partial(index.build, repo=repo, verbose=False, rebuild=True)
    report = rebuild()

    assert report["delta"]["new"] == 3
    assert report["delta"]["unchanged"] == 0
    assert report["delta"]["reprocessed"] == 3
    assert report["documents"] == 3
    # Reprocessing must preserve the evidence used by previous answers.
    versions = repo.versions(SOLO_URL)
    assert len(versions) == 2
    assert versions[0].is_active and not versions[1].is_active


# --------------------------------------------------------------- the report


def test_the_report_names_every_document_and_what_happened_to_it(harness):
    """An assessor should be able to read which documents were skipped and why."""
    run, _repo = harness
    run(THREE)
    report = run(THREE)

    changes = {r["url"]: r["change"] for r in report["rows"]}
    assert len(changes) == 3
    assert set(changes.values()) == {"unchanged"}
    assert all("content hash unchanged" in r["note"]
               for r in report["rows"] if r["change"] == "unchanged")


def test_a_missing_cached_file_is_reported_rather_than_fatal(harness, tmp_path):
    """One unreadable file must not cost the other ninety-three."""
    run, repo = harness
    run(THREE)

    cache = stage(tmp_path, THREE)
    (cache / "pages" / "duro.html").unlink()
    index.CACHE = cache
    report = index.build(repo=repo, verbose=False)

    assert report["delta"]["failed"] == 1
    assert any(r["change"] == "failed" for r in report["rows"])
    assert report["documents"] == 3, "a failed read must not deactivate a document"
