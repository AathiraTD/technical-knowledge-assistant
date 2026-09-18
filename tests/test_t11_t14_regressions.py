"""Regressions for T11 (multi-source) and T14 (comparison) acceptance failures."""

import csv
from pathlib import Path
import pytest

from assistant.answering.engine import Assistant
from assistant.model import Snapshot
from assistant.store import SQLiteKnowledgeRepository
from assistant.infrastructure import ollama
from test_engine import CHUNKING_VERSION, chunk, document, unit, version


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "eval" / "evalset" / "Set 1" / "lime_green_ui_acceptance_tests.csv").open(
        encoding="utf-8-sig", newline="") as stream:
    CASES = {row["test_id"]: row["question"] for row in csv.DictReader(stream)}


def forbidden(*_args, **_kwargs):
    raise AssertionError("this boundary must not call")


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    """Fixture with Ultra, Solo, Duro for T11 and T14."""
    entries = [
        ("Ultra", "Description", "Ultra is suitable for most masonry and lath backgrounds."),
        ("Ultra", "Preparation", "Removing dust, surface contaminants and loose or friable coatings is essential preparation."),
        ("Ultra", "Application", "Ultra should be applied to a straight, flat, dampened surface in a uniform thickness of between 10 and 30 mm."),
        ("Ultra", "Mixing", "Mix with approximately 4 to 4.5 litres of clean water per bag."),
        ("Ultra", "Coverage", "Ultra covers approximately 1.5 m2 at 10 mm thickness."),
        ("Ultra", "Conditions", "Only use Ultra above 5 degrees C and below 30 degrees C."),
        ("Ultra", "Finishing Coats", "The finish coat should be 3 to 6 mm thick."),
        ("Solo", "Description", "Solo is suitable for lath backgrounds."),
        ("Solo", "Application", "Apply Solo at a thickness of 3 to 6 mm."),
        ("Solo", "Mixing", "Mix Solo with 5 to 6 litres of water per bag."),
        ("Solo", "Suitability", "Solo is suitable for most masonry and lath backgrounds."),
        ("Solo", "Backgrounds", "Solo is suitable for most masonry and lath backgrounds."),
        ("Duro", "Mixing", "Mix Duro with 9 litres of water per bag."),
        ("Bond", "Preparation", "Prime Ultra with Bond."),
    ]
    docs, versions, chunks, hits = [], [], [], []
    for index, (product, section, text) in enumerate(entries):
        url = f"https://example.invalid/{product.lower()}/{index}"
        doc = document(url, product)
        passage = chunk(url, 0, section, text, product, 0)
        docs.append(doc)
        versions.append(version(url))
        chunks.append(passage)
        hits.append((passage, doc, 0.71))

    snapshot = Snapshot(
        snapshot_id="regression-test", created_at="2026-09-18T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL, embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=len(docs),
        chunk_count=len(chunks), notes={"products": ["Ultra", "Solo", "Duro", "Bond"]})

    repo = SQLiteKnowledgeRepository(tmp_path / "knowledge.db")
    from assistant.model import Retrieved
    repo.publish(docs, versions, chunks, snapshot, [])
    app = Assistant(repo, log=False)

    # Return appropriate hits for each search
    def mock_search(*_args, **_kwargs):
        return [Retrieved(chunk=h[0], document=h[1], score=h[2]) for h in hits]

    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", forbidden)
    monkeypatch.setattr(app.retriever, "search", mock_search)

    try:
        yield app
    finally:
        repo.close()


def test_t11_multisource_synthesis(assistant):
    """T11: Multi-source synthesis should compose without asking for substrate."""
    # "For Lime Green Ultra, tell me the preparation required, application thickness,
    # mixing water and curing or application conditions."
    answer = assistant.answer_part(CASES["T11"])

    assert not answer.refused, f"T11 should not refuse, but got: {answer.text}"
    assert "10 and 30 mm" in answer.text or "10-30" in answer.text or "10–30" in answer.text
    assert "4 to 4.5 litres" in answer.text or "4-4.5" in answer.text
    assert "removing dust" in answer.text.lower()
    # Should have multiple sources
    assert len(answer.sources) >= 2


def test_t14_product_comparison(assistant):
    """T14: Comparison of Ultra and Solo should not ask for substrate."""
    # "Compare Lime Green Ultra and Solo for suitable backgrounds and application thickness."
    answer = assistant.answer_part(CASES["T14"], carried={"product": "Duro"})

    # Should not be refused
    assert not answer.refused, f"T14 should not refuse, but got: {answer.text}"

    # Should mention both products clearly
    text_lower = answer.text.lower()
    assert ("ultra" in text_lower and "solo" in text_lower), \
        f"T14 should compare both products, but got: {answer.text}"

    # Should contain their respective thickness ranges
    assert ("10 and 30 mm" in answer.text or "10-30" in answer.text or "10–30" in answer.text), \
        f"T14 should contain Ultra thickness, but got: {answer.text}"
    assert ("3 to 6 mm" in answer.text or "3-6" in answer.text or "3–6" in answer.text), \
        f"T14 should contain Solo thickness, but got: {answer.text}"

    # Should not use the carried Duro product
    assert "duro" not in text_lower or "duro" not in answer.text.split(), \
        f"T14 should ignore carried Duro, but got: {answer.text}"
