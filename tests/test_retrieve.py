"""Retrieval: the refusal that matters most, and the synonyms that close the gap.

The important test in this file is the mismatch refusal. An index built with
one embedding model and queried with another does not fail — it returns
plausible, confidently wrong passages, which is the exact failure the whole
design exists to avoid. So it is checked once, at construction, and raised
rather than warned about.

Runs without Ollama: the embedding call is replaced, because what is under test
is the gate around it rather than the model behind it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.index import CHUNKING_VERSION
from assistant import ollama                                  # noqa: E402
from assistant.model import Chunk, Document, DocumentVersion, Snapshot  # noqa: E402
from assistant.repository import IndexMismatch                # noqa: E402
from assistant.retrieve import (                              # noqa: E402
    DEFAULT_THRESHOLD,
    QUERY_INSTRUCTION,
    Retriever,
    as_query,
)
from assistant.store import SQLiteKnowledgeRepository         # noqa: E402

DIMS = ollama.EMBED_DIMENSIONS
URL = "https://example/solo"


def unit(*leading: float) -> list[float]:
    v = list(leading) + [0.0] * (DIMS - len(leading))
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


A = unit(1, 0, 0)
B = unit(0, 1, 0)


def build(model: str = ollama.EMBED_MODEL, dims: int = DIMS,
          chunks: bool = True) -> SQLiteKnowledgeRepository:
    repo = SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "r.db")
    rows = ([Chunk(canonical_url=URL, version=1, chunk_index=0, section="Mixing",
                   content="Add between 5 and 6 litres per 25kg sack.",
                   product="Solo", document_type="datasheet", authority=1,
                   embedding=A)] if chunks else [])
    repo.publish(
        [Document(canonical_url=URL, title="Solo", document_type="datasheet",
                  authority=1, product="Solo")],
        [DocumentVersion(canonical_url=URL, version=1, content_hash="h",
                         source_path="p", is_active=True)],
        rows,
        Snapshot(snapshot_id="s1", created_at="2026-01-01T00:00:00Z",
                 embedding_model=model, embedding_dimensions=dims,
                 chunking_version=CHUNKING_VERSION, document_count=1,
                 chunk_count=len(rows)),
    )
    return repo


# ------------------------------------------------------- the mismatch refusal


def test_an_index_built_with_another_model_is_refused():
    """Querying it would return plausible, confidently wrong passages rather than fail."""
    repo = build(model="some-other-embedding-model")
    with pytest.raises(IndexMismatch) as raised:
        Retriever(repo)
    assert "rebuild" in str(raised.value).lower()
    assert "some-other-embedding-model" in str(raised.value)


def test_an_index_of_the_wrong_width_is_refused():
    repo = build(dims=DIMS // 2)
    with pytest.raises(IndexMismatch):
        Retriever(repo)


def test_an_empty_store_names_the_command_that_fixes_it():
    """The most likely reader of this error is an assessor running it for the first time."""
    repo = SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "empty.db")
    with pytest.raises(IndexMismatch) as raised:
        Retriever(repo)
    assert "assistant.index" in str(raised.value)


def test_a_matching_index_is_accepted_and_its_snapshot_kept():
    r = Retriever(build())
    assert r.snapshot.embedding_model == ollama.EMBED_MODEL
    assert r.threshold == DEFAULT_THRESHOLD


# --------------------------------------------------------------- the question


def test_the_query_carries_the_instruction_prefix():
    """Qwen3-Embedding is trained asymmetrically; without this, worse passages come back."""
    assert as_query("how much water").startswith(QUERY_INSTRUCTION)
    assert as_query("how much water").endswith("how much water")


def test_a_customer_saying_bag_reaches_a_sheet_saying_sack():
    """The vocabulary gap is the entire reason retrieval is semantic rather than keyword."""
    expanded = Retriever(build()).expand("how many litres per bag")
    assert "sack" in expanded
    assert expanded.startswith("how many litres per bag")


def test_expansion_adds_nothing_when_no_synonym_applies():
    question = "what is the drying time"
    assert Retriever(build()).expand(question) == question


def test_a_synonym_already_present_is_not_repeated():
    expanded = Retriever(build()).expand("a bag or a sack")
    assert expanded.count("sack") == 1


# ------------------------------------------------------------------- search


def test_search_embeds_the_expanded_question_and_returns_passages(monkeypatch):
    seen = {}

    def fake_embed_one(text, model=ollama.EMBED_MODEL):
        seen["text"] = text
        return A

    monkeypatch.setattr(ollama, "embed_one", fake_embed_one)
    hits = Retriever(build()).search("how many litres per bag")
    assert len(hits) == 1
    assert "5 and 6 litres" in hits[0].chunk.content
    assert seen["text"].startswith(QUERY_INSTRUCTION)
    assert "sack" in seen["text"], "the synonym was not carried into the embedding"


def test_a_close_passage_clears_the_threshold(monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: A)
    r = Retriever(build())
    hits = r.search("anything")
    assert r.above_threshold(hits) is True
    assert r.best_score(hits) > 0.9


def test_a_distant_passage_does_not_clear_the_threshold(monkeypatch):
    """This is the abstention rule: nothing may generate from a passage this far away."""
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: B)
    r = Retriever(build())
    hits = r.search("something unrelated")
    assert r.best_score(hits) < DEFAULT_THRESHOLD
    assert r.above_threshold(hits) is False


def test_the_threshold_can_be_overridden_for_the_sweep(monkeypatch):
    """The sweep prints behaviour either side of the chosen value, so it must be settable."""
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: A)
    hits = Retriever(build()).search("anything")
    assert Retriever(build(), threshold=0.1).above_threshold(hits) is True
    assert Retriever(build(), threshold=1.5).above_threshold(hits) is False


def test_no_hits_is_below_threshold_and_scores_zero():
    r = Retriever(build())
    assert r.above_threshold([]) is False
    assert r.best_score([]) == 0.0


def test_an_index_with_no_embedded_chunks_returns_nothing(monkeypatch):
    """A snapshot can exist with nothing searchable in it; that must not raise."""
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: A)
    assert Retriever(build(chunks=False)).search("anything") == []
