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

from assistant.indexing.index import CHUNKING_VERSION
from assistant.infrastructure import ollama
from assistant.knowledge.model import (  # noqa: E402
    Chunk,
    Document,
    DocumentVersion,
    Snapshot,
)
from assistant.knowledge.repository import RetrievalRequest               # noqa: E402
from assistant.knowledge.repository import IndexMismatch                # noqa: E402
from assistant.retrieval.retrieve import (
    DEFAULT_THRESHOLD,
    OVERFETCH,
    QUERY_INSTRUCTION,
    Retriever,
    _distinct,
    as_query,
)
from assistant.knowledge.store import SQLiteKnowledgeRepository         # noqa: E402

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
    assert "assistant.indexing.index" in str(raised.value)


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


def test_a_passage_exactly_on_the_threshold_is_admitted(monkeypatch):
    """The boundary itself, which no other test touches.

    `above_threshold` is `score >= threshold`. Changing it to `>` leaves the
    whole suite green — confirmed by mutation — because every other case sits
    far from the line. The distinction is not academic: the threshold is the
    abstention rule, the sweep prints behaviour at the chosen value and at plus
    and minus 0.1, and a silent flip of the comparison would move what the
    sweep reports without moving the number it reports it against.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: A)
    hits = Retriever(build()).search("anything")
    exactly = Retriever(build(), threshold=hits[0].score)

    assert exactly.above_threshold(hits) is True, (
        "a passage exactly on the threshold was refused; the rule is >=, "
        "so the documented value is the lowest score that still answers")


def test_a_passage_just_under_the_threshold_is_refused(monkeypatch):
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: A)
    hits = Retriever(build()).search("anything")
    just_over = Retriever(build(), threshold=hits[0].score + 1e-9)

    assert just_over.above_threshold(hits) is False


# ------------------------------------------- the product the question named


WARMSHELL = "https://example/warmshell"
# The reviewer's case at the retriever rather than the adapter: WarmShell is
# the closer vector to the question, and the question says "Ultra".
WS_VECTOR = unit(1, 0, 0)
ULTRA_VECTOR = unit(0.985, 0.174, 0)


def build_two_products() -> SQLiteKnowledgeRepository:
    repo = SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "two.db")
    repo.publish(
        [Document(canonical_url=URL, title="Ultra", document_type="datasheet",
                  authority=1, product="Ultra"),
         Document(canonical_url=WARMSHELL, title="WarmShell",
                  document_type="system_guide", authority=2,
                  product="WarmShell")],
        [DocumentVersion(canonical_url=URL, version=1, content_hash="h",
                         source_path="p", is_active=True),
         DocumentVersion(canonical_url=WARMSHELL, version=1, content_hash="h2",
                         source_path="p2", is_active=True)],
        [Chunk(canonical_url=URL, version=1, chunk_index=0, section="Coverage",
               content="Ultra covers 16 to 20 square metres per 25kg sack.",
               product="Ultra", document_type="datasheet", authority=1,
               embedding=ULTRA_VECTOR),
         Chunk(canonical_url=WARMSHELL, version=1, chunk_index=0,
               section="Internal walls",
               content="WarmShell insulates an old internal wall.",
               product="WarmShell", document_type="system_guide", authority=2,
               embedding=WS_VECTOR)],
        Snapshot(snapshot_id="s2", created_at="2026-01-01T00:00:00Z",
                 embedding_model=ollama.EMBED_MODEL,
                 embedding_dimensions=DIMS,
                 chunking_version=CHUNKING_VERSION, document_count=2,
                 chunk_count=2),
    )
    return repo


def test_a_question_naming_ultra_is_not_answered_from_warmshell(monkeypatch):
    """Semantic adjacency is not consent to answer about a different product."""
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: WS_VECTOR)
    r = Retriever(build_two_products())

    unnamed = r.search("insulating an old internal wall", top_k=2)
    assert unnamed[0].chunk.product == "WarmShell"

    named = r.search("insulating an old internal wall with Lime Green Ultra",
                     top_k=2, product="Ultra")
    assert named[0].chunk.product == "Ultra"
    assert named[1].chunk.product == "WarmShell", (
        "the named product filtered the neighbour out instead of demoting it")
    assert named[0].score < named[1].score, (
        "the score reported is the boosted one, not the real cosine")


def test_the_named_product_reaches_the_repository_as_a_request(monkeypatch):
    """The engine expresses what it wants; it does not learn any SQL to do it."""
    monkeypatch.setattr(ollama, "embed_one", lambda text, model=None: A)
    repo = build()
    seen = {}
    original = repo.retrieve_for

    def spy(request):
        seen["request"] = request
        return original(request)

    repo.retrieve_for = spy
    hits = Retriever(repo).search("how much water", audiences=("trade",),
                                  top_k=4, per_document_cap=2, product="Solo")
    request = seen["request"]
    assert isinstance(request, RetrievalRequest)
    assert request.product == "Solo"
    assert request.audiences == ("trade",)
    # The cap travels through unchanged; `top_k` deliberately does not. The
    # repository is asked wider than the caller wants so that dropping a
    # duplicate passage frees the slot for a different document instead of
    # returning three passages where four were asked for -- see `_distinct`.
    # What the *caller* gets is still bounded by what it asked for, which is the
    # property this line now states.
    assert request.per_document_cap == 2
    assert request.top_k == 4 * OVERFETCH
    assert len(hits) <= 4


# ------------------------------------------ the targeted second retrieval


def test_a_targeted_lookup_finds_coverage_without_embedding_anything(monkeypatch):
    """The calculation path's recovery, and it must not need the model at all."""
    def refuse(*args, **kwargs):
        raise AssertionError("a targeted lookup embedded something")

    monkeypatch.setattr(ollama, "embed_one", refuse)
    r = Retriever(build_two_products())
    hits = r.find_property("Ultra", ("coverage",))
    assert len(hits) == 1
    assert "16 to 20" in hits[0].chunk.content
    assert hits[0].score == 0.0


def test_a_targeted_lookup_finds_nothing_for_an_unpublished_property():
    r = Retriever(build_two_products())
    assert r.find_property("Ultra", ("pot life",)) == []
