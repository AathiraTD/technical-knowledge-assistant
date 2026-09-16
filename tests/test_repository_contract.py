"""One behavioural contract, run against every `KnowledgeRepository` adapter.

The point of a repository boundary is that the engine cannot tell the adapters
apart. That claim is only worth making if something checks it, so the tests
below are written against the Protocol and parametrised over adapters — the
SQLite one today, and the PostgreSQL one against a live pgvector instance when
one is reachable.

What is checked here is not "does it store rows" but the invariants that make
answers safe: exactly one active version per document, superseded versions
unreachable by retrieval, audience filtering applied to rows rather than to a
prompt, authority outranking similarity inside a band, per-document caps, and a
failed publish leaving the previously serving snapshot intact.

Runs without Ollama. Embeddings here are hand-made unit vectors, because the
adapter's job is to rank what it is given, not to produce it.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.model import (                                    # noqa: E402
    AnswerLogEntry, Caveat, Chunk, CrawlRun, Document, DocumentUpdate,
    DocumentVersion, Excluded, Snapshot,
)
from assistant.repository import (                               # noqa: E402
    KnowledgeRepository, RetrievalRequest,
)
from assistant.store import SQLiteKnowledgeRepository            # noqa: E402

DIMS = 1024


def vec(*leading: float) -> list[float]:
    """A unit vector whose first components are given; the rest are zero."""
    v = list(leading) + [0.0] * (DIMS - len(leading))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


A = vec(1, 0, 0)      # "about mixing"
B = vec(0, 1, 0)      # "about coverage"
NEAR_A = vec(0.99, 0.14, 0)


def doc(url: str, dtype: str = "datasheet", authority: int = 1,
        audience: str = "public", product: str = "Solo") -> Document:
    return Document(canonical_url=url, title=url, document_type=dtype,
                    authority=authority, audience=audience, product=product,
                    link_text=f"{product} {dtype}")


def ver(url: str, n: int = 1, active: bool = True) -> DocumentVersion:
    return DocumentVersion(canonical_url=url, version=n, content_hash=f"h{n}",
                           source_path=f"{url}.v{n}", is_active=active,
                           first_seen_at="2026-01-01", fetched_at="2026-01-01",
                           checked_at="2026-01-01")


def chunk(url: str, i: int, text: str, emb: list[float], version: int = 1,
          audience: str = "public", authority: int = 1,
          product: str = "Solo", date: str = "2026-01-01",
          dtype: str = "datasheet") -> Chunk:
    return Chunk(canonical_url=url, version=version, chunk_index=i,
                 section=f"Section {i}", content=text, audience=audience,
                 product=product, document_type=dtype, authority=authority,
                 source_date=date, embedding=emb)


def snap(sid: str = "snap-1", chunks: int = 0, docs: int = 0) -> Snapshot:
    return Snapshot(snapshot_id=sid, created_at="2026-01-01T00:00:00Z",
                    embedding_model="test-model", embedding_dimensions=DIMS,
                    chunking_version="test/1.0", document_count=docs,
                    chunk_count=chunks, notes={"products": ["Solo"]})


# --------------------------------------------------------------- the adapters


def sqlite_adapter():
    path = Path(tempfile.mkdtemp()) / "contract.db"
    return SQLiteKnowledgeRepository(path)


def postgres_adapter():
    """Returns an adapter against a live pgvector instance, or None."""
    try:
        from assistant.store.postgres import PostgresKnowledgeRepository
    except ImportError:
        return None
    import os
    dsn = os.environ.get("ASSISTANT_POSTGRES_DSN")
    if not dsn:
        return None
    try:
        return PostgresKnowledgeRepository(dsn)
    except Exception:
        return None


ADAPTERS = [("sqlite", sqlite_adapter), ("postgres", postgres_adapter)]


# ----------------------------------------------------------------- the tests


def test_satisfies_the_protocol(repo):
    assert isinstance(repo, KnowledgeRepository)


def test_publish_then_retrieve(repo):
    url = "https://example/solo"
    repo.publish([doc(url)], [ver(url)],
                 [chunk(url, 0, "Add 5 to 6 litres per sack.", A)],
                 snap(chunks=1, docs=1))
    hits = repo.retrieve(A)
    assert len(hits) == 1
    assert "5 to 6 litres" in hits[0].chunk.content
    assert hits[0].score > 0.9


def test_snapshot_records_the_model(repo):
    repo.publish([], [], [], snap())
    s = repo.snapshot()
    assert s.embedding_model == "test-model"
    assert s.embedding_dimensions == DIMS
    assert s.chunking_version == "test/1.0"


def test_only_one_version_is_active(repo):
    """The invariant that stops two coverage figures being retrievable at once."""
    url = "https://example/solo"
    repo.publish(
        [doc(url)],
        [ver(url, 1, active=False), ver(url, 2, active=True)],
        [chunk(url, 0, "OLD: coverage 16-20 m2.", A, version=1),
         chunk(url, 0, "NEW: coverage 14-18 m2.", A, version=2)],
        snap(chunks=2, docs=1),
    )
    hits = repo.retrieve(A, top_k=10)
    texts = " ".join(h.chunk.content for h in hits)
    assert "NEW" in texts
    assert "OLD" not in texts, "a superseded version was retrievable"


def test_active_version_is_reported(repo):
    url = "https://example/solo"
    repo.publish([doc(url)],
                 [ver(url, 1, active=False), ver(url, 2, active=True)],
                 [], snap(docs=1))
    v = repo.active_version(url)
    assert v is not None and v.version == 2


def test_audience_filter_is_applied_to_rows(repo):
    """Staff material must be unreachable, not merely unmentioned."""
    pub, staff = "https://example/public", "fixture://staff/margin"
    repo.publish(
        [doc(pub), doc(staff, audience="staff", dtype="knowledge_base", authority=4)],
        [ver(pub), ver(staff)],
        [chunk(pub, 0, "Public passage about mixing.", A),
         chunk(staff, 0, "The internal margin is 42%.", A,
               audience="staff", authority=4)],
        snap(chunks=2, docs=2),
    )
    public_hits = repo.retrieve(A, audiences=("public",), top_k=10)
    assert all(h.chunk.audience == "public" for h in public_hits)
    assert not any("42%" in h.chunk.content for h in public_hits)

    staff_hits = repo.retrieve(A, audiences=("staff",), top_k=10)
    assert any("42%" in h.chunk.content for h in staff_hits)


def test_manifest_respects_audience(repo):
    pub, staff = "https://example/public", "fixture://staff/margin"
    repo.publish([doc(pub), doc(staff, audience="staff")],
                 [ver(pub), ver(staff)], [], snap(docs=2))
    public = {d.canonical_url for d in repo.manifest(("public",))}
    assert pub in public and staff not in public
    assert staff in {d.canonical_url for d in repo.manifest(("staff",))}


def test_authority_breaks_a_near_tie(repo):
    """A near-identical FAQ answer must not outrank the datasheet."""
    sheet, faq = "https://example/sheet", "https://example/faq"
    repo.publish(
        [doc(sheet, "datasheet", 1), doc(faq, "faq", 5)],
        [ver(sheet), ver(faq)],
        [chunk(sheet, 0, "DATASHEET says 5 to 6 litres.", NEAR_A,
               authority=1, dtype="datasheet"),
         chunk(faq, 0, "FAQ says about 5 litres.", A, authority=5, dtype="faq")],
        snap(chunks=2, docs=2),
    )
    hits = repo.retrieve(A, top_k=2)
    assert hits[0].chunk.document_type == "datasheet", (
        "the FAQ outranked the datasheet on a similarity rounding difference")


def test_the_newer_of_two_equal_sheets_wins(repo):
    """"Newest wins within a type" is documented policy with nothing testing it.

    Decision 12 makes recency a tiebreaker *within* an authority class, and the
    ranking sort is stable over a date-descending pass to achieve it. Deleting
    that pass left the entire suite green, which means the rule was asserted in
    three documents and enforced by nothing a test could see. Two datasheets of
    equal authority and equal similarity, differing only in printed date: the
    2025 one must come first.
    """
    # Named so that alphabetical order contradicts date order. The load query
    # sorts by canonical_url, so "old"/"new" would have put the 2025 sheet first
    # by accident and the test would have passed with the recency sort deleted —
    # which is exactly what it did before these names were chosen.
    old_sheet, new_sheet = "https://example/a-2015", "https://example/b-2025"
    repo.publish(
        [doc(old_sheet, "datasheet", 1), doc(new_sheet, "datasheet", 1)],
        [ver(old_sheet), ver(new_sheet)],
        [chunk(old_sheet, 0, "Solo needs 5 to 6 litres per sack.", A,
               authority=1, dtype="datasheet", date="2015-08-01"),
         chunk(new_sheet, 0, "Solo needs 5 to 6 litres per sack.", A,
               authority=1, dtype="datasheet", date="2025-10-01")],
        snap(chunks=2, docs=2),
    )

    hits = repo.retrieve(A, top_k=2)
    assert hits[0].chunk.source_date == "2025-10-01", (
        "the 2015 sheet outranked the 2025 one at equal authority and equal "
        "similarity; recency is the documented tiebreaker")


def test_recency_cannot_overturn_authority(repo):
    """The other half of the rule, and the one that protects the datasheet.

    Decision 12 is explicit that newer content does not automatically outrank a
    more authoritative source. A FAQ edited this morning must still lose to a
    datasheet printed in 2015, or the tiebreaker has quietly become the ranking.
    """
    sheet, faq = "https://example/sheet", "https://example/faq"
    repo.publish(
        [doc(sheet, "datasheet", 1), doc(faq, "faq", 5)],
        [ver(sheet), ver(faq)],
        [chunk(sheet, 0, "Solo needs 5 to 6 litres per sack.", NEAR_A,
               authority=1, dtype="datasheet", date="2015-08-01"),
         chunk(faq, 0, "Solo needs about 5 litres per sack.", A,
               authority=5, dtype="faq", date="2026-09-01")],
        snap(chunks=2, docs=2),
    )

    hits = repo.retrieve(A, top_k=2)
    assert hits[0].chunk.document_type == "datasheet", (
        "a newer FAQ outranked an older datasheet; recency is a tiebreaker "
        "within an authority class, not across one")


def test_per_document_cap(repo):
    """A multi-source question must see several documents, not one page five times."""
    one, two = "https://example/one", "https://example/two"
    repo.publish(
        [doc(one), doc(two)],
        [ver(one), ver(two)],
        [chunk(one, i, f"One, passage {i}.", A) for i in range(5)]
        + [chunk(two, 0, "Two, the only passage.", NEAR_A)],
        snap(chunks=6, docs=2),
    )
    hits = repo.retrieve(A, top_k=5, per_document_cap=2)
    from collections import Counter
    counts = Counter(h.chunk.canonical_url for h in hits)
    assert max(counts.values()) <= 2
    assert len(counts) >= 2


def test_caveats_come_back_for_their_document(repo):
    url = "https://example/solo"
    repo.publish([doc(url)], [ver(url)], [chunk(url, 0, "Mixing.", A)],
                 snap(chunks=1, docs=1),
                 caveats=[Caveat(url, "temperature",
                                 "Do not apply below 5 degrees C.", "Mixing")])
    cavs = repo.caveats(url)
    assert len(cavs) == 1
    assert cavs[0].caveat_type == "temperature"
    assert repo.caveats("https://example/nothing") == []


def test_excluded_documents_are_on_file(repo):
    repo.publish([], [], [], snap(),
                 excluded=[Excluded("https://example/sds.pdf",
                                    "safety data sheets are read whole")])
    ex = repo.excluded()
    assert len(ex) == 1
    assert "safety data sheet" in ex[0].reason


def test_document_lookup(repo):
    url = "https://example/solo"
    repo.publish([doc(url)], [ver(url)], [], snap(docs=1))
    d = repo.document(url)
    assert d is not None and d.document_type == "datasheet"
    assert repo.document("https://example/missing") is None


def test_a_failed_publish_leaves_the_serving_index_intact(repo):
    """Either the whole snapshot becomes live or none of it does."""
    url = "https://example/solo"
    repo.publish([doc(url)], [ver(url)],
                 [chunk(url, 0, "GOOD passage.", A)], snap("snap-good", 1, 1))

    # A chunk whose version does not exist, plus a snapshot that will violate
    # the one-active-version index: the transaction must roll back whole.
    broken_url = "https://example/broken"
    try:
        repo.publish(
            [doc(broken_url)],
            [ver(broken_url, 1, active=True), ver(broken_url, 2, active=True)],
            [chunk(broken_url, 0, "BAD passage.", A)],
            snap("snap-bad", 1, 1),
        )
        raise AssertionError("publishing two active versions should have failed")
    except AssertionError:
        raise
    except Exception:
        pass  # the database refused it, which is the point

    hits = repo.retrieve(A, top_k=5)
    assert any("GOOD" in h.chunk.content for h in hits), (
        "a failed publish destroyed the serving index")
    assert repo.snapshot().snapshot_id == "snap-good"


# ------------------------------------------- the product the question named
#
# Retrieval used to know about similarity, audience, version and authority and
# nothing about the product a question named outright. These hold both adapters
# to the same answer on that.


ULTRA = "https://example/ultra-datasheet"
WARMSHELL = "https://example/warmshell-guide"

# The reviewer's case, as vectors. The WarmShell passage is the *better*
# semantic match - both documents are about old walls and internal insulation,
# and the corpus is dense there - and the Ultra passage is a few hundredths
# behind it. Nothing in the old retriever could prefer Ultra.
WS_VECTOR = vec(1, 0, 0)
ULTRA_VECTOR = vec(0.985, 0.174, 0)      # cosine ~0.985 against the question


def publish_two_products(repo, ultra_authority: int = 1,
                         ultra_type: str = "datasheet") -> None:
    repo.publish(
        [doc(ULTRA, ultra_type, ultra_authority, product="Ultra"),
         doc(WARMSHELL, "system_guide", 2, product="WarmShell")],
        [ver(ULTRA), ver(WARMSHELL)],
        [chunk(ULTRA, 0, "ULTRA on an old internal wall.", ULTRA_VECTOR,
               product="Ultra", authority=ultra_authority, dtype=ultra_type),
         chunk(WARMSHELL, 0, "WARMSHELL insulates an old internal wall.",
               WS_VECTOR, product="WarmShell", authority=2,
               dtype="system_guide")],
        snap(chunks=2, docs=2),
    )


def test_without_a_named_product_the_neighbour_still_wins(repo):
    """The baseline, so the next test measures the boost and not the fixture."""
    publish_two_products(repo)
    hits = repo.retrieve(WS_VECTOR, top_k=2)
    assert hits[0].chunk.product == "WarmShell"
    assert hits[1].chunk.product == "Ultra"
    assert hits[0].score > hits[1].score


def test_a_named_product_is_not_answered_from_its_semantic_neighbour(repo):
    """Say Ultra, get Ultra - even when WarmShell is the closer vector.

    This is the whole reason the request exists. Both documents concern an old
    internal wall, so semantic adjacency alone puts WarmShell first; the corpus
    already carried `product` on every chunk and retrieval had no way to be told
    which one the customer said out loud.
    """
    publish_two_products(repo)
    hits = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, top_k=2, product="Lime Green Ultra"))
    assert hits[0].chunk.product == "Ultra", (
        "a question naming Ultra was answered from WarmShell because WarmShell "
        "was the closer vector")
    # And the reported score is still the real cosine, not the boosted one.
    assert hits[0].score < hits[1].score


def test_the_boost_does_not_filter_the_other_product_out(repo):
    """A boost, not a filter: multi-source questions must still see both.

    A hard product filter is the obvious implementation and it would break the
    brief's own multi-source questions, which legitimately span two documents
    and sometimes two products. The neighbour must be demoted, not deleted.
    """
    publish_two_products(repo)
    hits = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, top_k=5, product="Ultra"))
    assert {h.chunk.product for h in hits} == {"Ultra", "WarmShell"}


def test_the_boost_is_bounded_and_cannot_drag_in_the_unrelated(repo):
    """The other half of "bounded": a name is not a licence to return anything."""
    publish_two_products(repo)
    far = "https://example/ultra-unrelated"
    repo.apply_delta(
        [DocumentUpdate(
            document=doc(far, "datasheet", 1, product="Ultra"),
            version=DocumentVersion(canonical_url=far, version=1,
                                    content_hash="hf", source_path="f",
                                    is_active=True, fetched_at="2026-01-01"),
            chunks=[chunk(far, 0, "ULTRA, on something else entirely.", B,
                          product="Ultra")])],
        [], snap("snap-far", 3, 3))
    hits = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, top_k=2, product="Ultra"))
    assert not any("entirely" in h.chunk.content for h in hits), (
        "a passage 0.0 similar was boosted into the top two by its product tag")


def test_the_product_boost_does_not_reorder_within_the_product(repo):
    """Authority still orders the named product's own passages.

    Every chunk of the named product gets the same bonus, so the group moves and
    the ordering inside it is untouched - which is what keeps "authority beats a
    marginal similarity difference" true rather than approximately true.
    """
    sheet, faq = "https://example/ultra-tds", "https://example/ultra-faq"
    repo.publish(
        [doc(sheet, "datasheet", 1, product="Ultra"),
         doc(faq, "faq", 5, product="Ultra")],
        [ver(sheet), ver(faq)],
        [chunk(sheet, 0, "DATASHEET on Ultra.", NEAR_A, authority=1,
               dtype="datasheet", product="Ultra"),
         chunk(faq, 0, "FAQ on Ultra.", A, authority=5,
               dtype="faq", product="Ultra")],
        snap(chunks=2, docs=2),
    )
    hits = repo.retrieve_for(RetrievalRequest(embedding=A, top_k=2,
                                              product="Ultra"))
    assert hits[0].chunk.document_type == "datasheet"


def test_the_product_boost_cannot_reach_a_forbidden_audience(repo):
    """The audience filter runs before anything is boosted or ranked."""
    pub, staff = "https://example/public", "fixture://staff/ultra"
    repo.publish(
        [doc(pub, product="WarmShell"),
         doc(staff, "knowledge_base", 4, audience="staff", product="Ultra")],
        [ver(pub), ver(staff)],
        [chunk(pub, 0, "Public passage.", WS_VECTOR, product="WarmShell"),
         chunk(staff, 0, "The internal Ultra margin is 42 percent.",
               ULTRA_VECTOR, audience="staff", authority=4, product="Ultra")],
        snap(chunks=2, docs=2),
    )
    hits = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, audiences=("public",), top_k=5, product="Ultra"))
    assert not any("42 percent" in h.chunk.content for h in hits)


def test_an_unknown_product_name_changes_nothing(repo):
    publish_two_products(repo)
    plain = repo.retrieve(WS_VECTOR, top_k=2)
    named = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, top_k=2, product="Nonesuch"))
    assert [h.chunk.content for h in named] == [h.chunk.content for h in plain]


def test_an_untagged_chunk_is_never_boosted(repo):
    """An empty product tag must not match every name by containment."""
    publish_two_products(repo)
    untagged = "https://example/untagged"
    repo.apply_delta(
        [DocumentUpdate(
            document=doc(untagged, "datasheet", 1, product=""),
            version=DocumentVersion(canonical_url=untagged, version=1,
                                    content_hash="hu", source_path="u",
                                    is_active=True, fetched_at="2026-01-01"),
            chunks=[chunk(untagged, 0, "UNTAGGED passage.", ULTRA_VECTOR,
                          product="")])],
        [], snap("snap-untagged", 3, 3))
    hits = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, top_k=1, product="Ultra"))
    assert hits[0].chunk.product == "Ultra"


def test_the_named_product_still_respects_the_per_document_cap(repo):
    """The cap is the multi-source guarantee; a boost must not be able to lift it."""
    repo.publish(
        [doc(ULTRA, "datasheet", 1, product="Ultra"),
         doc(WARMSHELL, "system_guide", 2, product="WarmShell")],
        [ver(ULTRA), ver(WARMSHELL)],
        [chunk(ULTRA, i, "Ultra passage %d." % i, ULTRA_VECTOR, product="Ultra")
         for i in range(5)]
        + [chunk(WARMSHELL, 0, "WarmShell passage.", WS_VECTOR,
                 product="WarmShell", authority=2, dtype="system_guide")],
        snap(chunks=6, docs=2),
    )
    hits = repo.retrieve_for(RetrievalRequest(
        embedding=WS_VECTOR, top_k=5, per_document_cap=2, product="Ultra"))
    from collections import Counter
    counts = Counter(h.chunk.canonical_url for h in hits)
    assert counts[ULTRA] == 2
    assert counts[WARMSHELL] == 1


# ------------------------------------------ the targeted second retrieval


def publish_a_coverage_sheet(repo) -> None:
    repo.publish(
        [doc(ULTRA, "datasheet", 1, product="Ultra"),
         doc(WARMSHELL, "system_guide", 2, product="WarmShell")],
        [ver(ULTRA), ver(WARMSHELL)],
        [chunk(ULTRA, 0, "Mixing: add 5 to 6 litres per sack.", A,
               product="Ultra"),
         chunk(ULTRA, 1, "Coverage: 16 to 20 square metres per 25kg sack.", B,
               product="Ultra"),
         chunk(WARMSHELL, 0, "Coverage: 8 square metres per board pack.", A,
               product="WarmShell", authority=2, dtype="system_guide")],
        snap(chunks=3, docs=2),
    )


def test_a_targeted_lookup_finds_the_property_the_top_five_missed(repo):
    """The calculation path's recovery: coverage by metadata, not by similarity."""
    publish_a_coverage_sheet(repo)
    hits = repo.find_passages("Ultra", ("coverage",))
    assert len(hits) == 1
    assert "16 to 20" in hits[0].chunk.content
    assert hits[0].chunk.product == "Ultra"
    assert hits[0].score == 0.0, "a lexical hit must not look like a similarity"


def test_a_targeted_lookup_is_confined_to_the_named_product(repo):
    """Unlike the boost, this one *is* a filter - the caller named the product."""
    publish_a_coverage_sheet(repo)
    hits = repo.find_passages("WarmShell", ("coverage",))
    assert [h.chunk.product for h in hits] == ["WarmShell"]


def test_a_targeted_lookup_matches_a_section_heading(repo):
    publish_a_coverage_sheet(repo)
    assert repo.find_passages("Ultra", ("section 1",))


def test_a_targeted_lookup_honours_audience_and_active_version(repo):
    staff = "fixture://staff/ultra"
    repo.publish(
        [doc(ULTRA, "datasheet", 1, product="Ultra"),
         doc(staff, "knowledge_base", 4, audience="staff", product="Ultra")],
        [ver(ULTRA, 1, active=False), ver(ULTRA, 2, active=True), ver(staff)],
        [chunk(ULTRA, 0, "OLD coverage: 30 square metres.", A, version=1,
               product="Ultra"),
         chunk(ULTRA, 0, "NEW coverage: 16 to 20 square metres.", A, version=2,
               product="Ultra"),
         chunk(staff, 0, "Staff coverage note.", A, audience="staff",
               authority=4, product="Ultra")],
        snap(chunks=3, docs=2),
    )
    texts = " ".join(h.chunk.content
                     for h in repo.find_passages("Ultra", ("coverage",), limit=5))
    assert "NEW" in texts
    assert "OLD" not in texts, "a superseded passage was reachable by metadata"
    assert "Staff" not in texts, "a staff passage was reachable by metadata"


def test_a_targeted_lookup_prefers_the_datasheet_then_the_newer_sheet(repo):
    """Same ordering rule as retrieval: authority first, recency inside it."""
    page, old_sheet, new_sheet = ("https://example/u-page",
                                  "https://example/u-2015",
                                  "https://example/u-2025")
    repo.publish(
        [doc(page, "product_page", 3, product="Ultra"),
         doc(old_sheet, "datasheet", 1, product="Ultra"),
         doc(new_sheet, "datasheet", 1, product="Ultra")],
        [ver(page), ver(old_sheet), ver(new_sheet)],
        [chunk(page, 0, "PAGE coverage, roughly.", A, authority=3,
               dtype="product_page", product="Ultra", date="2026-01-01"),
         chunk(old_sheet, 0, "2015 coverage: 18 square metres.", A,
               product="Ultra", date="2015-08-01"),
         chunk(new_sheet, 0, "2025 coverage: 16 square metres.", A,
               product="Ultra", date="2025-10-01")],
        snap(chunks=3, docs=3),
    )
    hits = repo.find_passages("Ultra", ("coverage",), limit=3)
    assert hits[0].chunk.source_date == "2025-10-01"
    assert hits[1].chunk.source_date == "2015-08-01"
    assert hits[2].chunk.document_type == "product_page"


def test_a_targeted_lookup_caps_what_it_returns(repo):
    publish_a_coverage_sheet(repo)
    assert len(repo.find_passages("Ultra", ("coverage", "mixing"), limit=2)) == 2
    assert len(repo.find_passages("Ultra", ("coverage", "mixing"), limit=1)) == 1


def test_a_targeted_lookup_with_nothing_to_go_on_returns_nothing(repo):
    publish_a_coverage_sheet(repo)
    assert repo.find_passages("", ("coverage",)) == []
    assert repo.find_passages("Ultra", ()) == []
    assert repo.find_passages("Ultra", ("pot life",)) == []


def test_retrieve_on_an_empty_index(repo):
    assert repo.retrieve(A) == []


# ------------------------------------------------------------------- runner


def main() -> int:
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    total_failed = []

    for name, factory in ADAPTERS:
        print(f"\n{name}")
        first = factory()
        if first is None:
            print("  skipped — no reachable instance "
                  "(set ASSISTANT_POSTGRES_DSN to run these)")
            continue
        first.close() if hasattr(first, "close") else None

        passed = 0
        for test_name, fn in tests:
            repo = factory()
            try:
                fn(repo)
                passed += 1
                print(f"  pass  {test_name}")
            except Exception:
                total_failed.append(f"{name}:{test_name}")
                print(f"  FAIL  {test_name}")
                traceback.print_exc(limit=2)
            finally:
                if hasattr(repo, "close"):
                    repo.close()
        print(f"  {passed}/{len(tests)} passed")

    if total_failed:
        print(f"\nFAILED: {', '.join(total_failed)}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------- the delta lifecycle

def update(url: str, content: str, emb, version_hash: str,
           product: str = "Solo", caveats=()) -> "DocumentUpdate":
    """One document's new state, as an indexing run would submit it."""
    return DocumentUpdate(
        document=doc(url, product=product),
        version=DocumentVersion(canonical_url=url, version=1,
                                content_hash=version_hash, source_path=f"{url}.src",
                                is_active=True, fetched_at="2026-01-01"),
        chunks=[chunk(url, 0, content, emb, product=product)],
        caveats=list(caveats),
    )


def test_a_first_delta_behaves_like_a_first_build(repo):
    """An empty store plus a delta is the normal starting state, not a special case."""
    repo.apply_delta([update("https://x/a", "A one.", A, "h1")], [], snap("s1", 1, 1))
    hits = repo.retrieve(A, top_k=5)
    assert len(hits) == 1
    assert repo.active_content_hashes() == {"https://x/a": "h1"}


def test_a_changed_document_keeps_its_old_version_and_serves_only_the_new(repo):
    """This is the audit claim: an answer given last year must still be explicable."""
    repo.apply_delta([update("https://x/a", "OLD coverage 16-20.", A, "h1")], [],
                     snap("s1", 1, 1))
    repo.apply_delta([update("https://x/a", "NEW coverage 14-18.", A, "h2")], [],
                     snap("s2", 1, 1))

    versions = repo.versions("https://x/a")
    assert len(versions) == 2, "the superseded version was discarded"
    assert [v.is_active for v in versions] == [True, False]
    assert versions[0].content_hash == "h2"
    assert versions[1].content_hash == "h1"

    served = " ".join(h.chunk.content for h in repo.retrieve(A, top_k=10))
    assert "NEW" in served
    assert "OLD" not in served, "a superseded version was retrievable"


def test_an_unchanged_document_is_not_touched_by_a_later_delta(repo):
    """Unchanged must mean no work, or the pipeline is a rebuild wearing a delta's name."""
    repo.apply_delta([update("https://x/a", "A one.", A, "h1"),
                      update("https://x/b", "B one.", B, "h1b")], [], snap("s1", 2, 2))
    before = repo.versions("https://x/b")

    repo.apply_delta([update("https://x/a", "A two.", A, "h2")], [], snap("s2", 2, 2))

    after = repo.versions("https://x/b")
    assert len(after) == len(before) == 1
    assert after[0].content_hash == "h1b"
    assert len(repo.versions("https://x/a")) == 2


def test_a_removed_document_stops_being_served_but_keeps_its_history(repo):
    """A withdrawn datasheet must not keep being quoted as current."""
    repo.apply_delta([update("https://x/a", "Still here.", A, "h1"),
                      update("https://x/gone", "Withdrawn.", A, "h1g")], [],
                     snap("s1", 2, 2))
    repo.apply_delta([], ["https://x/gone"], snap("s2", 1, 1))

    served = " ".join(h.chunk.content for h in repo.retrieve(A, top_k=10))
    assert "Withdrawn" not in served
    assert "Still here" in served
    assert len(repo.versions("https://x/gone")) == 1, "history was deleted"
    assert repo.versions("https://x/gone")[0].is_active is False
    assert "https://x/gone" not in repo.active_content_hashes()
    assert repo.document("https://x/gone") is not None


def test_removing_a_document_that_was_never_indexed_is_not_an_error(repo):
    repo.apply_delta([update("https://x/a", "A one.", A, "h1")], [], snap("s1", 1, 1))
    repo.apply_delta([], ["https://x/never"], snap("s2", 1, 1))
    assert len(repo.retrieve(A, top_k=5)) == 1


def test_content_hashes_are_what_the_indexer_diffs_against(repo):
    repo.apply_delta([update("https://x/a", "A.", A, "h1"),
                      update("https://x/b", "B.", B, "h2")], [], snap("s1", 2, 2))
    assert repo.active_content_hashes() == {"https://x/a": "h1", "https://x/b": "h2"}


def test_a_crawl_run_is_recorded_rather_than_printed(repo):
    """A console line saying 94 unchanged disappears; a row is evidence."""
    run = CrawlRun(started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:05:00Z",
                   documents_checked=94, documents_new=0, documents_changed=1,
                   documents_unchanged=93)
    repo.apply_delta([update("https://x/a", "A.", A, "h1")], [], snap("s1", 1, 1),
                     crawl_run=run)
    runs = repo.crawl_runs()
    assert len(runs) == 1
    assert runs[0].documents_unchanged == 93
    assert runs[0].documents_changed == 1
    assert runs[0].reprocessed == 1


def test_caveats_follow_the_active_version_and_not_the_old_one(repo):
    """A caveat quoted from a superseded sheet is a caveat about nothing served."""
    repo.apply_delta([update("https://x/a", "A one.", A, "h1",
                             caveats=[Caveat("https://x/a", "temperature",
                                             "Old limit: above 3 C.", "Mixing")])],
                     [], snap("s1", 1, 1))
    repo.apply_delta([update("https://x/a", "A two.", A, "h2",
                             caveats=[Caveat("https://x/a", "temperature",
                                             "New limit: above 5 C.", "Mixing")])],
                     [], snap("s2", 1, 1))
    sentences = [c.sentence for c in repo.caveats("https://x/a")]
    assert sentences == ["New limit: above 5 C."], sentences


def test_a_failed_delta_leaves_the_previous_state_intact(repo):
    """Half an applied delta is a corrupt index, so it must be all or nothing."""
    repo.apply_delta([update("https://x/a", "GOOD.", A, "h1")], [], snap("s1", 1, 1))

    broken = update("https://x/b", "BAD.", A, "h1b")
    broken.chunks[0].embedding = ["not a number"]
    try:
        repo.apply_delta([broken], [], snap("s2", 2, 2))
    except Exception:
        pass

    served = " ".join(h.chunk.content for h in repo.retrieve(A, top_k=10))
    assert "GOOD" in served
    assert "BAD" not in served
    assert repo.snapshot().snapshot_id == "s1"


# ------------------------------------------------- referential integrity

def _counts(repo) -> dict:
    """Row counts straight from the tables, bypassing the active-version joins."""
    tables = ("documents", "document_versions", "chunks", "document_caveats")
    out = {}
    for t in tables:
        if hasattr(repo, "db"):                       # SQLite
            out[t] = repo.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        else:                                          # PostgreSQL
            with repo.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                out[t] = cur.fetchone()[0]
    return out


def test_a_superseded_version_keeps_its_own_chunks_and_caveats(repo):
    """History is only auditable if the passages that produced an answer survive."""
    url = "https://x/a"
    repo.apply_delta([update(url, "OLD text.", A, "h1",
                             caveats=[Caveat(url, "temperature", "Old limit.", "Mixing")])],
                     [], snap("s1", 1, 1))
    repo.apply_delta([update(url, "NEW text.", A, "h2",
                             caveats=[Caveat(url, "temperature", "New limit.", "Mixing")])],
                     [], snap("s2", 1, 1))

    counts = _counts(repo)
    assert counts["documents"] == 1, "a new version must not duplicate the document"
    assert counts["document_versions"] == 2
    assert counts["chunks"] == 2, "the superseded version's passage was deleted"
    assert counts["document_caveats"] == 2


def test_reprocessing_every_document_preserves_rather_than_replaces(repo):
    """A rebuild is a delta against an empty comparison, not a wipe.

    There is deliberately no clear-and-refill path. Clearing first is how a
    rebuild that fails halfway leaves nothing serving, and it also discards the
    history that makes an old answer explicable. Reprocessing writes a new
    version for every document and keeps every old one.
    """
    repo.apply_delta(
        [update("https://x/a", "A one.", A, "h1",
                caveats=[Caveat("https://x/a", "diy", "Needs a plasterer.", "Use")]),
         update("https://x/b", "B one.", B, "h1b")],
        [], snap("s1", 2, 2))

    # What a rebuild does: submit every document again, ignoring the hashes.
    repo.apply_delta(
        [update("https://x/a", "A two.", A, "h1"),
         update("https://x/b", "B two.", B, "h1b")],
        [], snap("s2", 2, 2))

    counts = _counts(repo)
    assert counts["documents"] == 2, "reprocessing duplicated a document row"
    assert counts["document_versions"] == 4
    assert counts["chunks"] == 4, "a superseded passage was deleted"

    served = " ".join(h.chunk.content for h in repo.retrieve(A, top_k=10))
    assert "A two" in served
    assert "A one" not in served, "a superseded version was retrievable"


def test_a_delta_that_fails_leaves_every_row_where_it_was(repo):
    """Half an applied delta is a corrupt index, so it is all or nothing."""
    repo.apply_delta([update("https://x/a", "GOOD.", A, "h1")], [], snap("s1", 1, 1))
    before = _counts(repo)

    broken = update("https://x/b", "BAD.", A, "h1b")
    broken.chunks[0].embedding = ["not a number"]
    try:
        repo.apply_delta([broken], [], snap("s2", 1, 1))
    except Exception:
        pass

    assert _counts(repo) == before, "a failed delta left rows behind"
    assert "GOOD" in " ".join(h.chunk.content for h in repo.retrieve(A, top_k=5))
    assert repo.snapshot().snapshot_id == "s1"


# ------------------------------------------------------------- the answer log

URL = "https://example/solo"


def one_answer(repo) -> str:
    """Publish a single passage so there is a snapshot and a chunk to cite."""
    repo.publish([doc(URL)], [ver(URL)],
                 [chunk(URL, 0, "Add 5 to 6 litres per sack.", A)],
                 snap("snap-1", chunks=1, docs=1))
    return f"{URL}#v1-0"


def test_an_answer_records_the_evidence_that_produced_it(repo):
    """Without the snapshot and the passages, "why did it say that?" is guesswork."""
    cited = one_answer(repo)
    repo.log_answer(AnswerLogEntry(
        question="How much water does Solo need?",
        path_taken="extract",
        audiences=("public", "trade"),
        snapshot_id="snap-1",
        chunk_ids=[cited],
        generation_model="test-model",
        asked_at="2026-01-02T09:00:00+00:00",
    ))

    logged = repo.answer_log()
    assert len(logged) == 1
    entry = logged[0]
    assert entry.question == "How much water does Solo need?"
    assert entry.path_taken == "extract"
    assert entry.audiences == ("public", "trade")
    assert entry.snapshot_id == "snap-1"
    assert entry.chunk_ids == [cited], "the cited passages were not recoverable"
    assert entry.generation_model == "test-model"
    assert entry.check_failed == ""


def test_a_refusal_is_logged_with_the_check_that_stopped_it(repo):
    """A refusal nobody counted is an over-refusal rate nobody can quote."""
    one_answer(repo)
    repo.log_answer(AnswerLogEntry(
        question="What is the vapour permeability of Solo?",
        path_taken="refuse",
        snapshot_id="snap-1",
        check_failed="relevance",
        asked_at="2026-01-02T09:01:00+00:00",
    ))

    entry = repo.answer_log()[0]
    assert entry.path_taken == "refuse"
    assert entry.check_failed == "relevance"
    assert entry.chunk_ids == []
    assert entry.generation_model == "", "a refusal must not claim the model ran"


def test_an_answer_given_before_any_index_existed_is_still_logged(repo):
    """A question asked against an empty store is a fact about the store, not a write to drop."""
    repo.log_answer(AnswerLogEntry(question="Anything at all?", path_taken="refuse",
                                   check_failed="no_snapshot"))
    entry = repo.answer_log()[0]
    assert entry.snapshot_id == ""
    assert entry.audiences == ("public",)


def test_the_store_timestamps_an_answer_that_arrives_without_one(repo):
    """A row with no time in it cannot be put in order, which is all the log is for."""
    repo.log_answer(AnswerLogEntry(question="When was this asked?", path_taken="route"))
    assert repo.answer_log()[0].asked_at.startswith("20")


def test_the_answer_log_reads_newest_first_and_respects_its_limit(repo):
    """Oldest first, uncapped, is the shape that makes an audit trail unusable."""
    one_answer(repo)
    for minute, question in enumerate(["first", "second", "third"]):
        repo.log_answer(AnswerLogEntry(
            question=question, path_taken="extract", snapshot_id="snap-1",
            asked_at=f"2026-01-02T09:0{minute}:00+00:00"))

    assert [e.question for e in repo.answer_log()] == ["third", "second", "first"]
    assert [e.question for e in repo.answer_log(limit=2)] == ["third", "second"]


def test_an_empty_answer_log_is_a_state_and_not_a_failure(repo):
    assert repo.answer_log() == []


def test_logging_an_answer_is_not_rolled_back_with_the_snapshot_it_describes(repo):
    """The write shares a connection with the read snapshot and the audit row vanishes."""
    cited = one_answer(repo)

    with repo.read_snapshot() as pinned:
        assert pinned is not None and pinned.snapshot_id == "snap-1"
        repo.log_answer(AnswerLogEntry(
            question="Logged from inside the snapshot.", path_taken="extract",
            snapshot_id=pinned.snapshot_id, chunk_ids=[cited],
            asked_at="2026-01-02T09:02:00+00:00"))
        # The read must survive the write: same snapshot, same evidence.
        assert repo.snapshot().snapshot_id == "snap-1"
        assert len(repo.retrieve(A)) == 1

    logged = repo.answer_log()
    assert [e.question for e in logged] == ["Logged from inside the snapshot."]
    assert logged[0].chunk_ids == [cited]
