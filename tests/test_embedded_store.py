"""The SQLite adapter's own edges, beyond the shared repository contract.

The contract suite in `test_repository_contract.py` covers what every adapter
must do. This covers what this one does on its own: the context manager, the
guards that drop orphaned rows rather than writing dangling references, the
per-document cap, and the reporting helpers the ingestion report prints.

The orphan guards matter more than they look. A chunk whose document version
was never inserted has nowhere to attach, and writing it anyway would either
violate the foreign key or, worse, attach it to whatever row happened to have
that id. Skipping it keeps a partial publish honest.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.model import (                                  # noqa: E402
    Caveat, Chunk, Document, DocumentVersion, Snapshot,
)
from assistant.store import SQLiteKnowledgeRepository          # noqa: E402
from assistant.store.embedded import (                         # noqa: E402
    AUTHORITY_BONUS, TIE_BAND, _f32, _unpack,
)

DIMS = 1024
URL = "https://example/solo"


def unit(*leading: float) -> list[float]:
    v = list(leading) + [0.0] * (DIMS - len(leading))
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


A = unit(1, 0, 0)


def fresh() -> SQLiteKnowledgeRepository:
    return SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "store.db")


def doc(url=URL, product="Solo") -> Document:
    return Document(canonical_url=url, title=product, document_type="datasheet",
                    authority=1, product=product)


def ver(url=URL, n=1, active=True) -> DocumentVersion:
    return DocumentVersion(canonical_url=url, version=n, content_hash="h",
                           source_path="p", is_active=active)


def chunk(url=URL, i=0, version=1, content="Mix with water.") -> Chunk:
    return Chunk(canonical_url=url, version=version, chunk_index=i,
                 section="Mixing", content=content, product="Solo",
                 document_type="datasheet", authority=1, embedding=A)


def snap(docs=1, chunks=1) -> Snapshot:
    return Snapshot(snapshot_id="s1", created_at="2026-01-01T00:00:00Z",
                    embedding_model="test-model", embedding_dimensions=DIMS,
                    chunking_version="test/1.0", document_count=docs,
                    chunk_count=chunks)


# --------------------------------------------------------- the context manager


def test_the_store_closes_itself_when_used_as_a_context_manager():
    """An indexer that raises mid-publish must not leave the file handle open."""
    path = Path(tempfile.mkdtemp()) / "ctx.db"
    with SQLiteKnowledgeRepository(path) as repo:
        repo.publish([doc()], [ver()], [chunk()], snap())
        assert repo.snapshot() is not None
    assert path.exists()


# ------------------------------------------------------------- orphan guards


def test_a_version_for_an_unknown_document_is_dropped():
    """Writing it anyway would attach history to whatever row held that id."""
    repo = fresh()
    repo.publish([doc()],
                 [ver(), ver(url="https://example/ghost", n=1)],
                 [], snap())
    assert repo.active_version("https://example/ghost") is None
    assert repo.active_version(URL) is not None


def test_a_chunk_whose_version_was_never_written_is_dropped():
    repo = fresh()
    repo.publish([doc()], [ver()],
                 [chunk(), chunk(i=1, version=99)], snap(chunks=2))
    hits = repo.retrieve(A, top_k=10)
    assert len(hits) == 1


def test_a_caveat_for_an_unknown_document_is_dropped():
    repo = fresh()
    repo.publish([doc()], [ver()], [chunk()], snap(),
                 caveats=[Caveat(URL, "temperature", "Above 5 C.", "Mixing"),
                          Caveat("https://example/ghost", "diy", "Nope.", "X")])
    assert len(repo.caveats(URL)) == 1
    assert repo.caveats("https://example/ghost") == []


# ------------------------------------------------------------ the cap and rank


def test_the_per_document_cap_stops_one_page_filling_the_answer():
    """A multi-source question must see several documents, not five slices of one."""
    repo = fresh()
    other = "https://example/duro"
    repo.publish(
        [doc(), doc(other, "Duro")],
        [ver(), ver(other)],
        [chunk(i=i) for i in range(6)] + [chunk(other, 0)],
        snap(docs=2, chunks=7),
    )
    hits = repo.retrieve(A, top_k=6, per_document_cap=2)
    from collections import Counter
    counts = Counter(h.chunk.canonical_url for h in hits)
    assert counts[URL] == 2
    assert other in counts


def test_the_authority_bonus_cannot_overturn_a_clearly_better_match():
    """Authority breaks near-ties; it must not promote an unrelated passage."""
    assert AUTHORITY_BONUS * 6 <= TIE_BAND + 1e-9


# ------------------------------------------------------------------ lookups


def test_the_active_version_of_an_unknown_document_is_none():
    """A citation for a document that is not indexed must return nothing, not raise."""
    repo = fresh()
    repo.publish([doc()], [ver()], [chunk()], snap())
    assert repo.active_version("https://example/never-seen") is None


def test_a_document_with_no_active_version_reports_none():
    repo = fresh()
    repo.publish([doc()], [ver(active=False)], [], snap())
    assert repo.active_version(URL) is None


def test_counts_reports_every_table_the_ingestion_report_prints():
    """The report must not silently omit a table that failed to write."""
    repo = fresh()
    repo.publish([doc()], [ver()], [chunk()], snap(),
                 caveats=[Caveat(URL, "temperature", "Above 5 C.", "Mixing")])
    counts = repo.counts()
    assert counts["documents"] == 1
    assert counts["document_versions"] == 1
    assert counts["chunks"] == 1
    assert counts["document_caveats"] == 1
    assert counts["index_snapshots"] == 1
    assert "excluded_documents" in counts


# ------------------------------------------------------------- vector packing


def test_a_vector_survives_the_round_trip_through_the_blob_column():
    """SQLite has no vector type, so the packing is ours and has to be exact."""
    restored = _unpack(_f32(A))
    assert len(restored) == DIMS
    assert abs(float(restored[0]) - A[0]) < 1e-6


def test_a_chunk_with_no_embedding_is_stored_but_never_retrieved():
    """An unembedded passage is a build fault, and searching it would be worse."""
    repo = fresh()
    bare = chunk()
    bare.embedding = []
    repo.publish([doc()], [ver()], [bare], snap())
    assert repo.counts()["chunks"] == 1
    assert repo.retrieve(A, top_k=5) == []


def test_a_zero_length_query_vector_does_not_divide_by_zero():
    """A failed embedding can return zeros, and a crash here would lose the answer."""
    repo = fresh()
    repo.publish([doc()], [ver()], [chunk()], snap())
    hits = repo.retrieve([0.0] * DIMS, top_k=5)
    assert all(h.score == 0.0 for h in hits)
