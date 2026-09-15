"""The storage boundary.

The answer engine depends on this Protocol, never on a database. Two adapters
implement it:

  EmbeddedRepository  SQLite — ships with the submission, runs offline, no service
  PostgresRepository  PostgreSQL + pgvector — the deployment target

The schema and the version semantics are the same in both; only the dialect and
the similarity operator differ. That is what makes the demonstration path and
the production path the same system rather than two systems that resemble each
other.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .model import (
    Caveat,
    CrawlRun,
    Chunk,
    Document,
    DocumentVersion,
    DocumentUpdate,
    Excluded,
    Retrieved,
    Snapshot,
)


@runtime_checkable
class KnowledgeRepository(Protocol):
    """What the answer engine needs from storage, and nothing more."""

    # ---- indexing path ----------------------------------------------------

    def publish(
        self,
        documents: list[Document],
        versions: list[DocumentVersion],
        chunks: list[Chunk],
        snapshot: Snapshot,
        caveats: list[Caveat] | None = None,
        excluded: list[Excluded] | None = None,
    ) -> str:
        """Write a complete index and activate it atomically.

        Either the whole snapshot becomes live or none of it does. A query must
        never see a half-built index, and the previous version of a changed
        document must be deactivated in the same transaction that activates the
        new one — otherwise both are retrievable and the engine can blend them.

        Returns the snapshot id.
        """
        ...

    def apply_delta(
        self,
        updates: list[DocumentUpdate],
        removed: list[str],
        snapshot: Snapshot,
        excluded: list[Excluded] | None = None,
        crawl_run: CrawlRun | None = None,
    ) -> str:
        """Apply only what changed, and keep what it replaced.

        This is the difference between a pipeline that rebuilds and one that
        maintains. `publish` replaces the whole index, which is right for a
        first build and wrong for a site that changes one datasheet: it throws
        away the history that makes an old answer explicable.

        For each update, the document's currently active version is deactivated
        and the new one activated **in the same transaction**, with the old
        version retained. Documents named in `removed` have every version
        deactivated but nothing deleted, because a datasheet withdrawn from the
        site is a fact worth keeping rather than a row worth losing. Documents
        appearing in neither list are not touched at all, which is the whole
        point: unchanged means no work.

        Returns the snapshot id.
        """
        ...

    def active_content_hashes(self) -> dict[str, str]:
        """Canonical URL to the content hash of its live version.

        What the indexer diffs the crawl against to decide new, changed,
        unchanged and removed. Reading it from the store rather than from a
        local file means the decision is made against what is actually being
        served.
        """
        ...

    def versions(self, canonical_url: str) -> list[DocumentVersion]:
        """Every version of a document, newest first, active flag included.

        The audit answer to "why did it say that six months ago?" — without
        this, version history is a schema feature nothing can read.
        """
        ...

    def crawl_runs(self, limit: int = 10) -> list[CrawlRun]:
        """Recent crawl runs, newest first."""
        ...

    # ---- answering path ---------------------------------------------------

    def retrieve(
        self,
        query_embedding: list[float],
        audiences: tuple[str, ...] = ("public",),
        top_k: int = 5,
        per_document_cap: int = 3,
    ) -> list[Retrieved]:
        """Nearest chunks, filtered to the caller's audience set.

        The audience filter is applied in the query, never by prompt: a prompt
        instruction is not an access control. The per-document cap exists so a
        multi-source question sees several documents rather than five chunks of
        one page.
        """
        ...

    def document(self, canonical_url: str) -> Document | None:
        """One document by URL, for citations and document requests."""
        ...

    def manifest(self, audiences: tuple[str, ...] = ("public",)) -> list[Document]:
        """Every active document the caller may see.

        Answers "which datasheet do I send, and is it current?" without
        retrieval, and carries the name lists the real-names check needs.
        """
        ...

    def caveats(self, canonical_url: str) -> list[Caveat]:
        """The qualifying sentences held against a document.

        Appended by code wherever any chunk of that document is printed or
        composed over, so a temperature limit travels with the instruction it
        qualifies even when the two sit in different sections — decision 11.
        """
        ...

    def excluded(self) -> list[Excluded]:
        """Documents deliberately not indexed, with the reason.

        So that "why isn't the safety data sheet in here?" has an answer that
        is on file rather than improvised.
        """
        ...

    def snapshot(self) -> Snapshot | None:
        """The active snapshot: embedding model, chunking version, counts.

        The engine compares its configured embedding model against this and
        refuses to run on a mismatch.
        """
        ...

    def active_version(self, canonical_url: str) -> DocumentVersion | None:
        """The live version of a document, for freshness and change detection."""
        ...


class IndexMismatch(RuntimeError):
    """The index was built with a different model or chunking version.

    Raised rather than handled: querying an index with a different embedding
    model than built it returns results that look plausible and are not.
    """
