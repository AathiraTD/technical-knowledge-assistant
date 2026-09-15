"""The domain model.

These types are the contract between the indexing path and the answering path,
and they are deliberately storage-agnostic: the same fields are columns in
PostgreSQL and columns in SQLite. Changing the substrate does not change the
model, which is the whole point of the repository boundary.

Version semantics matter more here than they look. A datasheet whose coverage
changes from 16-20 to 14-18 must not have both figures retrievable at once, so
a document has many versions and exactly one is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Authority order for conflict resolution. Lower wins; recency breaks ties
# *within* a type, so a current datasheet still outranks a newer FAQ entry.
AUTHORITY = {
    "datasheet": 1,
    "system_guide": 2,
    "product_page": 3,
    "knowledge_base": 4,
    "faq": 5,
    "commercial": 6,
}

AUDIENCES = ("public", "trade", "staff")


@dataclass
class Document:
    """A source document, identified by its canonical URL."""

    canonical_url: str
    title: str
    document_type: str  # datasheet | product_page | knowledge_base | faq | system_guide | commercial
    authority: int
    audience: str = "public"
    product: str = ""
    link_text: str = ""  # how the site labelled the link; how a citation names it
    active_version: int = 1

    @property
    def citation_name(self) -> str:
        """What a person would call this document, for the Sources list."""
        return self.link_text or self.title or self.canonical_url


@dataclass
class DocumentVersion:
    """One fetched state of a document. Exactly one version is active."""

    canonical_url: str
    version: int
    content_hash: str
    source_path: str
    etag: str = ""
    source_last_modified: str = ""
    first_seen_at: str = ""
    fetched_at: str = ""
    checked_at: str = ""
    is_active: bool = True
    supersedes: str | None = None

    # Extraction quality, recorded so the ingestion report can be honest about
    # documents whose text came out badly rather than silently indexing them.
    extraction_quality: str = "unknown"  # clean | partial | failed | unknown
    notes: str = ""


@dataclass
class Chunk:
    """A retrievable passage: a document section, or a bullet kept whole.

    The unit of retrieval and the unit of citation are the same thing on
    purpose — "the Mixing section of the Solo datasheet" is something a
    plasterer can check, which a 512-token window is not.
    """

    canonical_url: str
    version: int
    chunk_index: int
    section: str
    content: str
    audience: str = "public"
    product: str = ""
    document_type: str = ""
    authority: int = 9
    source_date: str = ""
    embedding: list[float] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.canonical_url}#v{self.version}-{self.chunk_index}"


@dataclass
class Retrieved:
    """A chunk with its score, as returned by retrieval."""

    chunk: Chunk
    score: float
    document: Document


@dataclass
class Snapshot:
    """An index build. Answers record which snapshot produced them.

    The embedding model and chunking version are recorded because an index
    built with one model and queried with another returns confident nonsense —
    the engine refuses to run against a mismatch rather than degrade quietly.
    """

    snapshot_id: str
    created_at: str
    embedding_model: str
    embedding_dimensions: int
    chunking_version: str
    document_count: int
    chunk_count: int
    notes: dict[str, Any] = field(default_factory=dict)
