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
        """What a person would call this document, for the Sources list.

        Page titles on the site carry a pipe-separated tail for search engines —
        "Solo Onecoat Lime Plaster |Lime Plaster | Lime Green". A citation has to
        be checkable by a plasterer, so only the part before the first pipe is
        used, and the link text wins over the title when the site gave one.
        """
        name = self.link_text or self.title or self.canonical_url
        return name.split("|")[0].strip() or self.canonical_url


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
class Caveat:
    """A qualifying sentence, held against the document rather than a chunk.

    Fine Stuff states its 8 °C limit under Mixing and again under Curing while
    the steps a user asks about sit elsewhere, so no chunk boundary keeps the
    limit with the instruction it qualifies. Holding caveats at document level
    and appending them by code does — decision 11.
    """

    canonical_url: str
    caveat_type: str  # temperature | diy | incompatibility | other
    sentence: str
    section: str = ""


@dataclass
class Excluded:
    """A document deliberately not indexed, and why.

    Kept so that "why isn't the safety data sheet in here?" has an answer on
    file rather than an improvised one.
    """

    url: str
    reason: str
    link_text: str = ""


@dataclass
class DocumentUpdate:
    """One document's new state, as an indexing run wants to apply it.

    Grouped rather than passed as four parallel lists because they have to move
    together: a version, the chunks cut from it, and the caveats tagged on it
    are one unit of work, and applying two of the three is a corrupt index.
    """

    document: Document
    version: DocumentVersion
    chunks: list["Chunk"] = field(default_factory=list)
    caveats: list["Caveat"] = field(default_factory=list)


@dataclass
class CrawlRun:
    """What one pass over the site did, recorded rather than printed.

    A console line saying "94 unchanged" disappears when the terminal closes.
    The same fact in a row is evidence that a second crawl reprocessed nothing,
    which is the claim the delta pipeline exists to support.
    """

    started_at: str
    completed_at: str = ""
    documents_checked: int = 0
    documents_new: int = 0
    documents_changed: int = 0
    documents_unchanged: int = 0
    documents_removed: int = 0
    documents_failed: int = 0
    snapshot_id: str = ""

    @property
    def reprocessed(self) -> int:
        return self.documents_new + self.documents_changed


@dataclass
class AnswerLogEntry:
    """One answered question, recorded so the answer stays explicable.

    The audit chain runs documents → versions → chunks → snapshot, and this is
    its last link. Without it the store can say what the index held on a date
    and cannot say which part of it an answer actually used, which is the half
    of "why did it say that?" that matters. The route, the snapshot id and the
    chunk ids together are enough to reconstruct the evidence a reply was built
    from, because generation is deterministic against them.

    A refusal is worth logging for the same reason as an answer: `check_failed`
    names the check that stopped it, so over-refusal is measurable rather than
    anecdotal.

    `asked_at` may be left empty, in which case the store timestamps it.

    `source` names the surface that asked, and it exists because leaving it out
    was a measurable defect rather than a tidiness problem. The evaluation
    harness answers through the same `Assistant` as a person does, with logging
    on by default, so its questions landed in this table indistinguishable from
    real ones — and the harness's question set is deliberately loaded with the
    near-miss and far-miss probes that are *supposed* to refuse. Any refusal
    rate counted from these rows therefore counted the probes as failures of
    the system rather than as successes of the guardrail. The fix is to record
    who asked, not to stop the harness logging: dropping its rows would lose
    the only evidence of how the probes actually routed.

    The default is `unknown` rather than `web` or `cli`, because a row written
    before this column existed genuinely does not record which surface produced
    it, and guessing would make the contamination invisible instead of
    countable.
    """

    question: str
    path_taken: str  # route | extract | compose | defer | refuse
    audiences: tuple[str, ...] = ("public",)
    snapshot_id: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    generation_model: str = ""
    check_failed: str = ""
    asked_at: str = ""
    source: str = "unknown"  # cli | web | evaluation | unknown


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
