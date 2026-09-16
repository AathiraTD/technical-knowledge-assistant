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

from dataclasses import dataclass
from typing import ContextManager, Protocol, runtime_checkable

from .model import (
    AnswerLogEntry,
    Caveat,
    CrawlRun,
    Chunk,
    Document,
    DocumentVersion,
    DocumentUpdate,
    Excluded,
    Retrieved,
    Snapshot,
    TraceSpan,
)


# ---------------------------------------------------------------- the request
#
# Retrieval used to take a vector and four numbers, which meant it knew about
# similarity, audience, active version, authority and the per-document cap — and
# nothing about the product the caller had *named*. A question saying "Lime Green
# Ultra" could be answered from WarmShell passages, because both are about old
# walls and internal insulation and the corpus is semantically dense there. The
# metadata to prevent that was already on every chunk; retrieval simply had no
# way to be told.
#
# `RetrievalRequest` is that way. It is a structured ask rather than a longer
# argument list, so an adapter can grow a new constraint without every call site
# changing, and so the engine can express "this question names a product"
# without expressing any SQL.

# How much similarity an explicitly named product is worth.
#
# **Boost, not filter.** A hard product filter is the obvious move and it is
# wrong here: the brief's own multi-source questions legitimately span a product
# page and a datasheet, and sometimes a second product entirely ("can I use
# Ultra over Solo?"). Filtering to the named product would refuse those, and
# would do it silently, which is the worst shape of failure this design has.
#
# So the named product is a bounded bonus of the same shape as the authority
# bonus above it — large enough that a marginal similarity difference cannot
# overturn it, small enough that a passage which is genuinely about something
# else does not get dragged into the top five by the mention of a name.
#
# It is deliberately an order of magnitude above `TIE_BAND` (0.02): authority
# nudges between passages that already matched about equally, whereas a named
# product is an explicit instruction from the user and has to beat a real
# semantic gap — the reviewer's case is an Ultra passage that scores a few
# hundredths *below* a WarmShell one. Because every chunk of the named product
# receives the same bonus, authority and recency still order them among
# themselves: the boost moves a group, it does not flatten it.
PRODUCT_BAND = 0.15


def product_matches(named: str, candidate: str) -> bool:
    """Whether a chunk's product is the one the caller named.

    Deliberately loose and deliberately simple, because it drives a boost and
    not a filter: a false positive costs a small reordering, and the rule has to
    be expressible identically in NumPy and in SQL or the two adapters are not
    one system. Case-insensitive containment either way, so "Ultra" matches a
    chunk tagged "Lime Green Ultra" and a question naming "Lime Green Ultra"
    matches a chunk tagged "Ultra". An untagged chunk matches nothing.
    """
    n, c = named.strip().lower(), candidate.strip().lower()
    if not n or not c:
        return False
    return n in c or c in n


# Retention for `turn_traces`, as settled in
# docs/conversation-observability-review.md §8.1. Here rather than in either
# adapter, because a window that differed between the assessment path and the
# deployment path would make "the two adapters are one system" false about the
# one thing an operator would notice.
#
# Fourteen days is the policy: a trace is for debugging the answer somebody is
# asking about this week. The row cap is the backstop — measured against the
# audit table's own rate, a heavy development day is about two thousand span
# rows, so 200,000 is roughly three months and binds only when something is
# wrong. The stride keeps the delete off the answering path; both adapters
# expose it as an instance attribute so a test can prove the sweep without
# writing two hundred batches to reach it.
TRACE_RETENTION_DAYS = 14
TRACE_ROW_CAP = 200_000
TRACE_PRUNE_STRIDE = 200


@dataclass(frozen=True)
class RetrievalRequest:
    """What a caller wants retrieved, as a structure rather than a signature."""

    embedding: list[float]
    audiences: tuple[str, ...] = ("public",)
    top_k: int = 5
    per_document_cap: int = 3

    # The product the question named outright, if it named one. Empty means the
    # question named none, and retrieval behaves exactly as it did before.
    product: str = ""


@runtime_checkable
class KnowledgeRepository(Protocol):
    """What the answer engine needs from storage, and nothing more."""

    # ---- indexing path ----------------------------------------------------

    def read_snapshot(self) -> ContextManager[Snapshot | None]:
        """Keep every read for one answer on the same committed release."""
        ...

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

    def retrieve_for(self, request: RetrievalRequest) -> list[Retrieved]:
        """The same retrieval, told what the question actually asked for.

        Every guarantee `retrieve` makes holds here unchanged — active version
        only, audience filtered before ranking, per-document cap, authority
        beating a marginal similarity difference. What is added is that a
        product named in the question boosts its own passages by `PRODUCT_BAND`,
        so a semantically adjacent product cannot displace the one the user
        asked about. It is a boost and not a filter, for the reason recorded
        beside the constant.
        """
        ...

    def find_passages(
        self,
        product: str,
        terms: tuple[str, ...],
        audiences: tuple[str, ...] = ("public",),
        limit: int = 3,
    ) -> list[Retrieved]:
        """Targeted lookup: the passage that carries a property, by metadata.

        Semantic top-five is the wrong instrument for "how many bags do I need".
        The coverage figure lives in one short section that a natural-language
        question about square metres need not rank first, and the calculation
        path cannot recover from that because it only re-sorts what it was
        given. This is the recovery: given the product and the words the
        property is printed under, return the active, audience-filtered passages
        of that product whose section or text carries one of them, ordered by
        authority and then by recency within it.

        Lexical and deterministic on purpose. The returned `score` is 0.0 and is
        **not** a similarity — nothing here was embedded, and a caller must not
        compare it against the abstention threshold. What these hits carry
        instead is a stronger guarantee than a score: the asked-for term is
        provably present in the passage, which is what the relevance gate exists
        to establish.
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

    # ---- audit ------------------------------------------------------------

    def log_answer(self, entry: AnswerLogEntry) -> None:
        """Record which snapshot, route and passages produced one answer.

        `versions` explains what the store held; this explains what an answer
        used. The pair is the whole traceability claim: given a logged entry,
        the snapshot names the embedding model and the active document versions,
        the chunk ids name the passages, and the route names how they were
        turned into a reply. A question answered six months ago can then be
        reconstructed rather than guessed at.

        **It commits in its own transaction, independently of any snapshot read
        in progress.** A read snapshot holds a transaction that is rolled back
        when the answer finishes — read-only in PostgreSQL, rolled back in
        SQLite — so an insert made inside it would error or be silently
        discarded. The adapters therefore write on their own connection, which
        is what makes it safe for the call site to log from inside the snapshot
        it is describing. Do not move the write onto the reading connection to
        "tidy it up": that reintroduces exactly the bug this arrangement avoids.

        What it deliberately does not do: store the generated answer text. The
        route, the evidence and the model are what make a reply explicable, and
        retaining the prose of every conversation indefinitely is a privacy
        decision nobody has asked for. It also does not swallow failures — a
        store that cannot write its audit row says so rather than answering as
        though it had.
        """
        ...

    def answer_log(self, limit: int = 20) -> list[AnswerLogEntry]:
        """Recent answers, newest first, capped at `limit`.

        The read side of the audit trail, and the reason the write side is
        worth having: refusal rates and route mix are countable from these rows
        rather than estimated. Returns an empty list when nothing has been
        answered yet, which is a state rather than an error.
        """
        ...


    # ---- traces -----------------------------------------------------------

    def record_spans(self, spans: list[TraceSpan]) -> None:
        """Persist one turn's spans, and prune the table while it is open.

        The debugging half of the pair whose audit half is `log_answer`: that
        one says what an answer used, this one says how it got there and what
        each stage cost. The two stay separate tables because merging them would
        put timings into an audit trail and audit semantics into a debugging
        one, and because only one of them is allowed to be deleted from.

        **It writes on its own connection, outside any read snapshot**, for
        exactly the reason `log_answer` does: a read snapshot is rolled back in
        SQLite and `REPEATABLE READ READ ONLY` in PostgreSQL, so an insert made
        inside one is silently discarded or raises and takes the answer's
        remaining reads with it. Do not move this onto the reading connection.

        **It must not raise.** A failure here degrades observability and nothing
        else — the call site has a perfectly good answer in hand, and losing it
        to a tracing write would invert the priority the whole design rests on.
        The adapters therefore swallow, and say so on the event stream rather
        than silently, which is the same bargain `Assistant._log` makes.

        **Retention is enforced here**, on write, rather than on a schedule.
        Two bounds: `TRACE_RETENTION_DAYS` is the stated policy — a trace is for
        debugging the answer somebody is asking about this week — and
        `TRACE_ROW_CAP` is the backstop that makes the policy safe against a
        burst the window cannot see, such as a load test or a retry loop inside
        the fourteen days. A window alone does not bound a burst; a cap alone
        redefines retention as "however long two hundred thousand rows happen to
        last", which is not a number anyone can answer a colleague with.

        Pruning where the table is written follows `SessionStore`, which sweeps
        expired entries from `open()` rather than from a timer, and `AnswerCache`,
        which evicts at the point of insertion. It is amortised over
        `TRACE_PRUNE_STRIDE` writes so the delete stays off the answering path:
        a `DELETE ... WHERE started_at < ?` over an indexed column costs nothing
        when it matches nothing, but it is still a write per answer if it runs
        every time.

        Three costs, stated rather than hidden. A quiet system keeps rows past
        the window until something writes again, exactly as an idle
        `SessionStore` does — the bound is on growth, not on age. A burst can
        overshoot the cap between strides. And after the window an `answer_log`
        row outlives its trace, so "why this answer?" degrades to the older,
        weaker "what did this answer use".

        Deliberately not done: `answer_log` gains no prune. It retains question
        text, so deleting from it is a data-retention decision with a privacy
        argument attached that nobody has taken. Bounding one table and not the
        other is the honest split, and it leaves `answer_log`'s own unbounded
        growth as a stated open item rather than a solved one.
        """
        ...

    def traces(self, trace_id: str = "", session_id: str = "",
               limit: int = 1000) -> list[TraceSpan]:
        """Spans for one answer or one conversation, oldest first.

        Oldest first because the caller is reconstructing a tree and reading it
        in the order the stages ran, which is the opposite of `answer_log`'s
        newest-first: that one is a feed, this one is a recording.

        The two filters are the two indexed access paths of §2.3 and nothing
        else, which is the point — `trace_id` reconstructs one answer and
        `session_id` replays one conversation. Both empty returns the most
        recent spans, which is a developer convenience rather than an access
        path; `limit` bounds it either way.
        """
        ...


class PublicationBusy(RuntimeError):
    """Another indexer holds the publication lock and would not let go.

    Defined here rather than beside the PostgreSQL adapter because the indexer
    has to catch it, and the indexer must not import a database driver to do so
    — the whole point of the repository boundary. SQLite reaches the same
    situation through its own busy timeout; the name a caller handles is one
    either adapter can raise.
    """


class IndexMismatch(RuntimeError):
    """The index was built with a different model or chunking version.

    Raised rather than handled: querying an index with a different embedding
    model than built it returns results that look plausible and are not.
    """
