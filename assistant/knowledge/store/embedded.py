"""The assessment adapter: SQLite plus NumPy.

One file, no service, no container, standard library apart from the cosine.
That is the whole reason it exists — an assessor clones the repository and runs
one command, on a machine that has nothing installed and possibly no network.

It is not a toy. It carries the same eight tables as the PostgreSQL adapter,
the same column names and the same version semantics, and it enforces "exactly
one active version per document" with a partial unique index rather than with
application code that remembers to. Three things differ, all forced by SQLite:
embeddings are BLOBs and cosine runs in NumPy after loading; IDENTITY becomes
AUTOINCREMENT; JSON lives in TEXT.

At the corpus size that is not a compromise. Eight hundred chunks of 1024
float32s is about 3 MB, and one matrix multiply against it costs under a
millisecond — the network hop to embed the question is a thousand times the
cost of searching with it.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..model import (
    AUTHORITY,
    AnswerLogEntry,
    Caveat,
    CrawlRun,
    Chunk,
    Document,
    DocumentUpdate,
    DocumentVersion,
    Excluded,
    Retrieved,
    Snapshot,
    TraceSpan,
)
from ..repository import (
    PRODUCT_BAND,
    TRACE_PRUNE_STRIDE,
    TRACE_RETENTION_DAYS,
    TRACE_ROW_CAP,
    IndexMismatch,
    RetrievalRequest,
    product_matches,
)
from ... import paths

SCHEMA = paths.DB_DIR / "schema.sqlite.sql"

# How much similarity an authority rank is worth. Authority has to be able to
# act — no two cosine scores are ever exactly equal, so ranking on similarity
# alone lets an FAQ paraphrase outrank the datasheet it paraphrases. But it must
# not overturn a clearly better match either.
#
# So authority is a bounded bonus rather than a tie-break, because a tie-break
# needs a definition of "tied" and every bucketing definition is sensitive to
# where the bucket edge falls: scores of 1.00 and 0.99 are plainly tied and
# still land in different buckets of width 0.02. The contract test caught
# exactly that. A continuous bonus has no edges.
#
# The bonus is capped at TIE_BAND across the whole authority range, so the
# strongest possible authority preference — datasheet over commercial page —
# can only reorder passages already within TIE_BAND of each other.
TIE_BAND = 0.02
_AUTHORITY_SPREAD = max(len(AUTHORITY), 1)
AUTHORITY_BONUS = TIE_BAND / _AUTHORITY_SPREAD


def _now() -> str:
    """The house timestamp: UTC, to the second, as the rest of the pipeline writes it."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _f32(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class SQLiteKnowledgeRepository:
    """`KnowledgeRepository` over a single SQLite file."""

    def __init__(self, path: str | Path = "data/index/knowledge.db",
                 *, check_same_thread: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A connection is bound to its opening thread unless told otherwise.
        # The threaded server opens the store once and answers from request
        # threads, so it passes False and wraps this in LockedRepository; the
        # default stays True so a single-threaded caller keeps SQLite's own
        # guard rather than losing it silently.
        self.db = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA.read_text(encoding="utf-8"))
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(crawl_runs)")}
        for name, definition in (("documents_removed", "INTEGER NOT NULL DEFAULT 0"),
                                 ("snapshot_id", "TEXT NOT NULL DEFAULT ''")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE crawl_runs ADD COLUMN {name} {definition}")
        # Additive in the same way and for the same reason: `CREATE TABLE IF NOT
        # EXISTS` does not reshape a table that already exists, so a database
        # built before `answer_log.source` would fail on the next insert rather
        # than gain the column. Existing rows take the default, `unknown`, which
        # is the truth about them — nobody recorded which surface asked.
        logged = {r[1] for r in self.db.execute("PRAGMA table_info(answer_log)")}
        if "source" not in logged:
            self.db.execute(
                "ALTER TABLE answer_log ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
        # `turn_traces` itself arrives with the schema script above, which runs
        # on every open: a database created before the table simply gains it,
        # because `CREATE TABLE IF NOT EXISTS` creates what is missing. What it
        # does *not* do is reshape a table that already exists, which is why
        # `source` is migrated by hand here exactly as `answer_log.source` is —
        # a store written by an earlier iteration of this slice would otherwise
        # fail on its next span insert.
        traced = {r[1] for r in self.db.execute("PRAGMA table_info(turn_traces)")}
        if traced and "source" not in traced:
            self.db.execute(
                "ALTER TABLE turn_traces ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
        self.db.commit()
        # Retention, per docs/conversation-observability-review.md §8.1. Held as
        # instance attributes rather than read from the module constants at the
        # point of use, so a test can prove the sweep actually prunes without
        # writing two hundred batches and waiting fourteen days to reach it.
        self.trace_retention_days = TRACE_RETENTION_DAYS
        self.trace_row_cap = TRACE_ROW_CAP
        self.trace_prune_stride = TRACE_PRUNE_STRIDE
        self._trace_writes = 0
        self._matrix: np.ndarray | None = None
        self._rows: list[sqlite3.Row] = []

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def read_snapshot(self):
        """Pin metadata, evidence and caveats to one committed release."""
        self.db.execute("BEGIN")
        try:
            yield self.snapshot()
        finally:
            self.db.rollback()

    def __enter__(self) -> "SQLiteKnowledgeRepository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ indexing

    def publish(
        self,
        documents: list[Document],
        versions: list[DocumentVersion],
        chunks: list[Chunk],
        snapshot: Snapshot,
        caveats: list[Caveat] | None = None,
        excluded: list[Excluded] | None = None,
    ) -> str:
        """Write a whole index and activate it in one transaction.

        A query must never see a half-built index, so activation is the last
        statement and everything rolls back together if any part of it fails.
        """
        caveats = caveats or []
        excluded = excluded or []
        cur = self.db.cursor()
        try:
            cur.execute("BEGIN")

            # A publish replaces the index; it does not merge into it. Deleting
            # the previous rows here is what makes a rebuild reproducible rather
            # than dependent on what happened to be there already.
            for table in ("chunks", "document_caveats", "document_versions",
                          "documents", "excluded_documents"):
                cur.execute(f"DELETE FROM {table}")
            cur.execute("UPDATE index_snapshots SET is_active = 0")

            doc_ids: dict[str, int] = {}
            for d in documents:
                cur.execute(
                    """INSERT INTO documents
                       (canonical_url, title, link_text, document_type, authority,
                        audience, product, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (d.canonical_url, d.title, d.link_text, d.document_type,
                     d.authority, d.audience, d.product, snapshot.created_at),
                )
                doc_ids[d.canonical_url] = cur.lastrowid

            version_ids: dict[tuple[str, int], int] = {}
            for v in versions:
                did = doc_ids.get(v.canonical_url)
                if did is None:
                    continue
                cur.execute(
                    """INSERT INTO document_versions
                       (document_id, version_number, content_hash, etag,
                        source_last_modified, source_path, extraction_quality,
                        notes, first_seen_at, fetched_at, checked_at, is_active)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (did, v.version, v.content_hash, v.etag, v.source_last_modified,
                     v.source_path, v.extraction_quality, v.notes, v.first_seen_at,
                     v.fetched_at, v.checked_at, 1 if v.is_active else 0),
                )
                vid = cur.lastrowid
                version_ids[(v.canonical_url, v.version)] = vid
                if v.is_active:
                    cur.execute("UPDATE documents SET active_version_id = ? WHERE id = ?",
                                (vid, did))

            for c in chunks:
                vid = version_ids.get((c.canonical_url, c.version))
                if vid is None:
                    continue
                cur.execute(
                    """INSERT INTO chunks
                       (document_version_id, chunk_index, section, content,
                        audience, product, source_date, embedding)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (vid, c.chunk_index, c.section, c.content, c.audience,
                     c.product, c.source_date,
                     _f32(c.embedding) if c.embedding else None),
                )

            for cav in caveats:
                vid = next((v for (u, _n), v in version_ids.items()
                            if u == cav.canonical_url), None)
                if vid is None:
                    continue
                cur.execute(
                    """INSERT INTO document_caveats
                       (document_version_id, caveat_type, sentence, section)
                       VALUES (?,?,?,?)""",
                    (vid, cav.caveat_type, cav.sentence, cav.section),
                )

            for ex in excluded:
                cur.execute(
                    "INSERT INTO excluded_documents (url, link_text, reason) VALUES (?,?,?)",
                    (ex.url, ex.link_text, ex.reason),
                )

            cur.execute(
                """INSERT OR REPLACE INTO index_snapshots
                   (id, created_at, embedding_model, embedding_dimensions,
                    chunking_version, document_count, chunk_count, is_active, notes)
                   VALUES (?,?,?,?,?,?,?,1,?)""",
                (snapshot.snapshot_id, snapshot.created_at, snapshot.embedding_model,
                 snapshot.embedding_dimensions, snapshot.chunking_version,
                 snapshot.document_count, snapshot.chunk_count,
                 json.dumps(snapshot.notes, ensure_ascii=False)),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        self._matrix = None
        return snapshot.snapshot_id

    # -------------------------------------------------------------- delta

    def apply_delta(
        self,
        updates: list[DocumentUpdate],
        removed: list[str],
        snapshot: Snapshot,
        excluded: list[Excluded] | None = None,
        crawl_run: CrawlRun | None = None,
    ) -> str:
        """Apply only what changed, keeping what it replaced. One transaction.

        The ordering inside matters. The old version is deactivated *before*
        the new one is inserted, because the partial unique index permits
        exactly one active version per document and would otherwise refuse the
        insert. That refusal would be correct, which is the point: the database
        enforces the invariant rather than trusting this method to remember it.
        """
        cur = self.db.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            if "parent_snapshot" in snapshot.notes:
                current = self.snapshot()
                if snapshot.notes["parent_snapshot"] != (current.snapshot_id if current else None):
                    raise IndexMismatch("Another indexer published first; retry against the current release")

            for update in updates:
                self._apply_one(cur, update, snapshot.created_at)

            for url in removed:
                row = cur.execute(
                    "SELECT id FROM documents WHERE canonical_url = ?", (url,)
                ).fetchone()
                if row is None:
                    continue
                # Deactivate, never delete. A datasheet withdrawn from the site
                # is a fact about the site, and an answer given while it was
                # live still has to be explicable afterwards.
                cur.execute(
                    "UPDATE document_versions SET is_active = 0 WHERE document_id = ?",
                    (row["id"],))
                cur.execute(
                    "UPDATE documents SET active_version_id = NULL WHERE id = ?",
                    (row["id"],))

            if excluded is not None:
                cur.execute("DELETE FROM excluded_documents")
                for ex in excluded:
                    cur.execute(
                        "INSERT INTO excluded_documents (url, link_text, reason) "
                        "VALUES (?,?,?)", (ex.url, ex.link_text, ex.reason))

            if crawl_run is not None:
                self._insert_crawl_run(cur, crawl_run, snapshot.snapshot_id)

            # The snapshot records the live index, counted here rather than
            # taken from the caller. After a delta the totals are a function of
            # what was already stored plus what just changed, and only the store
            # knows both halves.
            live_docs = cur.execute(
                "SELECT COUNT(*) FROM documents d "
                "JOIN document_versions v ON v.id = d.active_version_id"
            ).fetchone()[0]
            live_chunks = cur.execute(
                "SELECT COUNT(*) FROM chunks c "
                "JOIN document_versions v ON v.id = c.document_version_id "
                "WHERE v.is_active = 1"
            ).fetchone()[0]
            counted = replace(snapshot, document_count=live_docs,
                              chunk_count=live_chunks)
            self._insert_snapshot(cur, counted)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        self._matrix = None
        return snapshot.snapshot_id

    def _apply_one(self, cur, update: DocumentUpdate, created_at: str) -> None:
        d, v = update.document, update.version

        row = cur.execute(
            "SELECT id, active_version_id FROM documents WHERE canonical_url = ?",
            (d.canonical_url,)).fetchone()

        if row is None:
            cur.execute(
                """INSERT INTO documents
                   (canonical_url, title, link_text, document_type, authority,
                    audience, product, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (d.canonical_url, d.title, d.link_text, d.document_type,
                 d.authority, d.audience, d.product, created_at))
            document_id, superseded = cur.lastrowid, None
        else:
            document_id, superseded = row["id"], row["active_version_id"]
            # The document's own metadata can change without its content
            # changing shape: a retitled page is still the same document.
            cur.execute(
                """UPDATE documents SET title = ?, link_text = ?, document_type = ?,
                          authority = ?, audience = ?, product = ? WHERE id = ?""",
                (d.title, d.link_text, d.document_type, d.authority,
                 d.audience, d.product, document_id))
            cur.execute(
                "UPDATE document_versions SET is_active = 0 WHERE document_id = ?",
                (document_id,))

        next_number = (cur.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM document_versions "
            "WHERE document_id = ?", (document_id,)).fetchone()[0])

        cur.execute(
            """INSERT INTO document_versions
               (document_id, version_number, content_hash, etag,
                source_last_modified, source_path, extraction_quality, notes,
                first_seen_at, fetched_at, checked_at, is_active, supersedes_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (document_id, next_number, v.content_hash, v.etag,
             v.source_last_modified, v.source_path, v.extraction_quality,
             v.notes, v.first_seen_at, v.fetched_at, v.checked_at, superseded))
        version_id = cur.lastrowid
        cur.execute("UPDATE documents SET active_version_id = ? WHERE id = ?",
                    (version_id, document_id))

        for c in update.chunks:
            cur.execute(
                """INSERT INTO chunks
                   (document_version_id, chunk_index, section, content,
                    audience, product, source_date, embedding)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (version_id, c.chunk_index, c.section, c.content, c.audience,
                 c.product, c.source_date,
                 _f32(c.embedding) if c.embedding else None))

        for cav in update.caveats:
            cur.execute(
                """INSERT INTO document_caveats
                   (document_version_id, caveat_type, sentence, section)
                   VALUES (?,?,?,?)""",
                (version_id, cav.caveat_type, cav.sentence, cav.section))

    @staticmethod
    def _insert_crawl_run(cur, run: CrawlRun, snapshot_id: str) -> None:
        cur.execute(
            """INSERT INTO crawl_runs
               (started_at, completed_at, documents_checked, documents_new,
                documents_changed, documents_unchanged, documents_failed,
                documents_removed, snapshot_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run.started_at, run.completed_at or run.started_at,
             run.documents_checked, run.documents_new, run.documents_changed,
             run.documents_unchanged, run.documents_failed,
             run.documents_removed, snapshot_id))

    @staticmethod
    def _insert_snapshot(cur, snapshot: Snapshot) -> None:
        members = cur.execute("""SELECT d.canonical_url, v.version_number,
            v.content_hash, v.source_path FROM documents d
            JOIN document_versions v ON v.id=d.active_version_id""").fetchall()
        notes = {**snapshot.notes, "active_versions": {
            r[0]: {"version": r[1], "content_hash": r[2], "source_path": r[3]} for r in members}}
        cur.execute("UPDATE index_snapshots SET is_active = 0")
        cur.execute(
            """INSERT OR REPLACE INTO index_snapshots
               (id, created_at, embedding_model, embedding_dimensions,
                chunking_version, document_count, chunk_count, is_active, notes)
               VALUES (?,?,?,?,?,?,?,1,?)""",
            (snapshot.snapshot_id, snapshot.created_at, snapshot.embedding_model,
             snapshot.embedding_dimensions, snapshot.chunking_version,
             snapshot.document_count, snapshot.chunk_count,
             json.dumps(notes, ensure_ascii=False)))

    def active_content_hashes(self) -> dict[str, str]:
        """What is live now, so the indexer can diff a crawl against it."""
        rows = self.db.execute(
            """SELECT d.canonical_url, v.content_hash
               FROM documents d
               JOIN document_versions v ON v.id = d.active_version_id"""
        ).fetchall()
        return {r["canonical_url"]: r["content_hash"] for r in rows}

    def versions(self, canonical_url: str) -> list[DocumentVersion]:
        rows = self.db.execute(
            """SELECT v.* FROM document_versions v
               JOIN documents d ON d.id = v.document_id
               WHERE d.canonical_url = ?
               ORDER BY v.version_number DESC""", (canonical_url,)).fetchall()
        return [
            DocumentVersion(
                canonical_url=canonical_url, version=r["version_number"],
                content_hash=r["content_hash"], source_path=r["source_path"],
                etag=r["etag"], source_last_modified=r["source_last_modified"],
                first_seen_at=r["first_seen_at"], fetched_at=r["fetched_at"],
                checked_at=r["checked_at"], is_active=bool(r["is_active"]),
                extraction_quality=r["extraction_quality"], notes=r["notes"])
            for r in rows
        ]

    def crawl_runs(self, limit: int = 10) -> list[CrawlRun]:
        rows = self.db.execute(
            "SELECT * FROM crawl_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            CrawlRun(
                started_at=r["started_at"], completed_at=r["completed_at"] or "",
                documents_checked=r["documents_checked"],
                documents_new=r["documents_new"],
                documents_changed=r["documents_changed"],
                documents_unchanged=r["documents_unchanged"],
                documents_failed=r["documents_failed"],
                documents_removed=r["documents_removed"], snapshot_id=r["snapshot_id"])
            for r in rows
        ]

    # ----------------------------------------------------------- answering

    def _load(self) -> None:
        """Load active chunk vectors once, normalised, so cosine is a dot product."""
        # A different process may have published since the last request. Read
        # current rows in one query instead of retaining vectors across releases.
        self._rows = self.db.execute(
            """SELECT c.id, c.chunk_index, c.section, c.content, c.audience,
                      c.product, c.source_date, c.embedding,
                      v.version_number, d.canonical_url, d.title, d.link_text,
                      d.document_type, d.authority, d.audience AS doc_audience
               FROM chunks c
               JOIN document_versions v ON v.id = c.document_version_id
               JOIN documents d         ON d.id = v.document_id
               WHERE v.is_active = 1 AND c.embedding IS NOT NULL
               ORDER BY d.canonical_url, c.chunk_index"""
        ).fetchall()
        if not self._rows:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        m = np.vstack([_unpack(r["embedding"]) for r in self._rows])
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        self._matrix = m / np.where(norms == 0, 1, norms)

    def retrieve(
        self,
        query_embedding: list[float],
        audiences: tuple[str, ...] = ("public",),
        top_k: int = 5,
        per_document_cap: int = 3,
    ) -> list[Retrieved]:
        """Nearest active chunks the caller may see, authority breaking near-ties.

        Kept as the plain four-argument form because most callers have nothing
        else to say; `retrieve_for` is the same search with a structured ask.
        """
        return self.retrieve_for(RetrievalRequest(
            embedding=query_embedding, audiences=audiences, top_k=top_k,
            per_document_cap=per_document_cap))

    def retrieve_for(self, request: RetrievalRequest) -> list[Retrieved]:
        """The search itself, told what the question named.

        The audience filter is applied to the rows, not to a prompt: a prompt
        instruction is a request, and an access control has to be a fact. It is
        applied *before* scoring, so a forbidden row is never ranked, never
        capped against and never returned — and the product boost cannot reach
        it either, because it is not in the candidate list to be boosted.
        """
        self._load()
        if self._matrix is None or self._matrix.size == 0:
            return []

        q = np.asarray(request.embedding, dtype=np.float32)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        scores = self._matrix @ q

        allowed = set(request.audiences)
        candidates = [
            (float(scores[i]), r) for i, r in enumerate(self._rows)
            if r["audience"] in allowed
        ]
        # Two stable passes. "Newest wins within a type" cannot be negated on a
        # date string the way a number can, so dates are sorted descending
        # first; the ranking sort is stable and preserves that order wherever
        # the effective scores are equal.
        candidates.sort(key=lambda t: t[1]["source_date"] or "", reverse=True)
        candidates.sort(key=lambda t: -self._effective(
            t[0], t[1]["authority"],
            product_matches(request.product, t[1]["product"] or "")))

        out, per_doc = [], {}
        for score, r in candidates:
            url = r["canonical_url"]
            if per_doc.get(url, 0) >= request.per_document_cap:
                continue
            per_doc[url] = per_doc.get(url, 0) + 1
            out.append(Retrieved(chunk=self._chunk(r), score=score,
                                 document=self._document(r)))
            if len(out) >= request.top_k:
                break
        return out

    def find_passages(
        self,
        product: str,
        terms: tuple[str, ...],
        audiences: tuple[str, ...] = ("public",),
        limit: int = 3,
    ) -> list[Retrieved]:
        """The passage that actually carries a property, found by metadata.

        No vector is involved, so `score` is 0.0 and means "not a similarity".
        """
        if not terms:
            return []
        marks = ",".join("?" * len(audiences))
        like = " OR ".join(
            ["(LOWER(c.content) LIKE ? OR LOWER(c.section) LIKE ?)"] * len(terms))
        params: list = list(audiences)
        for term in terms:
            pattern = f"%{term.strip().lower()}%"
            params += [pattern, pattern]
        rows = self.db.execute(
            f"""SELECT c.id, c.chunk_index, c.section, c.content, c.audience,
                       c.product, c.source_date, v.version_number,
                       d.canonical_url, d.title, d.link_text, d.document_type,
                       d.authority, d.audience AS doc_audience
                FROM chunks c
                JOIN document_versions v ON v.id = c.document_version_id
                JOIN documents d         ON d.id = v.document_id
                WHERE v.is_active = 1
                  AND c.audience IN ({marks})
                  AND ({like})
                ORDER BY d.authority ASC, c.source_date DESC,
                         d.canonical_url, c.chunk_index""",
            params,
        ).fetchall()
        # The product match is the one rule that must be identical in both
        # adapters, so it runs through the shared helper rather than through two
        # dialects of LIKE. An empty product means the caller named none and the
        # lookup is on the terms alone — which is the case for a question like
        # "why is my render patchy", where the evidence is an article rather
        # than a product document.
        matched = ([r for r in rows if product_matches(product, r["product"] or "")]
                   if product else rows)
        return [Retrieved(chunk=self._chunk(r), score=0.0,
                          document=self._document(r))
                for r in matched[:limit]]

    @staticmethod
    def _effective(similarity: float, authority: int,
                   named_product: bool = False) -> float:
        """Similarity, nudged by authority and by the product the caller named.

        A datasheet is preferred over a product page that matched marginally
        better; it is not preferred over one that matched clearly better. A
        passage about the product the question named is preferred over a
        semantic neighbour by a larger but still bounded margin. The score
        returned here orders results and is never shown — the similarity
        reported alongside an answer is the real cosine.
        """
        rank = max(1, min(authority, _AUTHORITY_SPREAD))
        boost = PRODUCT_BAND if named_product else 0.0
        return similarity + (_AUTHORITY_SPREAD - rank) * AUTHORITY_BONUS + boost

    @staticmethod
    def _chunk(r: sqlite3.Row) -> Chunk:
        return Chunk(
            canonical_url=r["canonical_url"], version=r["version_number"],
            chunk_index=r["chunk_index"], section=r["section"], content=r["content"],
            audience=r["audience"], product=r["product"],
            document_type=r["document_type"], authority=r["authority"],
            source_date=r["source_date"],
        )

    @staticmethod
    def _document(r: sqlite3.Row) -> Document:
        return Document(
            canonical_url=r["canonical_url"], title=r["title"],
            document_type=r["document_type"], authority=r["authority"],
            audience=r["doc_audience"], product=r["product"],
            link_text=r["link_text"],
        )

    def document(self, canonical_url: str) -> Document | None:
        r = self.db.execute(
            """SELECT canonical_url, title, link_text, document_type, authority,
                      audience AS doc_audience, product
               FROM documents WHERE canonical_url = ?""",
            (canonical_url,),
        ).fetchone()
        return self._document(r) if r else None

    def manifest(self, audiences: tuple[str, ...] = ("public",)) -> list[Document]:
        marks = ",".join("?" * len(audiences))
        rows = self.db.execute(
            f"""SELECT d.canonical_url, d.title, d.link_text, d.document_type,
                       d.authority, d.audience AS doc_audience, d.product
                FROM documents d
                JOIN document_versions v ON v.id = d.active_version_id
                WHERE d.audience IN ({marks})
                ORDER BY d.authority, d.canonical_url""",
            audiences,
        ).fetchall()
        return [self._document(r) for r in rows]

    def caveats(self, canonical_url: str) -> list[Caveat]:
        rows = self.db.execute(
            """SELECT dc.caveat_type, dc.sentence, dc.section
               FROM document_caveats dc
               JOIN document_versions v ON v.id = dc.document_version_id
               JOIN documents d         ON d.id = v.document_id
               WHERE d.canonical_url = ? AND v.is_active = 1""",
            (canonical_url,),
        ).fetchall()
        return [Caveat(canonical_url, r["caveat_type"], r["sentence"], r["section"])
                for r in rows]

    def excluded(self) -> list[Excluded]:
        rows = self.db.execute(
            "SELECT url, link_text, reason FROM excluded_documents ORDER BY url"
        ).fetchall()
        return [Excluded(r["url"], r["reason"], r["link_text"]) for r in rows]

    def snapshot(self) -> Snapshot | None:
        r = self.db.execute(
            "SELECT * FROM index_snapshots WHERE is_active = 1"
        ).fetchone()
        if not r:
            return None
        return Snapshot(
            snapshot_id=r["id"], created_at=r["created_at"],
            embedding_model=r["embedding_model"],
            embedding_dimensions=r["embedding_dimensions"],
            chunking_version=r["chunking_version"],
            document_count=r["document_count"], chunk_count=r["chunk_count"],
            notes=json.loads(r["notes"] or "{}"),
        )

    def active_version(self, canonical_url: str) -> DocumentVersion | None:
        r = self.db.execute(
            """SELECT v.*, d.canonical_url FROM document_versions v
               JOIN documents d ON d.id = v.document_id
               WHERE d.canonical_url = ? AND v.is_active = 1""",
            (canonical_url,),
        ).fetchone()
        if not r:
            return None
        return DocumentVersion(
            canonical_url=r["canonical_url"], version=r["version_number"],
            content_hash=r["content_hash"], source_path=r["source_path"],
            etag=r["etag"], source_last_modified=r["source_last_modified"],
            first_seen_at=r["first_seen_at"], fetched_at=r["fetched_at"],
            checked_at=r["checked_at"], is_active=bool(r["is_active"]),
            extraction_quality=r["extraction_quality"], notes=r["notes"],
        )

    # ------------------------------------------------------------------ audit

    def log_answer(self, entry: AnswerLogEntry) -> None:
        """Record one answer, on its own connection, committed on its own.

        Deliberately not on `self.db`. `read_snapshot()` opens a transaction
        there and rolls it back when the answer is finished, so a row inserted
        inside a snapshot read would be discarded at precisely the moment it
        was wanted — silently, because a rollback is not an error. A second
        connection to the same file commits independently, and the journal is
        WAL, so the write does not wait for the reader or disturb it.

        That is the property the call site depends on: logging may happen
        inside the snapshot it describes. Moving this onto the reading
        connection to save a handle would reintroduce the bug.
        """
        writer = sqlite3.connect(self.path)
        try:
            writer.execute("PRAGMA foreign_keys = ON")
            writer.execute(
                # The question is stored and the generated answer is not. The
                # route, the snapshot and the chunk ids are what make a reply
                # explicable; keeping the prose of every conversation
                # indefinitely is a privacy decision nobody has asked for.
                """INSERT INTO answer_log
                   (asked_at, question, audiences, path_taken, snapshot_id,
                    chunk_ids, generation_model, check_failed, source)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (entry.asked_at or _now(), entry.question,
                 json.dumps(list(entry.audiences)), entry.path_taken,
                 # Empty means "no snapshot was consulted", and the column is a
                 # foreign key: NULL is the only honest way to say that.
                 entry.snapshot_id or None,
                 json.dumps(list(entry.chunk_ids)),
                 entry.generation_model, entry.check_failed,
                 entry.source or "unknown"),
            )
            writer.commit()
        finally:
            writer.close()

    def answer_log(self, limit: int = 20) -> list[AnswerLogEntry]:
        """Recent answers, newest first."""
        rows = self.db.execute(
            """SELECT asked_at, question, audiences, path_taken, snapshot_id,
                      chunk_ids, generation_model, check_failed, source
               FROM answer_log ORDER BY asked_at DESC, id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            AnswerLogEntry(
                question=r["question"], path_taken=r["path_taken"],
                audiences=tuple(json.loads(r["audiences"])),
                snapshot_id=r["snapshot_id"] or "",
                chunk_ids=list(json.loads(r["chunk_ids"])),
                generation_model=r["generation_model"],
                check_failed=r["check_failed"], asked_at=r["asked_at"],
                source=r["source"])
            for r in rows
        ]

    # ----------------------------------------------------------------- traces

    def record_spans(self, spans: list[TraceSpan]) -> None:
        """Write one turn's spans, on a connection of its own, and never raise.

        The same second-connection arrangement as `log_answer`, for the same
        reason: `read_snapshot()` opens a transaction on `self.db` and rolls it
        back when the answer finishes, so rows inserted inside a snapshot read
        would be discarded at exactly the moment they were wanted — silently,
        because a rollback is not an error.

        It differs from `log_answer` in one way that matters: it swallows. A
        store that cannot write its audit row says so, because the audit trail
        is a promise to somebody. A store that cannot write a timing has cost
        the operator a debugging aid and the caller nothing, and failing an
        answer over it would invert the priority the whole design rests on. The
        failure is reported on the event stream rather than passed over in
        silence, which is the bargain `CLAUDE.md` asks for.

        The whole batch goes in one transaction. A half-written span tree is
        worse than none: it reads as stages that did not happen.
        """
        if not spans:
            return
        try:
            writer = sqlite3.connect(self.path)
            try:
                writer.executemany(
                    """INSERT INTO turn_traces
                       (session_id, turn_id, trace_id, span_id, parent_span_id,
                        name, started_at, duration_ms, status, source, attributes)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    [(s.session_id, s.turn_id, s.trace_id, s.span_id,
                      s.parent_span_id, s.name, s.started_at or _now(),
                      int(s.duration_ms), s.status or "ok",
                      s.source or "unknown",
                      json.dumps(s.attributes or {}, default=str))
                     for s in spans])
                self._prune_traces(writer)
                writer.commit()
            finally:
                writer.close()
        except Exception as error:                     # noqa: BLE001
            from ...infrastructure import observability as obs
            obs.event("store_error", operation="record_spans",
                      error=type(error).__name__, detail=str(error))

    def _prune_traces(self, writer) -> None:
        """Window then cap, on the stride, inside the caller's transaction.

        Amortised rather than run on every write. A `DELETE ... WHERE
        started_at < ?` over an indexed column costs nothing when it matches
        nothing, but it is still a write per answered question if it runs every
        time, and the answering path is where this is least welcome.

        The window is the stated policy and the cap is the backstop that makes
        it safe against a burst the window cannot see. The cap deletes by id
        rather than by timestamp because ids are monotonic here and timestamps
        are only to the second, so a cutoff by time could delete a whole
        second's worth of rows or none of it.
        """
        self._trace_writes += 1
        if self._trace_writes % max(self.trace_prune_stride, 1):
            return
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=self.trace_retention_days)
                  ).isoformat(timespec="seconds")
        writer.execute("DELETE FROM turn_traces WHERE started_at < ?", (cutoff,))
        # `id <= (subselect)` deletes everything older than the newest `cap`
        # rows. The offset is the cap itself and not one less: the subselect
        # names the first row that must *go*, and `id <=` takes it with the
        # rest. Offsetting by `cap - 1` names the oldest row worth keeping and
        # deletes it, leaving `cap - 1` — which is what the cap test caught.
        # With fewer rows than the cap the subselect is NULL, `id <= NULL` is
        # NULL, and nothing matches, so the under-cap case needs no branch.
        writer.execute(
            """DELETE FROM turn_traces WHERE id <= (
                   SELECT id FROM turn_traces ORDER BY id DESC LIMIT 1 OFFSET ?)""",
            (max(self.trace_row_cap, 1),))

    def traces(self, trace_id: str = "", session_id: str = "",
               limit: int = 1000) -> list[TraceSpan]:
        """Spans for one answer or one conversation, oldest first."""
        where, params = [], []
        if trace_id:
            where.append("trace_id = ?")
            params.append(trace_id)
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        # Newest-first inside the limit, then reversed, so an unfiltered call
        # returns the most *recent* spans rather than the oldest ever written —
        # a table bounded at two hundred thousand rows would otherwise answer a
        # developer's bare `traces()` with whatever survived from last fortnight.
        rows = self.db.execute(
            f"""SELECT session_id, turn_id, trace_id, span_id, parent_span_id,
                       name, started_at, duration_ms, status, source, attributes
                  FROM turn_traces {clause}
                 ORDER BY id DESC LIMIT ?""",
            (*params, limit)).fetchall()
        return [
            TraceSpan(
                session_id=r["session_id"], turn_id=r["turn_id"],
                trace_id=r["trace_id"], span_id=r["span_id"],
                parent_span_id=r["parent_span_id"], name=r["name"],
                started_at=r["started_at"], duration_ms=r["duration_ms"],
                status=r["status"], source=r["source"],
                attributes=json.loads(r["attributes"]))
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------- reporting

    def counts(self) -> dict:
        """Row counts per table, for the ingestion report and the transcript header."""
        tables = ("documents", "document_versions", "chunks", "document_caveats",
                  "excluded_documents", "index_snapshots")
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables}


# The name this adapter was built under, kept so existing imports keep working.
EmbeddedRepository = SQLiteKnowledgeRepository
