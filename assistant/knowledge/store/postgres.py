"""The deployment adapter: PostgreSQL with pgvector.

The same eight tables, the same column names and the same version semantics as
the SQLite adapter, and the same `KnowledgeRepository` Protocol above both. The
contract suite in `tests/test_repository_contract.py` is written against the
Protocol and runs against either, so "they are one system" is a claim something
checks rather than a claim the documentation makes.

Two things genuinely differ, and both are improvements rather than compromises.

**Filtering and similarity happen in one query.** SQLite has no vector type, so
the embedded adapter loads every active vector and does cosine in NumPy, then
filters in Python. Here the audience filter, the active-version join, the
authority ordering and the distance operator are one statement the planner sees
whole — which is the actual argument for pgvector over a separate vector
database, not raw speed at this corpus size.

**Authority is applied in SQL.** The same bounded bonus as the embedded adapter:
authority can reorder passages already close in similarity and cannot overturn a
clearly better match.

No ANN index is created by default. At six hundred passages an exact scan is
faster than an HNSW probe and always correct; the schema carries the index
definition for when the corpus justifies measuring it.

Requires `psycopg` and a reachable PostgreSQL with the `vector` extension.
Neither is needed for the assessment path, so the import is local to this module
and nothing above it changes.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from ..repository import (
    PRODUCT_BAND,
    TRACE_PRUNE_STRIDE,
    TRACE_RETENTION_DAYS,
    TRACE_ROW_CAP,
    IndexMismatch,
    PublicationBusy,
    RetrievalRequest,
)
from pathlib import Path

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
from ... import paths

SCHEMA = paths.DB_DIR / "schema.postgres.sql"

# How long a publisher waits for the one that is already publishing. Long
# enough that a genuinely slow large delta is not cut off, short enough that a
# nightly job fails with a diagnosis instead of still hanging in the morning.
PUBLISH_LOCK_TIMEOUT = 30

TIE_BAND = 0.02
_AUTHORITY_SPREAD = max(len(AUTHORITY), 1)
AUTHORITY_BONUS = TIE_BAND / _AUTHORITY_SPREAD


def _now() -> str:
    """The house timestamp: UTC, to the second, as the rest of the pipeline writes it."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vector_literal(values: list[float]) -> str:
    """pgvector's text input form. Parameterised, never interpolated into SQL."""
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"


class PostgresKnowledgeRepository:
    """`KnowledgeRepository` over PostgreSQL + pgvector."""

    def __init__(self, dsn: str, apply_schema: bool = True) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is not installed. The deployment adapter needs it; the "
                "assessment path does not. Install with: pip install 'psycopg[binary]' "
                "— on platforms with no binary wheel, run the application in the "
                "Docker Compose stack instead."
            ) from exc

        self.dsn = dsn
        self.conn = psycopg.connect(dsn, autocommit=False)
        if apply_schema:
            with self.conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            self.conn.commit()
        # Retention for turn_traces, per the review §8.1. Instance attributes
        # rather than the module constants read at the point of use, so a test
        # can prove the sweep prunes without writing two hundred batches and
        # waiting a fortnight to reach it. The values are shared with the SQLite
        # adapter through assistant/knowledge/repository.py: a retention window that
        # differed between the two would make "one system, two adapters" false
        # about the one thing an operator would notice.
        self.trace_retention_days = TRACE_RETENTION_DAYS
        self.trace_row_cap = TRACE_ROW_CAP
        self.trace_prune_stride = TRACE_PRUNE_STRIDE
        self._trace_writes = 0

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def read_snapshot(self):
        # Finish any earlier metadata read before beginning request isolation.
        self.conn.rollback()
        try:
            with self.conn.cursor() as cur:
                cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield self.snapshot()
        finally:
            self.conn.rollback()

    def __enter__(self) -> "PostgresKnowledgeRepository":
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

        Postgres gives this for free in a way SQLite does not: the partial
        unique index on `is_active` is checked inside the transaction, so an
        attempt to activate a second version of a document raises here and the
        entire publish rolls back rather than leaving a half-swapped index.
        """
        caveats = caveats or []
        excluded = excluded or []
        try:
            with self.conn.cursor() as cur:
                for table in ("chunks", "document_caveats", "document_versions",
                              "documents", "excluded_documents"):
                    cur.execute(f"DELETE FROM {table}")
                cur.execute("UPDATE index_snapshots SET is_active = FALSE")

                doc_ids: dict[str, int] = {}
                for d in documents:
                    cur.execute(
                        """INSERT INTO documents
                           (canonical_url, title, link_text, document_type,
                            authority, audience, product, created_at)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (d.canonical_url, d.title, d.link_text, d.document_type,
                         d.authority, d.audience, d.product, snapshot.created_at),
                    )
                    doc_ids[d.canonical_url] = cur.fetchone()[0]

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
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           RETURNING id""",
                        (did, v.version, v.content_hash, v.etag,
                         v.source_last_modified, v.source_path,
                         v.extraction_quality, v.notes, v.first_seen_at or snapshot.created_at,
                         v.fetched_at or snapshot.created_at, v.checked_at or snapshot.created_at, v.is_active),
                    )
                    vid = cur.fetchone()[0]
                    version_ids[(v.canonical_url, v.version)] = vid
                    if v.is_active:
                        cur.execute(
                            "UPDATE documents SET active_version_id = %s WHERE id = %s",
                            (vid, did))

                for c in chunks:
                    vid = version_ids.get((c.canonical_url, c.version))
                    if vid is None:
                        continue
                    cur.execute(
                        """INSERT INTO chunks
                           (document_version_id, chunk_index, section, content,
                            audience, product, source_date, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (vid, c.chunk_index, c.section, c.content, c.audience,
                         c.product, c.source_date,
                         _vector_literal(c.embedding) if c.embedding else None),
                    )

                for cav in caveats:
                    vid = next((v for (u, _n), v in version_ids.items()
                                if u == cav.canonical_url), None)
                    if vid is None:
                        continue
                    cur.execute(
                        """INSERT INTO document_caveats
                           (document_version_id, caveat_type, sentence, section)
                           VALUES (%s,%s,%s,%s)""",
                        (vid, cav.caveat_type, cav.sentence, cav.section))

                for ex in excluded:
                    cur.execute(
                        """INSERT INTO excluded_documents (url, link_text, reason)
                           VALUES (%s,%s,%s)""",
                        (ex.url, ex.link_text, ex.reason))

                cur.execute(
                    """INSERT INTO index_snapshots
                       (id, created_at, embedding_model, embedding_dimensions,
                        chunking_version, document_count, chunk_count,
                        is_active, notes)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,TRUE,%s)
                       ON CONFLICT (id) DO UPDATE SET
                         created_at = EXCLUDED.created_at,
                         embedding_model = EXCLUDED.embedding_model,
                         embedding_dimensions = EXCLUDED.embedding_dimensions,
                         chunking_version = EXCLUDED.chunking_version,
                         document_count = EXCLUDED.document_count,
                         chunk_count = EXCLUDED.chunk_count,
                         is_active = TRUE,
                         notes = EXCLUDED.notes""",
                    (snapshot.snapshot_id, snapshot.created_at,
                     snapshot.embedding_model, snapshot.embedding_dimensions,
                     snapshot.chunking_version, snapshot.document_count, snapshot.chunk_count,
                     json.dumps(snapshot.notes, ensure_ascii=False)))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
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

        Postgres enforces the one-active-version rule with a partial unique
        index, so the old version is deactivated before the new one is inserted
        rather than after. Getting that order wrong raises here rather than
        producing a document with two live versions, which is the behaviour
        worth having: the constraint is in the database, not in this method's
        memory.
        """
        try:
            with self.conn.cursor() as cur:
                # Bound the wait before taking the lock. `pg_advisory_xact_lock`
                # blocks forever, so an indexer that died without its connection
                # being reaped leaves every later run hanging with no output and
                # nothing in the log to explain it — the failure mode that looks
                # like "indexing is slow tonight" for a week. With a timeout the
                # same situation raises 55P03 and says what is wrong.
                cur.execute(f"SET LOCAL lock_timeout = '{PUBLISH_LOCK_TIMEOUT}s'")
                try:
                    cur.execute("SELECT pg_advisory_xact_lock(8675309)")
                except Exception as error:
                    # `psycopg` is imported lazily in __init__, so the error is
                    # matched on sqlstate rather than on an imported class.
                    if getattr(error, "sqlstate", None) != "55P03":
                        raise
                    raise PublicationBusy(
                        f"Another indexer has been publishing for more than "
                        f"{PUBLISH_LOCK_TIMEOUT}s and still holds the publication "
                        f"lock. Nothing was changed. If no indexer is running, a "
                        f"crashed one is still holding its connection open."
                    ) from error
                if "parent_snapshot" in snapshot.notes:
                    current = self.snapshot()
                    if snapshot.notes["parent_snapshot"] != (current.snapshot_id if current else None):
                        raise IndexMismatch("Another indexer published first; retry against the current release")
                for update in updates:
                    self._apply_one(cur, update, snapshot.created_at)

                for url in removed:
                    cur.execute("SELECT id FROM documents WHERE canonical_url = %s",
                                (url,))
                    row = cur.fetchone()
                    if row is None:
                        continue
                    # Deactivate, never delete. A withdrawn datasheet is a fact
                    # about the site, and an answer given while it was live has
                    # to stay explicable.
                    cur.execute(
                        "UPDATE document_versions SET is_active = FALSE "
                        "WHERE document_id = %s", (row[0],))
                    cur.execute(
                        "UPDATE documents SET active_version_id = NULL WHERE id = %s",
                        (row[0],))

                if excluded is not None:
                    cur.execute("DELETE FROM excluded_documents")
                    for ex in excluded:
                        cur.execute(
                            "INSERT INTO excluded_documents (url, link_text, reason) "
                            "VALUES (%s,%s,%s)", (ex.url, ex.link_text, ex.reason))

                if crawl_run is not None:
                    cur.execute(
                        """INSERT INTO crawl_runs
                           (started_at, completed_at, documents_checked,
                            documents_new, documents_changed, documents_unchanged,
                            documents_failed, documents_removed, snapshot_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (crawl_run.started_at,
                         crawl_run.completed_at or crawl_run.started_at,
                         crawl_run.documents_checked, crawl_run.documents_new,
                         crawl_run.documents_changed, crawl_run.documents_unchanged,
                         crawl_run.documents_failed, crawl_run.documents_removed,
                         snapshot.snapshot_id))

                # The snapshot records the live index, counted here rather
                # than taken from the caller: after a delta the totals depend on
                # what was already stored as well as what just changed.
                cur.execute("SELECT COUNT(*) FROM documents d "
                            "JOIN document_versions v ON v.id = d.active_version_id")
                live_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chunks c "
                            "JOIN document_versions v ON v.id = c.document_version_id "
                            "WHERE v.is_active")
                live_chunks = cur.fetchone()[0]

                cur.execute("UPDATE index_snapshots SET is_active = FALSE")
                cur.execute("""SELECT d.canonical_url, v.version_number,
                    v.content_hash, v.source_path FROM documents d
                    JOIN document_versions v ON v.id=d.active_version_id""")
                notes = {**snapshot.notes, "active_versions": {
                    r[0]: {"version": r[1], "content_hash": r[2], "source_path": r[3]}
                    for r in cur.fetchall()}}
                cur.execute(
                    """INSERT INTO index_snapshots
                       (id, created_at, embedding_model, embedding_dimensions,
                        chunking_version, document_count, chunk_count,
                        is_active, notes)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,TRUE,%s)
                       ON CONFLICT (id) DO UPDATE SET
                         created_at = EXCLUDED.created_at,
                         document_count = EXCLUDED.document_count,
                         chunk_count = EXCLUDED.chunk_count,
                         is_active = TRUE,
                         notes = EXCLUDED.notes""",
                    (snapshot.snapshot_id, snapshot.created_at,
                     snapshot.embedding_model, snapshot.embedding_dimensions,
                     snapshot.chunking_version, live_docs, live_chunks,
                     json.dumps(notes, ensure_ascii=False)))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return snapshot.snapshot_id

    def _apply_one(self, cur, update: DocumentUpdate, created_at: str) -> None:
        d, v = update.document, update.version

        cur.execute(
            "SELECT id, active_version_id FROM documents WHERE canonical_url = %s",
            (d.canonical_url,))
        row = cur.fetchone()

        if row is None:
            cur.execute(
                """INSERT INTO documents
                   (canonical_url, title, link_text, document_type, authority,
                    audience, product, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (d.canonical_url, d.title, d.link_text, d.document_type,
                 d.authority, d.audience, d.product, created_at))
            document_id, superseded = cur.fetchone()[0], None
        else:
            document_id, superseded = row[0], row[1]
            cur.execute(
                """UPDATE documents SET title=%s, link_text=%s, document_type=%s,
                          authority=%s, audience=%s, product=%s WHERE id=%s""",
                (d.title, d.link_text, d.document_type, d.authority,
                 d.audience, d.product, document_id))
            cur.execute(
                "UPDATE document_versions SET is_active = FALSE WHERE document_id = %s",
                (document_id,))

        cur.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM document_versions "
            "WHERE document_id = %s", (document_id,))
        next_number = cur.fetchone()[0]

        cur.execute(
            """INSERT INTO document_versions
               (document_id, version_number, content_hash, etag,
                source_last_modified, source_path, extraction_quality, notes,
                first_seen_at, fetched_at, checked_at, is_active, supersedes_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s) RETURNING id""",
            (document_id, next_number, v.content_hash, v.etag,
             v.source_last_modified, v.source_path, v.extraction_quality,
             v.notes, v.first_seen_at or created_at, v.fetched_at or created_at,
             v.checked_at or created_at, superseded))
        version_id = cur.fetchone()[0]
        cur.execute("UPDATE documents SET active_version_id = %s WHERE id = %s",
                    (version_id, document_id))

        for c in update.chunks:
            cur.execute(
                """INSERT INTO chunks
                   (document_version_id, chunk_index, section, content,
                    audience, product, source_date, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (version_id, c.chunk_index, c.section, c.content, c.audience,
                 c.product, c.source_date,
                 _vector_literal(c.embedding) if c.embedding else None))

        for cav in update.caveats:
            cur.execute(
                """INSERT INTO document_caveats
                   (document_version_id, caveat_type, sentence, section)
                   VALUES (%s,%s,%s,%s)""",
                (version_id, cav.caveat_type, cav.sentence, cav.section))

    def active_content_hashes(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT d.canonical_url, v.content_hash
                   FROM documents d
                   JOIN document_versions v ON v.id = d.active_version_id""")
            return {r[0]: r[1] for r in cur.fetchall()}

    def versions(self, canonical_url: str) -> list[DocumentVersion]:
        """Every version of one document, newest first — superseded ones included.

        The audit read, and the only one that returns inactive rows on purpose.
        A changed document supersedes rather than replaces, so the version that
        produced an answer last quarter is still here, marked inactive and
        unreachable by retrieval, which is what makes "why did it say that"
        answerable at all.

        Deliberately the same list, in the same order, as
        `assistant/knowledge/store/embedded.py` returns — decision 3's one
        boundary, two adapters. Columns are named rather than `SELECT *` and
        the timestamp columns are stringified, because PostgreSQL hands back
        `datetime` objects where SQLite hands back text and the domain type
        must not be able to tell which adapter filled it.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT v.version_number, v.content_hash, v.source_path, v.etag,
                          v.source_last_modified, v.first_seen_at, v.fetched_at,
                          v.checked_at, v.is_active, v.extraction_quality, v.notes
                   FROM document_versions v
                   JOIN documents d ON d.id = v.document_id
                   WHERE d.canonical_url = %s
                   ORDER BY v.version_number DESC""", (canonical_url,))
            rows = cur.fetchall()
        return [
            DocumentVersion(
                canonical_url=canonical_url, version=r[0], content_hash=r[1],
                source_path=r[2], etag=r[3] or "",
                source_last_modified=str(r[4] or ""), first_seen_at=str(r[5] or ""),
                fetched_at=str(r[6] or ""), checked_at=str(r[7] or ""),
                is_active=bool(r[8]), extraction_quality=r[9] or "unknown",
                notes=r[10] or "")
            for r in rows
        ]

    def crawl_runs(self, limit: int = 10) -> list[CrawlRun]:
        """The last few ingestion runs, newest first, with what each one found.

        New, changed, unchanged, failed and removed are separate counts because
        they are separate events: an unchanged document costs nothing, a failed
        one keeps the version it already had, and a removed one is deactivated
        rather than deleted. A run that checked ninety-four documents and
        changed none is not the same as a run that did nothing, and this is
        where that distinction survives the terminal.

        Written inside the publication transaction, so the counts and the
        `snapshot_id` beside them describe the same release. Same rows, same
        order as `assistant/knowledge/store/embedded.py`.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT started_at, completed_at, documents_checked,
                          documents_new, documents_changed, documents_unchanged,
                          documents_failed, documents_removed, snapshot_id
                   FROM crawl_runs ORDER BY id DESC LIMIT %s""", (limit,))
            rows = cur.fetchall()
        return [
            CrawlRun(started_at=str(r[0]), completed_at=str(r[1] or ""),
                     documents_checked=r[2], documents_new=r[3],
                     documents_changed=r[4], documents_unchanged=r[5],
                     documents_failed=r[6], documents_removed=r[7], snapshot_id=r[8])
            for r in rows
        ]

    # ----------------------------------------------------------- answering

    def retrieve(
        self,
        query_embedding: list[float],
        audiences: tuple[str, ...] = ("public",),
        top_k: int = 5,
        per_document_cap: int = 3,
    ) -> list[Retrieved]:
        """Nearest active chunks the caller may see, in one query."""
        return self.retrieve_for(RetrievalRequest(
            embedding=query_embedding, audiences=audiences, top_k=top_k,
            per_document_cap=per_document_cap))

    def retrieve_for(self, request: RetrievalRequest) -> list[Retrieved]:
        """The same search, told what the question named — still one statement.

        The audience filter is a `WHERE` clause, the per-document cap is a
        window function, and the product boost sits beside the distance operator
        in the same expression as the authority bonus. None of them depends on
        application code remembering to apply them, and none of them can be
        talked out of by a prompt.

        The product test is case-insensitive containment either way, which is
        exactly what `repository.product_matches` does in the embedded adapter —
        the contract suite holds both to the same result.
        """
        sql = """
        WITH scored AS (
            SELECT c.chunk_index, c.section, c.content, c.audience, c.product,
                   c.source_date, v.version_number,
                   d.canonical_url, d.title, d.link_text, d.document_type,
                   d.authority, d.audience AS doc_audience,
                   1 - (c.embedding <=> %(q)s::vector) AS similarity,
                   (1 - (c.embedding <=> %(q)s::vector))
                     + (%(spread)s - LEAST(GREATEST(d.authority, 1), %(spread)s))
                       * %(bonus)s
                     + CASE WHEN %(product)s <> ''
                              AND COALESCE(c.product, '') <> ''
                              AND (POSITION(LOWER(COALESCE(c.product, ''))
                                            IN LOWER(%(product)s)) > 0
                                OR POSITION(LOWER(%(product)s)
                                            IN LOWER(COALESCE(c.product, ''))) > 0)
                            THEN %(product_band)s ELSE 0 END AS effective
            FROM chunks c
            JOIN document_versions v ON v.id = c.document_version_id
            JOIN documents d         ON d.id = v.document_id
            WHERE v.is_active
              AND c.embedding IS NOT NULL
              AND c.audience = ANY(%(audiences)s)
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY canonical_url
                ORDER BY effective DESC, source_date DESC NULLS LAST
            ) AS per_doc
            FROM scored
        )
        SELECT * FROM ranked
        WHERE per_doc <= %(cap)s
        ORDER BY effective DESC, source_date DESC NULLS LAST
        LIMIT %(k)s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, {
                "q": _vector_literal(request.embedding),
                "audiences": list(request.audiences),
                "cap": request.per_document_cap,
                "k": request.top_k,
                "spread": _AUTHORITY_SPREAD,
                "bonus": AUTHORITY_BONUS,
                "product": request.product.strip(),
                "product_band": PRODUCT_BAND,
            })
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]

        return [Retrieved(chunk=self._chunk(r), score=float(r["similarity"]),
                          document=self._document(r)) for r in rows]

    def find_passages(
        self,
        product: str,
        terms: tuple[str, ...],
        audiences: tuple[str, ...] = ("public",),
        limit: int = 3,
    ) -> list[Retrieved]:
        """The passage that carries a property, found by metadata, in one query.

        No vector is involved, so `score` is 0.0 and is not a similarity.
        """
        if not terms:
            return []
        conditions = " OR ".join(
            f"(POSITION(%(t{i})s IN LOWER(c.content)) > 0"
            f" OR POSITION(%(t{i})s IN LOWER(COALESCE(c.section, ''))) > 0)"
            for i in range(len(terms)))
        params = {f"t{i}": t.strip().lower() for i, t in enumerate(terms)}
        params.update({"audiences": list(audiences),
                       "product": product.strip(), "limit": limit})
        sql = f"""
        SELECT c.chunk_index, c.section, c.content, c.audience, c.product,
               c.source_date, v.version_number,
               d.canonical_url, d.title, d.link_text, d.document_type,
               d.authority, d.audience AS doc_audience
        FROM chunks c
        JOIN document_versions v ON v.id = c.document_version_id
        JOIN documents d         ON d.id = v.document_id
        WHERE v.is_active
          AND c.audience = ANY(%(audiences)s)
          -- An empty product means the caller named none and the lookup is on
          -- the terms alone; the same containment-either-way rule as the
          -- embedded adapter, which the contract suite holds both to.
          AND (%(product)s = ''
            OR (COALESCE(c.product, '') <> ''
              AND (POSITION(LOWER(COALESCE(c.product, '')) IN LOWER(%(product)s)) > 0
                OR POSITION(LOWER(%(product)s) IN LOWER(COALESCE(c.product, ''))) > 0)))
          AND ({conditions})
        ORDER BY d.authority ASC, c.source_date DESC NULLS LAST,
                 d.canonical_url, c.chunk_index
        LIMIT %(limit)s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
        return [Retrieved(chunk=self._chunk(r), score=0.0,
                          document=self._document(r)) for r in rows]

    @staticmethod
    def _chunk(r: dict) -> Chunk:
        return Chunk(
            canonical_url=r["canonical_url"], version=r["version_number"],
            chunk_index=r["chunk_index"], section=r["section"] or "",
            content=r["content"], audience=r["audience"],
            product=r["product"] or "", document_type=r["document_type"],
            authority=r["authority"],
            source_date=str(r["source_date"]) if r["source_date"] else "",
        )

    @staticmethod
    def _document(r: dict) -> Document:
        return Document(
            canonical_url=r["canonical_url"], title=r["title"] or "",
            document_type=r["document_type"], authority=r["authority"],
            audience=r.get("doc_audience", "public"), product=r["product"] or "",
            link_text=r["link_text"] or "",
        )

    def document(self, canonical_url: str) -> Document | None:
        """One document's identity — title, type, authority, product, audience.

        Identity only: no version, no passages, no vectors. It answers "what is
        this URL" for a citation being rendered or a name being checked, and
        `None` for a URL the corpus has never held.

        No audience filter, deliberately — it returns the row's audience for
        the caller to act on and is not a retrieval path. Audience is enforced
        where evidence is chosen: in `retrieve()`, beside the distance
        operator, and in `manifest()`. Identical behaviour to
        `assistant/knowledge/store/embedded.py`; the row is zipped against
        `cur.description` so the shared `_document()` mapper can be fed the
        same dictionary shape SQLite's row factory produces.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT canonical_url, title, link_text, document_type,
                          authority, audience AS doc_audience, product
                   FROM documents WHERE canonical_url = %s""", (canonical_url,))
            row = cur.fetchone()
            if not row:
                return None
            columns = [c.name for c in cur.description]
        return self._document(dict(zip(columns, row)))

    def manifest(self, audiences: tuple[str, ...] = ("public",)) -> list[Document]:
        """What is currently published, as this audience is allowed to see it.

        Two constraints, both in the query rather than in the caller. The join
        through `active_version_id` drops a withdrawn document from the list
        while its history stays in the database, and the audience predicate
        means a staff-tagged document is never in a list handed to a public
        caller — which matters because this list answers a document request on
        the route path, where nothing downstream would filter it.

        Ordered by authority then URL, so a datasheet precedes a product page
        precedes an article, stably across runs.

        `= ANY(%s)` against `assistant/knowledge/store/embedded.py`'s
        `IN (?,?)` is the whole of the difference: a parameterised predicate
        either way, never string-built, and the same result. The dialect
        differs; the semantics are the ones decision 3 requires to match.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT d.canonical_url, d.title, d.link_text, d.document_type,
                          d.authority, d.audience AS doc_audience, d.product
                   FROM documents d
                   JOIN document_versions v ON v.id = d.active_version_id
                   WHERE d.audience = ANY(%s)
                   ORDER BY d.authority, d.canonical_url""", (list(audiences),))
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
        return [self._document(r) for r in rows]

    def caveats(self, canonical_url: str) -> list[Caveat]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT dc.caveat_type, dc.sentence, dc.section
                   FROM document_caveats dc
                   JOIN document_versions v ON v.id = dc.document_version_id
                   JOIN documents d         ON d.id = v.document_id
                   WHERE d.canonical_url = %s AND v.is_active""", (canonical_url,))
            rows = cur.fetchall()
        return [Caveat(canonical_url, r[0], r[1], r[2] or "") for r in rows]

    def excluded(self) -> list[Excluded]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT url, link_text, reason FROM excluded_documents "
                        "ORDER BY url")
            rows = cur.fetchall()
        return [Excluded(r[0], r[2], r[1] or "") for r in rows]

    def snapshot(self) -> Snapshot | None:
        """The release being served, or `None` when nothing has been published.

        The snapshot binds the embedding model and its dimension, the chunking
        version and the document and passage counts to the index an answer came
        from, which is what makes the answer explicable later. The engine reads
        it at construction and **refuses to run when the model or the dimension
        does not match what it is configured for** — an index built by one
        embedding model and queried by another returns confident nonsense
        rather than an error.

        Exactly one row is active, so there is a single value to return rather
        than a newest-of-several. `None` is a normal state and
        `assistant/infrastructure/health.py` reports it as "not built".

        `notes` is tolerated as either a mapping or JSON text because `jsonb`
        arrives already decoded here while SQLite stores a string; the domain
        type must not be able to tell which adapter filled it. Same record as
        `assistant/knowledge/store/embedded.py`.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, created_at, embedding_model, embedding_dimensions,
                          chunking_version, document_count, chunk_count, notes
                   FROM index_snapshots WHERE is_active""")
            row = cur.fetchone()
        if not row:
            return None
        notes = row[7]
        return Snapshot(
            snapshot_id=row[0], created_at=str(row[1]), embedding_model=row[2],
            embedding_dimensions=row[3], chunking_version=row[4],
            document_count=row[5], chunk_count=row[6],
            notes=notes if isinstance(notes, dict) else json.loads(notes or "{}"),
        )

    def active_version(self, canonical_url: str) -> DocumentVersion | None:
        """The one version of this document retrieval is allowed to reach.

        Singular by database constraint rather than by convention: a partial
        unique index on the active rows refuses a second active version of the
        same document, so an activation that failed to deactivate its
        predecessor raises an integrity error from PostgreSQL instead of
        letting two coverage figures for one product into the index. That is
        why no ordering or tiebreak is needed here.

        `None` means withdrawn or never successfully indexed — both real states
        — and `versions()` still returns the history that distinguishes either
        from a document the corpus never held. Same guarantee, same result as
        `assistant/knowledge/store/embedded.py`.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT v.version_number, v.content_hash, v.source_path, v.etag,
                          v.source_last_modified, v.first_seen_at, v.fetched_at,
                          v.checked_at, v.is_active, v.extraction_quality, v.notes
                   FROM document_versions v
                   JOIN documents d ON d.id = v.document_id
                   WHERE d.canonical_url = %s AND v.is_active""", (canonical_url,))
            row = cur.fetchone()
        if not row:
            return None
        return DocumentVersion(
            canonical_url=canonical_url, version=row[0], content_hash=row[1],
            source_path=row[2], etag=row[3] or "",
            source_last_modified=str(row[4] or ""), first_seen_at=str(row[5] or ""),
            fetched_at=str(row[6] or ""), checked_at=str(row[7] or ""),
            is_active=bool(row[8]), extraction_quality=row[9] or "unknown",
            notes=row[10] or "",
        )

    # ------------------------------------------------------------------ audit

    def log_answer(self, entry: AnswerLogEntry) -> None:
        """Record one answer, on its own connection, committed on its own.

        Deliberately not on `self.conn`. `read_snapshot()` puts that connection
        into `REPEATABLE READ READ ONLY` for the life of an answer, and an
        insert there does not fail quietly — it raises, aborts the transaction
        and takes the rest of the answer's reads down with it. A short-lived
        connection of its own commits independently, which is what makes it
        safe for the call site to log from inside the snapshot it describes.

        Do not move this onto the reading connection. The cost of a connection
        is paid once per answered question, against a retrieval and a
        generation; the cost of getting it wrong is an answer that dies on its
        own audit row.
        """
        import psycopg

        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(
                # The question is stored and the generated answer is not. The
                # route, the snapshot and the chunk ids are what make a reply
                # explicable; keeping the prose of every conversation
                # indefinitely is a privacy decision nobody has asked for.
                """INSERT INTO answer_log
                   (asked_at, question, audiences, path_taken, snapshot_id,
                    chunk_ids, generation_model, check_failed, source)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (entry.asked_at or _now(), entry.question,
                 list(entry.audiences), entry.path_taken,
                 # Empty means "no snapshot was consulted", and the column is a
                 # foreign key: NULL is the only honest way to say that.
                 entry.snapshot_id or None,
                 list(entry.chunk_ids), entry.generation_model,
                 entry.check_failed, entry.source or "unknown"),
            )

    def answer_log(self, limit: int = 20) -> list[AnswerLogEntry]:
        """Recent answers, newest first."""
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT asked_at, question, audiences, path_taken, snapshot_id,
                          chunk_ids, generation_model, check_failed, source
                   FROM answer_log ORDER BY asked_at DESC, id DESC LIMIT %s""",
                (limit,))
            rows = cur.fetchall()
        return [
            AnswerLogEntry(
                question=r[1], path_taken=r[3], audiences=tuple(r[2]),
                snapshot_id=r[4] or "", chunk_ids=list(r[5]),
                generation_model=r[6], check_failed=r[7], asked_at=str(r[0]),
                source=r[8])
            for r in rows
        ]

    # ----------------------------------------------------------------- traces

    def record_spans(self, spans: list[TraceSpan]) -> None:
        """Write one turn's spans on a connection of its own, and never raise.

        The same arrangement as `log_answer` and for the same reason:
        `read_snapshot()` puts `self.conn` into `REPEATABLE READ READ ONLY` for
        the life of an answer, so an insert there raises, aborts the transaction
        and takes the answer's remaining reads down with it.

        It differs from `log_answer` in swallowing. An audit row is a promise
        and says so when it cannot be kept; a timing row is a debugging aid, and
        failing an answer over one would invert the priority the design rests
        on. The failure goes onto the event stream rather than into silence.

        `autocommit=False` here, unlike `log_answer`: the batch and the prune
        are one transaction, because a half-written span tree reads as stages
        that did not happen.
        """
        if not spans:
            return
        import psycopg

        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO turn_traces
                           (session_id, turn_id, trace_id, span_id, parent_span_id,
                            name, started_at, duration_ms, status, source, attributes)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        [(sp.session_id, sp.turn_id, sp.trace_id, sp.span_id,
                          sp.parent_span_id, sp.name, sp.started_at or _now(),
                          int(sp.duration_ms), sp.status or "ok",
                          sp.source or "unknown",
                          json.dumps(sp.attributes or {}, default=str))
                         for sp in spans])
                    self._prune_traces(cur)
                conn.commit()
        except Exception as error:                     # noqa: BLE001
            from ...infrastructure import observability as obs
            obs.event("store_error", operation="record_spans",
                      error=type(error).__name__, detail=str(error))

    def _prune_traces(self, cur) -> None:
        """Window then cap, on the stride, inside the caller's transaction.

        Byte-for-byte the SQLite adapter's policy in the other dialect. The cap
        deletes by id rather than by timestamp because ids are monotonic and the
        house timestamp is only to the second, so a time cutoff would either
        take a whole second's rows or none of them.
        """
        self._trace_writes += 1
        if self._trace_writes % max(self.trace_prune_stride, 1):
            return
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=self.trace_retention_days)
                  ).isoformat(timespec="seconds")
        cur.execute("DELETE FROM turn_traces WHERE started_at < %s", (cutoff,))
        cur.execute(
            """DELETE FROM turn_traces WHERE id <= (
                   SELECT id FROM turn_traces ORDER BY id DESC
                   OFFSET %s LIMIT 1)""",
            (max(self.trace_row_cap, 1),))

    def traces(self, trace_id: str = "", session_id: str = "",
               limit: int = 1000) -> list[TraceSpan]:
        """Spans for one answer or one conversation, oldest first."""
        where, params = [], []
        if trace_id:
            where.append("trace_id = %s")
            params.append(trace_id)
        if session_id:
            where.append("session_id = %s")
            params.append(session_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        # Newest-first inside the limit, then reversed: a bare `traces()` should
        # answer with the most recent spans, not with whatever is oldest in a
        # table bounded at two hundred thousand rows.
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT session_id, turn_id, trace_id, span_id, parent_span_id,
                           name, started_at, duration_ms, status, source, attributes
                      FROM turn_traces {clause}
                     ORDER BY id DESC LIMIT %s""",
                (*params, limit))
            rows = cur.fetchall()
        return [
            TraceSpan(
                session_id=r[0], turn_id=r[1], trace_id=r[2], span_id=r[3],
                parent_span_id=r[4], name=r[5], started_at=r[6],
                duration_ms=int(r[7]), status=r[8], source=r[9],
                # JSONB comes back already decoded; the fallback is for a column
                # migrated in from TEXT rather than created as JSONB.
                attributes=r[10] if isinstance(r[10], dict) else json.loads(r[10]))
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------- reporting

    def counts(self) -> dict:
        tables = ("documents", "document_versions", "chunks", "document_caveats",
                  "excluded_documents", "index_snapshots")
        out = {}
        with self.conn.cursor() as cur:
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                out[t] = cur.fetchone()[0]
        return out

    def health(self) -> dict:
        """Readiness, not liveness: can this actually serve an answer?

        A connection that opens proves the process is up. Serving needs the
        schema present, a snapshot active, and chunks with vectors in it.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                has_vector = cur.fetchone() is not None
                cur.execute("""SELECT COUNT(*) FROM chunks c JOIN document_versions v
                    ON v.id=c.document_version_id WHERE v.is_active AND c.embedding IS NOT NULL""")
                embedded = cur.fetchone()[0]
            snapshot = self.snapshot()
            return {
                "connected": True,
                "pgvector": has_vector,
                "active_snapshot": snapshot.snapshot_id if snapshot else None,
                "embedded_chunks": embedded,
                "ready": bool(has_vector and snapshot and embedded),
            }
        except Exception as exc:
            return {"connected": False, "ready": False, "error": str(exc)}
