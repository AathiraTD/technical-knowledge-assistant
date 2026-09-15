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
from pathlib import Path

from ..model import (
    AUTHORITY,
    Caveat,
    CrawlRun,
    Chunk,
    Document,
    DocumentUpdate,
    DocumentVersion,
    Excluded,
    Retrieved,
    Snapshot,
)

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.postgres.sql"

TIE_BAND = 0.02
_AUTHORITY_SPREAD = max(len(AUTHORITY), 1)
AUTHORITY_BONUS = TIE_BAND / _AUTHORITY_SPREAD


def _vector_literal(values: list[float]) -> str:
    """pgvector's text input form. Parameterised, never interpolated into SQL."""
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"


class PostgresKnowledgeRepository:
    """`KnowledgeRepository` over PostgreSQL + pgvector."""

    def __init__(self, dsn: str, apply_schema: bool = True) -> None:
        try:
            import psycopg
        except ImportError as exc:                       # pragma: no cover
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

    def close(self) -> None:
        self.conn.close()

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
                         v.extraction_quality, v.notes, v.first_seen_at,
                         v.fetched_at, v.checked_at, v.is_active),
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
                         c.product, c.source_date or None,
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
                     snapshot.chunking_version, live_docs, live_chunks,
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
                            documents_failed)
                           VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                        (crawl_run.started_at,
                         crawl_run.completed_at or crawl_run.started_at,
                         crawl_run.documents_checked, crawl_run.documents_new,
                         crawl_run.documents_changed, crawl_run.documents_unchanged,
                         crawl_run.documents_failed))

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
                     json.dumps(snapshot.notes, ensure_ascii=False)))
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
             v.source_last_modified or None, v.source_path, v.extraction_quality,
             v.notes, v.first_seen_at or None, v.fetched_at or None,
             v.checked_at or None, superseded))
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
                 c.product, c.source_date or None,
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
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT started_at, completed_at, documents_checked,
                          documents_new, documents_changed, documents_unchanged,
                          documents_failed
                   FROM crawl_runs ORDER BY id DESC LIMIT %s""", (limit,))
            rows = cur.fetchall()
        return [
            CrawlRun(started_at=str(r[0]), completed_at=str(r[1] or ""),
                     documents_checked=r[2], documents_new=r[3],
                     documents_changed=r[4], documents_unchanged=r[5],
                     documents_failed=r[6])
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
        """Nearest active chunks the caller may see, in one query.

        The audience filter is a `WHERE` clause and the per-document cap is a
        window function, so neither depends on application code remembering to
        apply them — and neither can be talked out of by a prompt.
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
                       * %(bonus)s AS effective
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
                "q": _vector_literal(query_embedding),
                "audiences": list(audiences),
                "cap": per_document_cap,
                "k": top_k,
                "spread": _AUTHORITY_SPREAD,
                "bonus": AUTHORITY_BONUS,
            })
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]

        return [Retrieved(chunk=self._chunk(r), score=float(r["similarity"]),
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
                cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
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
