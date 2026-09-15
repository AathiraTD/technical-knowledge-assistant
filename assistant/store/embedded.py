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
from pathlib import Path

import numpy as np

from ..model import (
    AUTHORITY,
    Caveat,
    Chunk,
    Document,
    DocumentVersion,
    Excluded,
    Retrieved,
    Snapshot,
)

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sqlite.sql"

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


def _f32(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class SQLiteKnowledgeRepository:
    """`KnowledgeRepository` over a single SQLite file."""

    def __init__(self, path: str | Path = "data/index/knowledge.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA.read_text(encoding="utf-8"))
        self._matrix: np.ndarray | None = None
        self._rows: list[sqlite3.Row] = []

    def close(self) -> None:
        self.db.close()

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

    # ----------------------------------------------------------- answering

    def _load(self) -> None:
        """Load active chunk vectors once, normalised, so cosine is a dot product."""
        if self._matrix is not None:
            return
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

        The audience filter is applied to the rows, not to a prompt: a prompt
        instruction is a request, and an access control has to be a fact.
        """
        self._load()
        if self._matrix is None or self._matrix.size == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        scores = self._matrix @ q

        allowed = set(audiences)
        candidates = [
            (float(scores[i]), r) for i, r in enumerate(self._rows)
            if r["audience"] in allowed
        ]
        # Two stable passes. "Newest wins within a type" cannot be negated on a
        # date string the way a number can, so dates are sorted descending
        # first; the ranking sort is stable and preserves that order wherever
        # the effective scores are equal.
        candidates.sort(key=lambda t: t[1]["source_date"] or "", reverse=True)
        candidates.sort(key=lambda t: -self._effective(t[0], t[1]["authority"]))

        out, per_doc = [], {}
        for score, r in candidates:
            url = r["canonical_url"]
            if per_doc.get(url, 0) >= per_document_cap:
                continue
            per_doc[url] = per_doc.get(url, 0) + 1
            out.append(Retrieved(chunk=self._chunk(r), score=score,
                                 document=self._document(r)))
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _effective(similarity: float, authority: int) -> float:
        """Similarity, nudged by authority within a bounded margin.

        A datasheet is preferred over a product page that matched marginally
        better; it is not preferred over one that matched clearly better. The
        score returned here orders results and is never shown — the similarity
        reported alongside an answer is the real cosine.
        """
        rank = max(1, min(authority, _AUTHORITY_SPREAD))
        return similarity + (_AUTHORITY_SPREAD - rank) * AUTHORITY_BONUS

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

    # ------------------------------------------------------------- reporting

    def counts(self) -> dict:
        """Row counts per table, for the ingestion report and the transcript header."""
        tables = ("documents", "document_versions", "chunks", "document_caveats",
                  "excluded_documents", "index_snapshots")
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables}


# The name this adapter was built under, kept so existing imports keep working.
EmbeddedRepository = SQLiteKnowledgeRepository
