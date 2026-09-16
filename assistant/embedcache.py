"""A content-addressed cache of embeddings.

Embedding 644 passages on a laptop CPU takes about forty minutes. Doing that
again because one passage changed, or because a transient error ended the run
at 97 per cent, is waste that the assessor pays for in wall-clock time.

The key is the SHA-256 of the passage text together with the model tag and the
dimension count. That is the whole correctness argument: a vector is returned
only for the exact text it was computed from, by the exact model that computed
it. Point the indexer at a different embedding model and every lookup misses
and the run recomputes, which is the required behaviour — a cached vector from
another model is precisely the confident nonsense the index header exists to
prevent.

Because the cache is keyed that way it is safe to ship. A clean clone then
builds its index in about a minute instead of forty, and still rebuilds
honestly if anything it depends on has changed.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(os.environ.get("ASSISTANT_EMBEDDING_CACHE",
                    Path(__file__).resolve().parents[1] / "data" / "embeddings.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    dimensions  INTEGER NOT NULL,
    vector      BLOB NOT NULL
);
"""


def key_for(text: str, model: str, dimensions: int) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}|{dimensions}|{digest}"


class EmbeddingCache:
    """Read-through cache around the embedding call."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_many(self, texts: list[str], model: str, dimensions: int
                 ) -> dict[int, list[float]]:
        """Vectors already known, by position in the input list."""
        found: dict[int, list[float]] = {}
        keys = [key_for(t, model, dimensions) for t in texts]
        for start in range(0, len(keys), 500):     # SQLite parameter ceiling
            window = keys[start:start + 500]
            marks = ",".join("?" * len(window))
            rows = dict(self.db.execute(
                f"SELECT key, vector FROM embedding_cache WHERE key IN ({marks})",
                window,
            ).fetchall())
            for offset, k in enumerate(window):
                blob = rows.get(k)
                if blob is not None:
                    found[start + offset] = np.frombuffer(
                        blob, dtype=np.float32).tolist()
        self.hits += len(found)
        self.misses += len(texts) - len(found)
        return found

    def put_many(self, pairs: list[tuple[str, list[float]]], model: str,
                 dimensions: int) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO embedding_cache (key, model, dimensions, vector) "
            "VALUES (?,?,?,?)",
            [(key_for(t, model, dimensions), model, dimensions,
              np.asarray(v, dtype=np.float32).tobytes()) for t, v in pairs],
        )
        self.db.commit()

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
