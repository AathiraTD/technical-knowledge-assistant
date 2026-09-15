-- Offline assessment schema: SQLite.
--
-- The twin of db/schema.postgres.sql. Same tables, same column names, same
-- version semantics — one active version per document, one active snapshot,
-- audience filtering in the query. Three differences, all forced by SQLite:
--
--   1. Embeddings are stored as BLOB (float32 little-endian) and cosine
--      similarity is computed in NumPy after loading, because SQLite has no
--      vector type. At six to eight hundred chunks that is a single matrix
--      multiply and costs microseconds.
--   2. IDENTITY becomes INTEGER PRIMARY KEY AUTOINCREMENT.
--   3. JSONB and TEXT[] become TEXT holding JSON.
--
-- sqlite3 is in the standard library, so the assessment path needs no service,
-- no container and no extra dependency: the whole store is one file that ships
-- with the submission.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_url       TEXT    NOT NULL UNIQUE,
    title               TEXT    NOT NULL DEFAULT '',
    link_text           TEXT    NOT NULL DEFAULT '',
    document_type       TEXT    NOT NULL,
    authority           INTEGER NOT NULL,
    audience            TEXT    NOT NULL DEFAULT 'public'
                            CHECK (audience IN ('public', 'trade', 'staff')),
    product             TEXT    NOT NULL DEFAULT '',
    active_version_id   INTEGER,
    created_at          TEXT    NOT NULL
);

-- -------------------------------------------------------- document_versions
CREATE TABLE IF NOT EXISTS document_versions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id             INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    version_number          INTEGER NOT NULL,
    content_hash            TEXT    NOT NULL,
    etag                    TEXT    NOT NULL DEFAULT '',
    source_last_modified    TEXT    NOT NULL DEFAULT '',
    source_path             TEXT    NOT NULL,
    extraction_quality      TEXT    NOT NULL DEFAULT 'unknown',
    notes                   TEXT    NOT NULL DEFAULT '',
    first_seen_at           TEXT    NOT NULL,
    fetched_at              TEXT    NOT NULL,
    checked_at              TEXT    NOT NULL,
    is_active               INTEGER NOT NULL DEFAULT 0,
    supersedes_id           INTEGER REFERENCES document_versions (id),
    UNIQUE (document_id, version_number)
);

-- One active version per document, enforced by the database rather than by
-- the application remembering to. This is what stops a changed datasheet
-- making both its old and new coverage figures retrievable at once.
CREATE UNIQUE INDEX IF NOT EXISTS document_versions_one_active
    ON document_versions (document_id) WHERE is_active = 1;

-- ------------------------------------------------------------------- chunks
CREATE TABLE IF NOT EXISTS chunks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_version_id INTEGER NOT NULL REFERENCES document_versions (id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    section             TEXT    NOT NULL DEFAULT '',
    content             TEXT    NOT NULL,
    audience            TEXT    NOT NULL DEFAULT 'public',
    product             TEXT    NOT NULL DEFAULT '',
    source_date         TEXT    NOT NULL DEFAULT '',
    embedding           BLOB,   -- float32 little-endian; cosine computed in NumPy
    UNIQUE (document_version_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_audience_idx ON chunks (audience);

-- -------------------------------------------------------- document caveats
CREATE TABLE IF NOT EXISTS document_caveats (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_version_id INTEGER NOT NULL REFERENCES document_versions (id) ON DELETE CASCADE,
    caveat_type         TEXT    NOT NULL,   -- temperature | diy | incompatibility | other
    sentence            TEXT    NOT NULL,
    section             TEXT    NOT NULL DEFAULT ''
);

-- ------------------------------------------------------------------ excluded
CREATE TABLE IF NOT EXISTS excluded_documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    link_text   TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL
);

-- --------------------------------------------------------------- crawl_runs
CREATE TABLE IF NOT EXISTS crawl_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at              TEXT    NOT NULL,
    completed_at            TEXT,
    documents_checked       INTEGER NOT NULL DEFAULT 0,
    documents_new           INTEGER NOT NULL DEFAULT 0,
    documents_changed       INTEGER NOT NULL DEFAULT 0,
    documents_unchanged     INTEGER NOT NULL DEFAULT 0,
    documents_failed        INTEGER NOT NULL DEFAULT 0,
    documents_removed       INTEGER NOT NULL DEFAULT 0,
    snapshot_id             TEXT NOT NULL DEFAULT ''
);

-- ----------------------------------------------------------- index_snapshots
CREATE TABLE IF NOT EXISTS index_snapshots (
    id                      TEXT    PRIMARY KEY,
    created_at              TEXT    NOT NULL,
    embedding_model         TEXT    NOT NULL,
    embedding_dimensions    INTEGER NOT NULL,
    chunking_version        TEXT    NOT NULL,
    document_count          INTEGER NOT NULL,
    chunk_count             INTEGER NOT NULL,
    is_active               INTEGER NOT NULL DEFAULT 0,
    notes                   TEXT    NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS index_snapshots_one_active
    ON index_snapshots (is_active) WHERE is_active = 1;

-- ------------------------------------------------------------- answer_log
CREATE TABLE IF NOT EXISTS answer_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at            TEXT    NOT NULL,
    question            TEXT    NOT NULL,
    audiences           TEXT    NOT NULL DEFAULT '[]',
    path_taken          TEXT    NOT NULL,
    snapshot_id         TEXT    REFERENCES index_snapshots (id),
    chunk_ids           TEXT    NOT NULL DEFAULT '[]',
    generation_model    TEXT    NOT NULL DEFAULT '',
    check_failed        TEXT    NOT NULL DEFAULT ''
);
