-- Production schema: PostgreSQL + pgvector.
--
-- The deployment target. Raw HTML and PDFs stay on a versioned filesystem —
-- Postgres holds their identity, history and the retrievable chunks, not the
-- bytes. Filesystem is the original evidence; this is the control plane.
--
--   CREATE EXTENSION IF NOT EXISTS vector;
--
-- The same tables exist in db/schema.sqlite.sql for the offline assessment
-- adapter. Column names and version semantics are identical by design; only
-- the vector type and the similarity operator differ.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- documents
-- One row per source document, identified by canonical URL. Authority is the
-- outer sort in retrieval: a current datasheet outranks a newer FAQ entry.
CREATE TABLE IF NOT EXISTS documents (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    canonical_url       TEXT        NOT NULL UNIQUE,
    title               TEXT        NOT NULL DEFAULT '',
    link_text           TEXT        NOT NULL DEFAULT '',
    document_type       TEXT        NOT NULL,
    authority           SMALLINT    NOT NULL,
    audience            TEXT        NOT NULL DEFAULT 'public',
    product             TEXT        NOT NULL DEFAULT '',
    active_version_id   BIGINT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_audience_ck CHECK (audience IN ('public', 'trade', 'staff'))
);

-- -------------------------------------------------------- document_versions
-- Every fetched state. Exactly one row per document has is_active = TRUE.
-- Keeping superseded versions buys auditability, rollback, and an answer to
-- "why did the assistant say 16-20 last March?".
CREATE TABLE IF NOT EXISTS document_versions (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id             BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    version_number          INTEGER     NOT NULL,
    content_hash            TEXT        NOT NULL,
    etag                    TEXT        NOT NULL DEFAULT '',
    source_last_modified    TEXT        NOT NULL DEFAULT '',
    source_path             TEXT        NOT NULL,
    extraction_quality      TEXT        NOT NULL DEFAULT 'unknown',
    notes                   TEXT        NOT NULL DEFAULT '',
    first_seen_at           TIMESTAMPTZ NOT NULL,
    fetched_at              TIMESTAMPTZ NOT NULL,
    checked_at              TIMESTAMPTZ NOT NULL,
    is_active               BOOLEAN     NOT NULL DEFAULT FALSE,
    supersedes_id           BIGINT      REFERENCES document_versions (id),
    UNIQUE (document_id, version_number)
);

-- At most one active version per document, enforced by the database rather
-- than by the application remembering to.
CREATE UNIQUE INDEX IF NOT EXISTS document_versions_one_active
    ON document_versions (document_id) WHERE is_active;

-- ------------------------------------------------------------------- chunks
-- A retrievable passage: a document section, or a bullet kept whole. The unit
-- of retrieval and the unit of citation are the same thing.
CREATE TABLE IF NOT EXISTS chunks (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_version_id BIGINT      NOT NULL REFERENCES document_versions (id) ON DELETE CASCADE,
    chunk_index         INTEGER     NOT NULL,
    section             TEXT        NOT NULL DEFAULT '',
    content             TEXT        NOT NULL,
    audience            TEXT        NOT NULL DEFAULT 'public',
    product             TEXT        NOT NULL DEFAULT '',
    source_date         TEXT        NOT NULL DEFAULT '',
    embedding           vector(1024),
    UNIQUE (document_version_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_audience_idx ON chunks (audience);

-- -------------------------------------------------------- document caveats
-- Caveat sentences belong to the document, not the chunk: Fine Stuff puts its
-- 8 degree limit under Mixing and its DIY warning under Application, while the
-- steps a user asks about sit elsewhere again. Appended by code whenever any
-- chunk of that document is used.
CREATE TABLE IF NOT EXISTS document_caveats (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_version_id BIGINT NOT NULL REFERENCES document_versions (id) ON DELETE CASCADE,
    caveat_type         TEXT   NOT NULL,   -- temperature | diy | incompatibility | other
    sentence            TEXT   NOT NULL,
    section             TEXT   NOT NULL DEFAULT ''
);

-- ------------------------------------------------------------------ excluded
-- Documents deliberately not indexed, by name and link, with the reason.
CREATE TABLE IF NOT EXISTS excluded_documents (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url         TEXT NOT NULL,
    link_text   TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL
);

-- --------------------------------------------------------------- crawl_runs
CREATE TABLE IF NOT EXISTS crawl_runs (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at              TIMESTAMPTZ NOT NULL,
    completed_at            TIMESTAMPTZ,
    documents_checked       INTEGER NOT NULL DEFAULT 0,
    documents_new           INTEGER NOT NULL DEFAULT 0,
    documents_changed       INTEGER NOT NULL DEFAULT 0,
    documents_unchanged     INTEGER NOT NULL DEFAULT 0,
    documents_failed        INTEGER NOT NULL DEFAULT 0
);

-- ----------------------------------------------------------- index_snapshots
-- An index build. Every answer logs its snapshot id, so "why did it recommend
-- that?" is answerable six months later against the exact state it saw.
CREATE TABLE IF NOT EXISTS index_snapshots (
    id                      TEXT        PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL,
    embedding_model         TEXT        NOT NULL,
    embedding_dimensions    INTEGER     NOT NULL,
    chunking_version        TEXT        NOT NULL,
    document_count          INTEGER     NOT NULL,
    chunk_count             INTEGER     NOT NULL,
    is_active               BOOLEAN     NOT NULL DEFAULT FALSE,
    notes                   JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS index_snapshots_one_active
    ON index_snapshots ((TRUE)) WHERE is_active;

-- ------------------------------------------------------------- answer_log
-- Production auditability. Not built in the prototype, defined here because
-- the reproducibility claim depends on it existing.
CREATE TABLE IF NOT EXISTS answer_log (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    asked_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    question            TEXT        NOT NULL,
    audiences           TEXT[]      NOT NULL,
    path_taken          TEXT        NOT NULL,   -- route | extract | compose | defer | refuse
    snapshot_id         TEXT        REFERENCES index_snapshots (id),
    chunk_ids           BIGINT[]    NOT NULL DEFAULT '{}',
    generation_model    TEXT        NOT NULL DEFAULT '',
    check_failed        TEXT        NOT NULL DEFAULT ''
);

-- Retrieval, for reference: audience and active-version filtering happen in
-- the query, authority is the outer sort, similarity the inner one.
--
--   SELECT c.id, c.content, d.canonical_url, d.link_text, d.authority,
--          1 - (c.embedding <=> :q) AS similarity
--     FROM chunks c
--     JOIN document_versions dv ON c.document_version_id = dv.id
--     JOIN documents d          ON dv.document_id = d.id
--    WHERE dv.is_active
--      AND c.audience = ANY(:audiences)
--    ORDER BY d.authority ASC, similarity DESC
--    LIMIT :top_k;
