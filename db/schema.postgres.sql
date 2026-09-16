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
    embedding           vector,
    UNIQUE (document_version_id, chunk_index)
);

-- Exact search supports multiple historical embedding configurations. An ANN
-- index is an optional measured optimisation, not a correctness dependency.
DROP INDEX IF EXISTS chunks_embedding_idx;
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector;
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

ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS documents_removed INTEGER NOT NULL DEFAULT 0;
ALTER TABLE crawl_runs ADD COLUMN IF NOT EXISTS snapshot_id TEXT NOT NULL DEFAULT '';

CREATE UNIQUE INDEX IF NOT EXISTS index_snapshots_one_active
    ON index_snapshots ((TRUE)) WHERE is_active;

-- ------------------------------------------------------------- answer_log
-- What one answer used: the snapshot it read, the passages it cited and the
-- route it took. The question is kept and the generated answer is not, because
-- the route and the evidence are what make a reply explicable and retaining
-- the prose of every conversation indefinitely is a separate decision nobody
-- has taken. The twin of this table in db/schema.sqlite.sql holds the two
-- array columns as JSON in TEXT.
CREATE TABLE IF NOT EXISTS answer_log (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    asked_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    question            TEXT        NOT NULL,
    audiences           TEXT[]      NOT NULL,
    path_taken          TEXT        NOT NULL,   -- route | extract | compose | defer | refuse
    snapshot_id         TEXT        REFERENCES index_snapshots (id),
    chunk_ids           TEXT[]      NOT NULL DEFAULT '{}',
    generation_model    TEXT        NOT NULL DEFAULT '',
    -- `source` is the surface that asked: cli, web, evaluation, or unknown.
    -- See db/schema.sqlite.sql for the defect it closes — evaluation traffic
    -- and real questions shared one table, so any rate counted from it counted
    -- the probes designed to refuse.
    check_failed        TEXT        NOT NULL DEFAULT '',
    source              TEXT        NOT NULL DEFAULT 'unknown'
);

-- A chunk id in this system is the string 'url#vN-i' — the citation, not a row
-- number — so the column that records which passages produced an answer has to
-- hold text. It was declared BIGINT[] while nothing wrote to it. Corrected in
-- place as well as in the definition above, so a database created before this
-- migrates rather than being left with a type no chunk id fits.
DO $$
BEGIN
    IF (SELECT atttypid FROM pg_attribute
          WHERE attrelid = to_regclass('answer_log') AND attname = 'chunk_ids')
       = 'bigint[]'::regtype THEN
        ALTER TABLE answer_log
            ALTER COLUMN chunk_ids TYPE TEXT[] USING chunk_ids::text::text[];
    END IF;
END $$;

-- `source` is additive, so a database created before it migrates rather than
-- failing on the next insert. The default is the honest answer for every row
-- written while the column did not exist: nobody recorded which surface asked.
ALTER TABLE answer_log ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'unknown';

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

-- ------------------------------------------------------------ turn_traces
-- How one answer was produced, stage by stage. The twin of this table in
-- db/schema.sqlite.sql carries the rationale in full: it is the sibling of
-- answer_log rather than part of it, it holds no question, answer or passage
-- text, `session_id` is legitimately empty for a CLI caller, and `source`
-- carries no CHECK because an unrecognised surface must be recorded rather
-- than rejected — a rejected insert would be an observability write failing an
-- answer.
--
-- `attributes` is JSONB here and JSON-in-TEXT there, the same split the
-- two schemas already make for `notes` and `chunk_ids`. `started_at` is TEXT in
-- both, unlike answer_log.asked_at: the retention sweep compares it to a cutoff
-- string and it round-trips to the caller verbatim, so both adapters have to
-- agree on the bytes.
CREATE TABLE IF NOT EXISTS turn_traces (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id      TEXT    NOT NULL DEFAULT '',
    turn_id         TEXT    NOT NULL,
    trace_id        TEXT    NOT NULL,
    span_id         TEXT    NOT NULL,
    parent_span_id  TEXT    NOT NULL DEFAULT '',
    name            TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    duration_ms     BIGINT  NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'ok',
    source          TEXT    NOT NULL DEFAULT 'unknown',
    attributes      JSONB   NOT NULL DEFAULT '{}'::jsonb
);

-- Additive, so a database created before the column migrates rather than
-- failing on its next insert, exactly as answer_log.source does.
ALTER TABLE turn_traces ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'unknown';

-- The two access paths: reconstruct one answer, and replay one conversation.
CREATE INDEX IF NOT EXISTS turn_traces_trace ON turn_traces (trace_id);

-- Partial, because every CLI row shares the empty session id and the query this
-- index exists for never asks for those rows. Both dialects support this and
-- the repository already relies on a partial index in both — see
-- index_snapshots_one_active above.
CREATE INDEX IF NOT EXISTS turn_traces_session
    ON turn_traces (session_id, turn_id) WHERE session_id <> '';

-- The prune's own access path; retention is enforced inside the span write.
CREATE INDEX IF NOT EXISTS turn_traces_started_at ON turn_traces (started_at);

-- ------------------------------------------------------------------ sessions
-- Multi-turn conversation state, persistent across server restarts.
-- One row per active session: facts about the building, the pending question,
-- and the turn history. Idle sessions are swept on read/write.
--
-- `slots` holds the carried facts: substrate, location, exposure. Only these
-- three carry between turns; they are the three decision 10 names as load-bearing
-- and the three the router prints back as stated assumptions.
--
-- `pending` is the question waiting on a missing fact, or empty. When the user
-- provides the missing fact, the pending question is re-asked with the new slots
-- and then cleared.
--
-- `turns` is the conversation so far, capped at MAX_TURNS (8): list of
-- [question, answer[:400]]. Oldest first; stored as JSONB.
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT        PRIMARY KEY,
    audience            TEXT        NOT NULL DEFAULT 'public',
    touched             DOUBLE PRECISION NOT NULL,
    slots               JSONB       NOT NULL DEFAULT '{}'::jsonb,
    pending             TEXT        NOT NULL DEFAULT '',
    turns               JSONB       NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT sessions_audience_ck CHECK (audience IN ('public', 'trade', 'staff'))
);

-- Touch-based LRU eviction: find and sweep expired sessions on read/write.
CREATE INDEX IF NOT EXISTS sessions_touched ON sessions (touched);
