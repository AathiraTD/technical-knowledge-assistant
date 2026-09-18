"""Readiness, tested on the four ways a live process can still answer nothing.

Liveness and readiness are different things, and conflating them is how a
service stays in rotation while returning nothing useful. The process starts
perfectly well with no index, with an index built by a different embedding
model, with no chunks in it, and with Ollama stopped. Each of those has to come
back NOT READY with a reason someone can act on, and the mismatch one has to
fail closed rather than degrade quietly: querying an index with a different
embedding model than built it returns confident nonsense.

Every test here builds its own SQLite store in a temporary directory and fakes
Ollama, so the suite reports on the code rather than on whichever models happen
to be pulled on the machine running it.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from assistant.indexing.index import CHUNKING_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.infrastructure import health, ollama
from assistant.knowledge.model import Chunk, Document, DocumentVersion, Snapshot
from assistant.knowledge.store import SQLiteKnowledgeRepository              # noqa: E402

URL = "https://example.invalid/solo-datasheet"


@pytest.fixture(autouse=True)
def no_ambient_postgres(monkeypatch):
    """A developer's DSN in the environment must not change what these prove."""
    monkeypatch.delenv("ASSISTANT_POSTGRES_DSN", raising=False)


def build_store(tmp_path, embedding_model: str = "", chunks: int = 2) -> Path:
    """A store with one document, one active version and an active snapshot."""
    path = tmp_path / "index" / "knowledge.db"
    repo = SQLiteKnowledgeRepository(path)
    try:
        documents = [Document(canonical_url=URL, title="Solo datasheet",
                              document_type="datasheet", authority=1,
                              product="Solo", link_text="Datasheet")]
        versions = [DocumentVersion(canonical_url=URL, version=1, content_hash="h1",
                                    source_path="cache/solo.pdf",
                                    first_seen_at="2026-01-01",
                                    fetched_at="2026-01-01T00:00:00Z",
                                    checked_at="2026-01-01T00:00:00Z")]
        made = [Chunk(canonical_url=URL, version=1, chunk_index=i,
                      section="Mixing", content=f"passage {i}", product="Solo",
                      document_type="datasheet", authority=1,
                      embedding=[0.0] * 1024) for i in range(chunks)]
        snapshot = Snapshot(
            snapshot_id="snap-test", created_at="2026-01-01T00:00:00Z",
            embedding_model=embedding_model or ollama.EMBED_MODEL,
            embedding_dimensions=ollama.EMBED_DIMENSIONS,
            chunking_version=CHUNKING_VERSION, document_count=1, chunk_count=chunks,
            notes={"products": ["Solo"]})
        repo.publish(documents, versions, made, snapshot)
    finally:
        repo.close()
    return path


def empty_store(tmp_path) -> Path:
    """A store whose schema exists and which has never been indexed."""
    path = tmp_path / "index" / "empty.db"
    SQLiteKnowledgeRepository(path).close()
    return path


@pytest.fixture
def ollama_has(monkeypatch):
    """Answer `ollama.available()` from a list rather than from the machine."""
    def install(tags):
        def available():
            if isinstance(tags, Exception):
                raise tags
            return list(tags)

        monkeypatch.setattr(ollama, "available", available)

    return install


def both_models() -> list[str]:
    return [ollama.EMBED_MODEL, ollama.GENERATION_MODEL]


# ------------------------------------------------------------------ the good


def test_ready_when_the_store_snapshot_chunks_and_models_are_all_there(
        tmp_path, ollama_has):
    """The one state in which the container should be taking traffic."""
    ollama_has(both_models())
    report = health.check(str(build_store(tmp_path)))

    assert report["ready"] is True
    assert report["store"] == "sqlite"
    assert report["snapshot"] == "snap-test"
    assert report["chunks"] == 2
    assert all(report["checks"].values())
    assert "error" not in report


# ------------------------------------------------------------- the four bad


def test_no_active_snapshot_names_the_command_that_builds_one(tmp_path, ollama_has):
    """A process with no index is alive and useless, and has to say which."""
    ollama_has(both_models())
    report = health.check(str(empty_store(tmp_path)))

    assert report["ready"] is False
    assert report["checks"]["active_snapshot"] is False
    assert "python -m assistant.indexing.index" in report["error"]


def test_an_index_built_by_another_model_is_not_ready(tmp_path, ollama_has):
    """Querying an index with the wrong embedding model returns confident nonsense."""
    ollama_has(["nomic-embed-text", ollama.GENERATION_MODEL])
    report = health.check(str(build_store(tmp_path, embedding_model="nomic-embed-text")))

    assert report["ready"] is False
    assert report["checks"]["embedding_model_matches_index"] is False
    assert "nomic-embed-text" in report["error"]
    assert ollama.EMBED_MODEL in report["error"]
    assert "rebuild the index" in report["error"]


def test_an_unreachable_ollama_is_not_ready(tmp_path, ollama_has):
    """Nothing can be embedded or composed, so readiness is false, not degraded."""
    ollama_has(ollama.OllamaUnavailable("Ollama is not answering. Run `ollama serve`."))
    report = health.check(str(build_store(tmp_path)))

    assert report["ready"] is False
    assert report["checks"]["ollama_reachable"] is False
    assert "ollama serve" in report["error"]


def test_a_model_that_is_not_pulled_is_not_ready(tmp_path, ollama_has):
    """A reachable Ollama holding neither model answers no question at all."""
    ollama_has(["llama3.2:1b"])
    report = health.check(str(build_store(tmp_path)))

    assert report["ready"] is False
    assert report["checks"]["ollama_reachable"] is True
    assert report["checks"]["embedding_model_pulled"] is False
    assert report["checks"]["generation_model_pulled"] is False


def test_an_index_with_no_chunks_is_not_ready(tmp_path, ollama_has):
    """A snapshot that published nothing retrieves nothing to cite."""
    ollama_has(both_models())
    report = health.check(str(build_store(tmp_path, chunks=0)))

    assert report["ready"] is False
    assert report["checks"]["chunks_present"] is False


def test_an_unreachable_store_is_reported_with_its_reason(tmp_path, monkeypatch):
    """A locked or missing database file must name itself, not raise into the logs."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("unable to open database file")

    monkeypatch.setattr("assistant.knowledge.store.SQLiteKnowledgeRepository", boom)
    report = health.check(str(tmp_path / "knowledge.db"))

    assert report["ready"] is False
    assert report["checks"]["store_reachable"] is False
    assert "unable to open database file" in report["error"]


# --------------------------------------------------- the deployment adapter


def test_a_configured_dsn_checks_postgres_instead_of_sqlite(monkeypatch, ollama_has):
    """In the Compose stack the store under test is Postgres, not the shipped file."""
    ollama_has(both_models())

    class FakePostgresRepository:                # deliberately has no close()
        def __init__(self, dsn, apply_schema=True):
            self.dsn = dsn

        def snapshot(self):
            return Snapshot(snapshot_id="snap-pg", created_at="2026-01-01",
                            embedding_model=ollama.EMBED_MODEL,
                            embedding_dimensions=ollama.EMBED_DIMENSIONS,
                            chunking_version=CHUNKING_VERSION, document_count=1,
                            chunk_count=7)

    module = types.ModuleType("assistant.knowledge.store.postgres")
    module.PostgresKnowledgeRepository = FakePostgresRepository
    monkeypatch.setitem(sys.modules, "assistant.knowledge.store.postgres", module)

    report = health.check(dsn="postgresql://db:5432/assistant")

    assert report["store"] == "postgresql+pgvector"
    assert report["chunks"] == 7
    assert report["ready"] is True


def test_a_dsn_in_the_environment_is_used_when_none_is_passed(monkeypatch, ollama_has):
    """Configuration is externalised, so the env var has to be read."""
    ollama_has(both_models())

    class Unreachable:
        def __init__(self, dsn, apply_schema=True):
            raise RuntimeError(f"could not connect to {dsn}")

    module = types.ModuleType("assistant.knowledge.store.postgres")
    module.PostgresKnowledgeRepository = Unreachable
    monkeypatch.setitem(sys.modules, "assistant.knowledge.store.postgres", module)
    monkeypatch.setenv("ASSISTANT_POSTGRES_DSN", "postgresql://db:5432/assistant")

    report = health.check()

    assert report["ready"] is False
    assert "could not connect to postgresql://db:5432/assistant" in report["error"]


# ------------------------------------------------------------ the entry point


def test_main_exits_zero_when_ready(tmp_path, ollama_has, capsys):
    """An orchestrator reads the exit code, not the prose."""
    ollama_has(both_models())
    code = health.main(["--db", str(build_store(tmp_path))])

    assert code == 0
    assert "ready" in capsys.readouterr().out


def test_main_exits_one_and_prints_the_reason_when_not_ready(
        tmp_path, ollama_has, capsys):
    """A container that is not ready must not be put into rotation."""
    ollama_has(both_models())
    code = health.main(["--db", str(empty_store(tmp_path))])

    printed = capsys.readouterr().out
    assert code == 1
    assert "NOT READY" in printed
    assert "active snapshot" in printed
    assert "python -m assistant.indexing.index" in printed


def test_json_output_is_parseable(tmp_path, ollama_has, capsys):
    """The JSON form is what a readiness probe and a log collector consume."""
    ollama_has(both_models())
    code = health.main(["--db", str(build_store(tmp_path)), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["ready"] is True
    assert report["checks"]["embedding_model_matches_index"] is True
    assert report["embedding_model"] == ollama.EMBED_MODEL
