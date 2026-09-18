"""`/ready` over HTTP, and the distinction from `/health` that it exists for.

`/health` answers "is this process running" and returns a static OK. That is
correct for liveness and was, until this endpoint existed, the only thing a
container orchestrator could ask -- so a container with no index, an index built
by a different embedding model, or no Ollama to reach reported *healthy* and
answered nothing. The compose healthcheck would have kept it in rotation.

So these tests are about the difference: the same process, the same port, one
path saying OK and the other saying 503 with the reason. They run a real
threaded server against a temporary store, because the class attribute carrying
the store path is exactly the kind of thing a library-level test cannot check.

The last test is a privacy one and is not incidental. This endpoint is
unauthenticated -- `assistant/infrastructure/trace.py` declines to be an endpoint at all for
that reason -- and readiness only qualifies because what it publishes is
operational state. A filesystem path or a database host is not, so `/ready`
must not print one even though `health.check()` collects it.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.infrastructure import health, ollama
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.indexing.index import CHUNKING_VERSION                       # noqa: E402
from assistant.knowledge.model import Chunk, Document, DocumentVersion, Snapshot
from assistant.knowledge.store import SQLiteKnowledgeRepository              # noqa: E402
from assistant.knowledge.store.factory import open_repository                # noqa: E402
from assistant.interfaces.ui import (  # noqa: E402
    Handler,
)

URL = "https://example.invalid/solo"
DIMS = 1024

_UNSET = object()


@pytest.fixture(autouse=True)
def no_ambient_postgres(monkeypatch):
    """A developer's DSN in the environment must not change what these prove."""
    monkeypatch.delenv("ASSISTANT_POSTGRES_DSN", raising=False)
    monkeypatch.delenv(health.VISION_DEMO_VAR, raising=False)


@pytest.fixture(autouse=True)
def restore_the_handler_class():
    """`main()` sets these on the class, so a test that starts a server leaks."""
    names = ("assistant", "meta", "audiences", "sessions", "db_path")
    saved = {name: Handler.__dict__.get(name, _UNSET) for name in names}
    yield
    left = Handler.__dict__.get("assistant", _UNSET)
    if left is not _UNSET and left is not saved["assistant"]:
        left.repo.close()
    for name, value in saved.items():
        if value is _UNSET:
            if name in Handler.__dict__:
                delattr(Handler, name)
        else:
            setattr(Handler, name, value)


def build_index(path: Path, embedding_model: str = "", chunks: int = 2) -> Path:
    """A store with one document, one active version and an active snapshot."""
    repo = SQLiteKnowledgeRepository(path)
    try:
        repo.publish(
            [Document(canonical_url=URL, title="Solo datasheet",
                      document_type="datasheet", authority=1, product="Solo",
                      link_text="Datasheet")],
            [DocumentVersion(canonical_url=URL, version=1, content_hash="h1",
                             source_path="cache/solo.pdf",
                             first_seen_at="2026-01-01",
                             fetched_at="2026-01-01T00:00:00Z",
                             checked_at="2026-01-01T00:00:00Z")],
            [Chunk(canonical_url=URL, version=1, chunk_index=i, section="Mixing",
                   content=f"passage {i}", product="Solo",
                   document_type="datasheet", authority=1,
                   embedding=[0.0] * DIMS) for i in range(chunks)],
            Snapshot(snapshot_id="snap-ready", created_at="2026-01-01T00:00:00Z",
                     embedding_model=embedding_model or ollama.EMBED_MODEL,
                     embedding_dimensions=DIMS,
                     chunking_version=CHUNKING_VERSION, document_count=1,
                     chunk_count=chunks, notes={"products": ["Solo"]}))
    finally:
        repo.close()
    return path


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Start a real server against a named store, as `main()` would."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: [0.0] * DIMS)

    started: list = []

    def start(db: Path, pulled=("__both__",)):
        tags = ([ollama.EMBED_MODEL, ollama.GENERATION_MODEL]
                if pulled == ("__both__",) else list(pulled))
        if isinstance(pulled, Exception):
            monkeypatch.setattr(ollama, "available",
                                lambda: (_ for _ in ()).throw(pulled))
        else:
            monkeypatch.setattr(ollama, "available", lambda: tags)

        repo = open_repository(str(db), dsn="", thread_safe=True)
        Handler.assistant = Assistant(repo)
        Handler.meta = "test"
        Handler.audiences = ("public",)
        Handler.db_path = str(db)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        started.append((httpd, thread, repo))
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start

    for httpd, thread, repo in started:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        repo.close()


def get(base: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


# ------------------------------------------------- liveness is not readiness


def test_a_ready_server_answers_200_with_every_check_true(tmp_path, serve):
    """The one state in which the container should be taking traffic."""
    base = serve(build_index(tmp_path / "index" / "knowledge.db"))

    status, body = get(base, "/ready")
    report = json.loads(body)

    assert status == 200
    assert report["ready"] is True
    assert all(report["checks"].values())
    assert report["snapshot"] == "snap-ready"
    assert report["chunks"] == 2


def test_health_says_ok_while_ready_says_503_when_ollama_goes_away(
        tmp_path, serve, monkeypatch):
    """The whole reason this endpoint exists, asserted as one comparison.

    A fault that is present *before* start does not need this endpoint: the
    engine refuses to construct on an absent or mismatched index, so the
    process exits rather than serving. What it cannot refuse is a fault that
    arrives afterwards, and Ollama being stopped is the common one -- a laptop
    sleeps, a container is restarted, someone frees the memory. The process
    stays perfectly alive and can no longer embed a question, so `/health`
    answers OK and must not be the thing an orchestrator reads.
    """
    base = serve(build_index(tmp_path / "index" / "knowledge.db"))
    assert get(base, "/ready")[0] == 200            # ready before the fault

    def stopped():
        raise ollama.OllamaUnavailable(
            "Ollama is not answering. Run `ollama serve`.")

    monkeypatch.setattr(ollama, "available", stopped)

    assert get(base, "/health") == (200, "OK\n")

    status, body = get(base, "/ready")
    report = json.loads(body)

    assert status == 503
    assert report["ready"] is False
    assert report["checks"]["ollama_reachable"] is False
    assert "ollama serve" in report["error"]


def test_a_configuration_change_under_a_running_server_fails_the_probe(
        tmp_path, serve, monkeypatch):
    """Querying an index with another embedding model returns confident nonsense.

    The engine verifies the release when it starts, which catches this on a
    restart and not on a machine where the variable was exported into a shell
    that is already serving. The probe closes that window.
    """
    base = serve(build_index(tmp_path / "index" / "knowledge.db"))
    monkeypatch.setattr(ollama, "EMBED_MODEL", "nomic-embed-text")

    status, body = get(base, "/ready")
    report = json.loads(body)

    assert status == 503
    assert report["checks"]["embedding_model_matches_index"] is False
    assert "rebuild the index" in report["error"]


def test_a_model_deleted_under_a_running_server_fails_the_probe(
        tmp_path, serve, monkeypatch):
    """A reachable Ollama that no longer holds the model composes nothing."""
    base = serve(build_index(tmp_path / "index" / "knowledge.db"))
    monkeypatch.setattr(ollama, "available", lambda: ["llama3.2:1b"])

    status, body = get(base, "/ready")
    report = json.loads(body)

    assert status == 503
    assert report["checks"]["ollama_reachable"] is True
    assert report["checks"]["generation_model_pulled"] is False


def test_an_empty_index_stops_the_server_starting_rather_than_serving(
        tmp_path, monkeypatch):
    """The complement: a fault present at start is refused, not reported.

    Worth asserting beside the endpoint, because it is what makes the endpoint
    narrow. If this ever regressed into a warning, `/ready` would become the
    only thing standing between an empty store and a confident empty answer.
    """
    monkeypatch.setattr(ollama, "available",
                        lambda: [ollama.EMBED_MODEL, ollama.GENERATION_MODEL])
    path = tmp_path / "index" / "knowledge.db"
    SQLiteKnowledgeRepository(path).close()

    repo = open_repository(str(path), dsn="", thread_safe=True)
    try:
        with pytest.raises(Exception, match="python -m assistant.indexing.index"):
            Assistant(repo)
    finally:
        repo.close()


# ------------------------------------------------------------ the text form


def test_the_text_form_is_the_table_the_command_line_prints(tmp_path, serve):
    """One renderer, so the page and the CLI cannot drift into disagreeing."""
    base = serve(build_index(tmp_path / "index" / "knowledge.db"))

    status, body = get(base, "/ready?format=text")

    assert status == 200
    assert body.startswith("READY")
    for row in ("application", "database", "knowledge index", "active snapshot",
                "embedding model", "generation model", "Ollama"):
        assert row in body


# ------------------------------------------------------------- what it hides


def test_the_endpoint_never_publishes_where_the_store_is(tmp_path, serve):
    """Unauthenticated: reachable is operational state, the path is not.

    `health.check()` collects `store_target` because an operator at a terminal
    needs it. This endpoint is reachable by anyone who can reach the page, so
    the field is filtered out rather than merely left unmentioned -- a test on
    the filter, not on the current contents of the dictionary.
    """
    db = build_index(tmp_path / "index" / "knowledge.db")
    base = serve(db)

    _status, body = get(base, "/ready")
    report = json.loads(body)

    assert "store_target" not in report
    assert "store" not in report
    assert str(db) not in body
    assert "knowledge.db" not in body
    # And the collected report does still carry it, so this proves a filter
    # rather than an absence.
    assert health.check(str(db))["store_target"] == str(db)


def test_a_password_never_reaches_the_report(monkeypatch):
    """A DSN is the one configured value that is a secret."""
    dsn = "postgresql://assistant:hunter2@db:5432/assistant"

    class Unreachable:
        def __init__(self, dsn, apply_schema=True):
            raise RuntimeError(f"could not connect to {dsn}")

    import types
    module = types.ModuleType("assistant.knowledge.store.postgres")
    module.PostgresKnowledgeRepository = Unreachable
    monkeypatch.setitem(sys.modules, "assistant.knowledge.store.postgres", module)

    report = health.check(dsn=dsn)

    assert report["ready"] is False
    assert "hunter2" not in json.dumps(report)
    assert "***@db:5432/assistant" in report["error"]


# ------------------------------------------------------------------- vision


def test_vision_is_reported_but_does_not_decide_readiness(tmp_path, monkeypatch):
    """Images are roadmap, so a machine without the model still answers text."""
    monkeypatch.setattr(ollama, "available",
                        lambda: [ollama.EMBED_MODEL, ollama.GENERATION_MODEL])
    monkeypatch.setattr("assistant.answering.vision.VISION_MODEL", "qwen3-vl:4b")

    report = health.check(str(build_index(tmp_path / "index" / "knowledge.db")))

    assert report["vision"] == {"model": "qwen3-vl:4b", "pulled": False,
                                "required": False}
    assert "vision_model_pulled" not in report["checks"]
    assert report["ready"] is True


def test_the_image_demo_makes_the_vision_model_load_bearing(tmp_path, monkeypatch):
    """Announced as part of the demonstration, a missing model is a fault."""
    monkeypatch.setenv(health.VISION_DEMO_VAR, "1")
    monkeypatch.setattr(ollama, "available",
                        lambda: [ollama.EMBED_MODEL, ollama.GENERATION_MODEL])
    monkeypatch.setattr("assistant.answering.vision.VISION_MODEL", "qwen3-vl:4b")

    report = health.check(str(build_index(tmp_path / "index" / "knowledge.db")))

    assert report["vision"]["required"] is True
    assert report["checks"]["vision_model_pulled"] is False
    assert report["ready"] is False
    assert "ASSISTANT_VISION_DEMO" in health.summary(report)
