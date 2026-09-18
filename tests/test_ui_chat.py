"""Chat UI integration tests for the multi-turn interface.

Tests the new chat-based UI replacing the old single-turn diagnostics page.
Covers: landing page, message flow, citations, source disclosure, diagnostics,
image upload, new chat reset, and audience display.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import observability as obs, ollama
from assistant import ui
from assistant.engine import Assistant
from assistant.indexing.index import CHUNKING_VERSION
from assistant.model import (
    Caveat, Chunk, Document, DocumentVersion, Snapshot,
)
from assistant.store import SQLiteKnowledgeRepository
from assistant.store.factory import open_repository
from assistant.session import SessionStore
from assistant.ui import SESSION_COOKIE, Handler

DIMS = 1024
SOLO = "https://example.invalid/solo"
DURO = "https://example.invalid/duro"
CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}


def unit(axis: int) -> list[float]:
    v = [0.0] * DIMS
    v[axis] = 1.0
    return v


def between(a: int, b: int) -> list[float]:
    """A query halfway between two axes, so both documents clear the gate."""
    v = [0.0] * DIMS
    v[a] = v[b] = 0.5 ** 0.5
    return v


def build_repo(tmp_path, snapshot_id: str = "snap-chat-test",
               revision: str = "a") -> SQLiteKnowledgeRepository:
    """Build a test repository with Solo and Duro datasheets."""
    documents = [
        Document(canonical_url=SOLO, title="Solo datasheet",
                 document_type="datasheet", authority=1, product="Solo",
                 link_text="Solo Datasheet"),
        Document(canonical_url=DURO, title="Duro datasheet",
                 document_type="datasheet", authority=1, product="Duro",
                 link_text="Duro Datasheet"),
    ]
    versions = [
        DocumentVersion(canonical_url=SOLO, version=1, content_hash=f"h1-{revision}",
                        source_path="cache/solo.pdf", fetched_at="2026-01-01"),
        DocumentVersion(canonical_url=DURO, version=1, content_hash=f"h2-{revision}",
                        source_path="cache/duro.pdf", fetched_at="2026-01-01"),
    ]
    chunks = [
        Chunk(canonical_url=SOLO, version=1, chunk_index=0, section="Mixing",
              content="Mix Solo with 5-6 litres of clean water per 25 kg sack.",
              product="Solo", document_type="datasheet", authority=1,
              source_date="2024-07-01", embedding=unit(0)),
        Chunk(canonical_url=DURO, version=1, chunk_index=0, section="Coverage",
              content="Duro covers approximately 2.5 m2 per 25 kg bag at 11 mm thickness.",
              product="Duro", document_type="datasheet", authority=1,
              source_date="2024-07-01", embedding=unit(1)),
    ]

    snapshot = Snapshot(
        snapshot_id=snapshot_id, created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=len(documents),
        chunk_count=len(chunks),
        notes={"products": ["Solo", "Duro"], "colours": ["York"],
               "merchants": [], "contact": CONTACT, "excluded": []})

    repo = SQLiteKnowledgeRepository(str(tmp_path / "index" / "knowledge.db"))
    repo.publish(documents, versions, chunks, snapshot, [])
    return repo


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Start a test HTTP server with the chat UI."""
    # Mock Ollama to avoid needing a running instance
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: ("Mix Solo with 5-6 litres of clean water per "
                           "25 kg sack [1].", 0.01))

    obs.configure(None)  # suppress logging

    # Build repo using the same pattern as existing tests
    build_repo(tmp_path).close()  # write the index, then let go of it

    # Open it for the server, just like ui.main() does
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    Handler.assistant = Assistant(repo, source="test")
    Handler.sessions = SessionStore()
    Handler.uploads = ui.UploadBudget()
    Handler.audiences = ("public", "trade", "staff")
    Handler.meta = "test"

    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    repo.close()


def get(server: str, path: str, **params) -> tuple[int, str]:
    """Make a GET request to the server."""
    if params:
        path += "?" + urllib.parse.urlencode(params)
    url = server + path
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


# ============================================================================
# Chat UI Tests
# ============================================================================

def test_landing_page_on_first_load(server):
    """Landing page is shown on first load with no question."""
    status, body = get(server, "/")

    assert status == 200
    assert "How can I help?" in body, "landing page should show"
    assert "Ask a question" in body, "input placeholder should be visible"


def test_question_and_answer_appear_in_response(server):
    """Asking a question returns an answer in the page."""
    status, body = get(server, "/", q="How much water does Solo need per bag")

    assert status == 200
    # The answer should contain the question in the input field or in a message
    # and should contain cited passages
    assert "5-6 litres" in body or "5-6 litre" in body, "answer should cite the datasheet"


def test_citations_are_rendered_as_spans(server):
    """Citation markers are rendered as clickable spans."""
    status, body = get(server, "/", q="How much water does Solo need per bag")

    assert status == 200
    # Should have citation spans
    assert 'class="citation"' in body, "citations should be rendered as spans"
    assert "[1]" in body, "citation marker should appear"


def test_sources_disclosure_in_answer(server):
    """Answer includes collapsible sources section."""
    status, body = get(server, "/", q="How much water does Solo need per bag")

    assert status == 200
    assert "Sources" in body or "sources" in body, "sources disclosure should be present"
    assert "Solo" in body, "product name should appear in sources"


def test_diagnostics_disclosure_in_answer(server):
    """The routing view is the operator's, behind `v=1`, not a customer's."""
    status, body = get(server, "/", q="How much water does Solo need per bag", v="1")

    assert status == 200
    assert "Why this answer?" in body, "diagnostics disclosure should be present"


def test_no_diagnostics_disclosure_for_a_normal_visitor(server):
    """The demo page answers the question without explaining its own routing."""
    status, body = get(server, "/", q="How much water does Solo need per bag")

    assert status == 200
    # Asserted against the rendered markup rather than the whole document: the
    # caption and the class name both still occur inside the page's own script,
    # which builds the panel when an operator asks for it. What matters is that
    # this response rendered no panel, and that the script's gate is shut.
    assert 'class="diagnostics-disclosure"' not in body
    assert 'class="diagnostics-list"' not in body
    assert "renderDiagnostics" in body, "the operator's renderer still ships"
    assert "DIAGNOSTICS_VISIBLE" in body
    assert "get('v') === '1'" in body, "the client gate reads the same switch"


def test_refusal_shows_landing_info(server):
    """A refusal still shows relevant information and contact line."""
    status, body = get(server, "/", q="What is the pot life of Solo in hours")

    assert status == 200
    # Refusal path should still have a message
    assert "I could not find" in body or "not stated" in body or "refused" in body, \
        "refusal should explain the situation"


def test_audience_display_in_header(server):
    """Audience badge is displayed in the header."""
    status, body = get(server, "/", a="public")

    assert status == 200
    assert "public" in body.lower(), "audience should be displayed"


def test_new_chat_button_present(server):
    """New chat button is visible in the interface."""
    status, body = get(server, "/")

    assert status == 200
    assert "New chat" in body, "New chat button should be present"


def test_json_endpoint_returns_structured_answer(server):
    """The /ask endpoint returns JSON for programmatic use."""
    status, body = get(server, "/ask", q="How much water does Solo need per bag")

    assert status == 200
    data = json.loads(body)
    assert "parts" in data, "JSON response should have parts"
    assert len(data["parts"]) > 0, "should have at least one answer part"
    assert "text" in data["parts"][0], "part should have text"
    assert "sources" in data["parts"][0], "part should have sources"


def test_citations_in_json_response(server):
    """JSON response includes citation markers in the text."""
    status, body = get(server, "/ask", q="How much water does Solo need per bag")

    assert status == 200
    data = json.loads(body)
    text = data["parts"][0]["text"]
    assert "[1]" in text or "[2]" in text, "text should have citation markers"


def test_input_field_has_placeholder(server):
    """Input field has a helpful placeholder."""
    status, body = get(server, "/")

    assert status == 200
    assert "Ask a question" in body, "input should have placeholder text"


def test_upload_button_present(server):
    """Image upload button (+) is visible."""
    status, body = get(server, "/")

    assert status == 200
    # The button might be text "+" or aria-labeled
    assert "+" in body or "upload" in body.lower() or "attachment" in body.lower(), \
        "upload button should be present"


def test_grounding_message_in_ui(server):
    """UI displays message that responses are grounded in Lime Green material."""
    status, body = get(server, "/")

    assert status == 200
    assert "Lime Green" in body, "should mention Lime Green"
    assert "source" in body.lower() or "grounded" in body.lower(), \
        "should indicate source grounding"
