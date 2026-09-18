"""The web page under its own server, which nothing else tests.

Every other test calls the library directly. That is the right shape for the
guardrails, and it is exactly why this file exists: `ThreadingHTTPServer` hands
each request to a new thread, and a store opened on the main thread is not
usable from one. No amount of library-level testing finds that, because the
library is never the thing that crosses the boundary.

The bug this was written to reproduce is real and was live: any question that
touched the store raised

    sqlite3.ProgrammingError: SQLite objects created in a thread can only be
    used in that same thread

so the page answered policy questions and broke on everything else.

Runs a real server on an ephemeral port against a temporary store, and never
touches Ollama: the embedding call is answered locally and generation is
replaced, because what is under test is the plumbing rather than the model.

It now also covers what `main()` does before any of that — binding, reporting
an address that can actually be opened, refusing to start on an index it cannot
query, and stopping on Ctrl-C — and what the rendered page puts in front of a
person, because the page is a deliverable of the submission rather than a
convenience over the library.

Three of those deserve naming. The diagnostics view is the only place a person
sees which router step produced an answer and which check refused one, so it is
asserted on its values rather than on a status code. The page echoes the
question back into an input element, which makes it the one HTML escaping
boundary in the system. And the engine re-verifies the release on every
request, so an index rebuilt with a different embedding model under a running
page has to stop it answering rather than quietly return the wrong neighbours.
"""

from __future__ import annotations

import contextlib
import html
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import observability as obs, ollama                 # noqa: E402
from assistant import ui                                           # noqa: E402
from assistant.engine import Assistant                             # noqa: E402
from assistant.indexing.index import CHUNKING_VERSION                       # noqa: E402
from assistant.model import (                                      # noqa: E402
    Caveat, Chunk, Document, DocumentVersion, Snapshot,
)
from assistant.store import SQLiteKnowledgeRepository              # noqa: E402
from assistant.store.factory import open_repository                # noqa: E402
from assistant.session import SessionStore                         # noqa: E402
from assistant.ui import SESSION_COOKIE, Handler                   # noqa: E402

DIMS = 1024
SOLO = "https://example.invalid/solo"
DURO = "https://example.invalid/duro"
STAFF = "fixture://staff/margin"
CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}
DURO_TEXT = "Duro covers approximately 2.5 m2 per 25 kg bag at 11 mm thickness."

# Two documents have to clear the threshold before the router chooses Compose;
# one document is a factual ask and takes Extract, where no model runs. A query
# halfway between the two axes scores 0.707 against each.
COMPOSE_QUESTION = "What water and coverage does Solo have"


def unit(axis: int) -> list[float]:
    v = [0.0] * DIMS
    v[axis] = 1.0
    return v


def between(a: int, b: int) -> list[float]:
    """A query halfway between two axes, so both documents clear the gate."""
    v = [0.0] * DIMS
    v[a] = v[b] = 0.5 ** 0.5
    return v


def build_repo(tmp_path, *, embedding_model: str = "",
               snapshot_id: str = "snap-ui-test", revision: str = "a",
               second_product: bool = False) -> SQLiteKnowledgeRepository:
    """One public datasheet and one staff-only note, with an active snapshot.

    The keyword arguments exist for two shapes the defaults cannot produce, and
    both are properties rather than conveniences. `embedding_model` builds an
    index the running engine cannot query, which is the state a rebuild with a
    changed model leaves behind. `second_product` adds a public document, which
    is what moves the router off Extract and onto Compose — the only path where
    a model runs and the only one that records a generation time.
    """
    documents = [
        Document(canonical_url=SOLO, title="Solo datasheet",
                 document_type="datasheet", authority=1, product="Solo",
                 link_text="Solo Datasheet"),
        Document(canonical_url=STAFF, title="Internal margin note",
                 document_type="knowledge_base", authority=4, audience="staff",
                 product="Solo", link_text="Internal margin note"),
    ]
    versions = [
        DocumentVersion(canonical_url=SOLO, version=1, content_hash=f"h1-{revision}",
                        source_path="cache/solo.pdf", fetched_at="2026-01-01"),
        DocumentVersion(canonical_url=STAFF, version=1, content_hash=f"h2-{revision}",
                        source_path="fixtures/staff.json", fetched_at="2026-01-01"),
    ]
    chunks = [
        Chunk(canonical_url=SOLO, version=1, chunk_index=0, section="Mixing",
              content="Mix Solo with 5-6 litres of clean water per 25 kg sack.",
              product="Solo", document_type="datasheet", authority=1,
              source_date="2024-07-01", embedding=unit(0)),
        Chunk(canonical_url=STAFF, version=1, chunk_index=0,
              section="Internal margin",
              content="The internal margin on Solo is 42 per cent at list price.",
              product="Solo", document_type="knowledge_base", authority=4,
              audience="staff", source_date="2026-01-01", embedding=unit(0)),
    ]
    if second_product:
        documents.append(
            Document(canonical_url=DURO, title="Duro datasheet",
                     document_type="datasheet", authority=1, product="Duro",
                     link_text="Duro Datasheet"))
        versions.append(
            DocumentVersion(canonical_url=DURO, version=1,
                            content_hash=f"h3-{revision}",
                            source_path="cache/duro.pdf", fetched_at="2026-01-01"))
        chunks.append(
            Chunk(canonical_url=DURO, version=1, chunk_index=0, section="Coverage",
                  content=DURO_TEXT, product="Duro", document_type="datasheet",
                  authority=1, source_date="2024-07-01", embedding=unit(1)))

    snapshot = Snapshot(
        snapshot_id=snapshot_id, created_at="2026-01-01T00:00:00Z",
        embedding_model=embedding_model or ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=len(documents),
        chunk_count=len(chunks),
        notes={"products": ["Solo", "Duro"], "colours": ["York"],
               "merchants": ["The Lime Centre"], "contact": CONTACT})

    repo = SQLiteKnowledgeRepository(tmp_path / "index" / "knowledge.db")
    repo.publish(documents, versions, chunks, snapshot,
                 [Caveat(SOLO, "temperature", "Do not apply below 5 degrees C.",
                         "Mixing")])
    return repo


_UNSET = object()


@pytest.fixture(autouse=True)
def restore_the_handler_class():
    """`Handler` keeps its assistant, meta and audiences as class attributes.

    `main()` sets them on the class, not on an instance, so a test that starts a
    server leaves the next one pointed at a store in a deleted temporary
    directory — and leaves that store's connection open, which on Windows keeps
    the file locked. Put the class back, and close whatever was left behind.
    """
    names = ("assistant", "meta", "audiences", "sessions")
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


@pytest.fixture(autouse=True)
def detach_the_log():
    """Take the handler back off after every test.

    The page turns structured logging on by default, so `main()` binds a
    handler to whichever stream it was handed. Under capsys that stream belongs
    to one test, and a handler left attached writes into a buffer that no
    longer exists.
    """
    yield
    for handler in list(obs.logger.handlers):
        if getattr(handler, "_assistant_handler", False):
            obs.logger.removeHandler(handler)


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A real threaded server on an ephemeral port. Yields its base URL."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: ("Mix Solo with 5-6 litres of clean water per "
                           "25 kg sack [1].", 0.01))

    build_repo(tmp_path).close()          # write the index, then let go of it

    # Opened exactly as ui.main() opens it: this is the thing under test, and a
    # fixture that constructs the store its own way would prove nothing about
    # the server.
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    Handler.assistant = Assistant(repo)
    Handler.meta = "test"
    Handler.audiences = ("public",)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        repo.close()


def quoting(prompt: str, **_kwargs) -> tuple[str, float]:
    """A model that does the one thing this system wants: quote and cite.

    It reads the passages out of the prompt it was given and returns the first
    sentence of each with the marker it arrived under, so the six checks pass
    and the Compose path actually prints. Hard-coding an answer instead would
    pin the test to whichever order retrieval happened to return.
    """
    sentences = []
    for block in prompt.split("Passages:", 1)[-1].split("\n\n"):
        # `.strip()` first: the passage list opens with a newline, so the first
        # block began with one and its marker line landed in the wrong half of
        # the partition. The stub then quoted every passage but the first — an
        # answer that looked multi-source and cited one document.
        head, _, body = block.strip().partition("\n")
        head, body = head.strip(), body.strip()
        if head.startswith("[") and "]" in head and body:
            marker = head[:head.index("]") + 1]
            sentences.append(f"{body.split('. ')[0].rstrip('.')} {marker}.")
    return " ".join(sentences) or "Nothing was supplied.", 0.01


@pytest.fixture
def compose_server(tmp_path, monkeypatch):
    """The same server over two public documents, so the model runs.

    Compose is the only path where a model composes and the only one that
    records a generation time, so the diagnostics view cannot be tested whole
    without it. The query sits halfway between the two document axes, which
    puts both above the gate and the router on step 8.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: between(0, 1))
    monkeypatch.setattr(ollama, "generate", quoting)

    build_repo(tmp_path, second_product=True).close()
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    Handler.assistant = Assistant(repo)
    Handler.meta = "test"
    Handler.audiences = ("public",)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        repo.close()


@pytest.fixture
def db(tmp_path, monkeypatch) -> str:
    """An index on disk and no Ollama: what `main()` finds on a demo machine."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: ("Mix Solo with 5-6 litres of clean water per "
                           "25 kg sack [1].", 0.01))
    build_repo(tmp_path).close()
    return str(tmp_path / "index" / "knowledge.db")


@pytest.fixture
def bound(monkeypatch):
    """Every server `main()` constructs, recorded so a test can reach and stop it.

    `main()` binds its own socket and keeps the object to itself, which is
    right for a program and useless for a test: with `--port 0` the port is not
    known until after the bind. A recording subclass is the smallest seam that
    leaves the real server class, the real bind and the real request loop under
    test — nothing here replaces the server, it only remembers it.
    """
    made: list[ThreadingHTTPServer] = []
    real = ui.ThreadingHTTPServer

    class Recorded(real):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            made.append(self)

    monkeypatch.setattr(ui, "ThreadingHTTPServer", Recorded)
    yield made
    # server_close() only closes the socket and cannot block. shutdown() waits
    # on an event that serve_forever sets, so it hangs if serve_forever never
    # ran — it belongs to the test that started one, not to teardown.
    for server in made:
        server.server_close()


@contextlib.contextmanager
def serving(argv: list[str], bound: list):
    """`main()` on a thread, yielded once it actually answers a request."""
    outcome: list[int] = []
    thread = threading.Thread(target=lambda: outcome.append(ui.main(argv)),
                              daemon=True)
    thread.start()
    # Poll with a real request rather than for the bind. The socket listens from
    # construction, so a connection succeeds before serve_forever is running —
    # and shutdown() called in that window would never return.
    base, deadline = "", time.monotonic() + 30
    while time.monotonic() < deadline:
        if bound:
            base = f"http://127.0.0.1:{bound[0].server_address[1]}"
            with contextlib.suppress(OSError):
                if get(base, "/")[0] == 200:
                    break
        time.sleep(0.02)
    else:
        raise AssertionError("main() never served a request")
    try:
        yield base, outcome
    finally:
        bound[0].shutdown()
        thread.join(timeout=30)


def escaped_in(body: str, text: str) -> bool:
    """Is this text on the page, escaped the way the renderer escapes it?"""
    return html.escape(text, quote=True) in body


def get(base: str, path: str, **params) -> tuple[int, str]:
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


# ------------------------------------------------------- the thread boundary


def test_a_question_that_reaches_the_store_is_answered_rather_than_crashing(server):
    """The bug: the store was opened on one thread and used from another."""
    status, body = get(server, "/ask", q="How much water does Solo need per bag")
    assert status == 200, body
    assert "ProgrammingError" not in body
    payload = json.loads(body)
    assert payload["parts"], body
    assert "5-6 litres" in payload["parts"][0]["text"]


def test_the_html_page_answers_the_same_question(server):
    status, body = get(server, "/", q="How much water does Solo need per bag")
    assert status == 200
    assert "ProgrammingError" not in body
    assert "5-6 litres" in body


def test_concurrent_questions_all_succeed(server):
    """One request per thread is the server's normal mode, not an edge case."""
    questions = ["How much water does Solo need per bag",
                 "What is the coverage of Solo",
                 "How much does Solo cost",
                 "Where can I buy Solo"] * 3

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda q: get(server, "/ask", q=q), questions))

    for status, body in results:
        assert status == 200, body
        assert "ProgrammingError" not in body
        assert "Traceback" not in body


# --------------------------------------------------------------- the audience


def test_the_audience_cannot_be_raised_from_the_query_string(server):
    """A staff flag on a CLI is an assertion; on a URL it is self-service."""
    status, body = get(server, "/ask", q="What is the internal margin on Solo",
                       a="staff")
    assert status == 200, body
    payload = json.loads(body)
    text = " ".join(part["text"] for part in payload["parts"])
    cited = [source["url"] for part in payload["parts"]
             for source in part["sources"]]

    assert "42 per cent" not in text, "staff evidence reached a public caller"
    assert not any(url.startswith("fixture://") for url in cited), cited


def test_a_server_started_for_staff_can_see_staff_material(server, monkeypatch):
    """The filter must be a real boundary, not a blanket refusal."""
    monkeypatch.setattr(Handler, "audiences", ("staff",))
    status, body = get(server, "/ask", q="What is the internal margin on Solo")
    assert status == 200, body
    payload = json.loads(body)
    cited = [source["url"] for part in payload["parts"]
             for source in part["sources"]]
    assert any(url.startswith("fixture://") for url in cited), cited


# ------------------------------------------------------------ error handling


def test_an_unreachable_model_is_reported_rather_than_traced_back(server, monkeypatch):
    """The JSON endpoint had no handler for this and returned a traceback."""
    def unavailable(*_a, **_k):
        raise ollama.OllamaUnavailable("Ollama is not answering. Run `ollama serve`.")

    monkeypatch.setattr(ollama, "embed_one", unavailable)
    status, body = get(server, "/ask", q="How much water does Solo need per bag")

    assert status == 503, body
    assert "Traceback" not in body
    assert "ollama serve" in body


def test_an_unknown_path_is_a_clean_404(server):
    status, _body = get(server, "/does-not-exist")
    assert status == 404


def test_the_health_endpoint_returns_ok(server):
    """Health check endpoint for load balancers and orchestration."""
    status, body = get(server, "/health")
    assert status == 200
    assert body.strip() == "OK"


def test_the_page_without_a_question_invites_one_rather_than_answering(server):
    """The first thing an assessor sees. It must not look like a broken page."""
    status, body = get(server, "/")

    assert status == 200
    assert "How can I help?" in body, body
    assert "<h3>Sources</h3>" not in body, "an empty page cited a document"


def test_a_routed_question_prints_the_referral_with_nothing_to_cite(server):
    """Route is answered from the routing table, so there is no source list.

    Worth pinning on the page rather than only in the library: the renderer has
    to omit the Sources and caveat blocks entirely rather than print two empty
    headings, which is what a person would read as a missing citation.
    """
    status, body = get(server, "/", q="How much does Solo cost")

    assert status == 200
    assert "does not publish prices" in body, body
    assert "<h3>Sources</h3>" not in body, "a routed answer cited a document"
    assert "<ul class='caveats'>" not in body, "a routed answer printed a caveat list"


def test_a_cued_substrate_is_printed_on_the_page_as_the_callers_own_words(server):
    """Decision 10: a cued load-bearing slot is used, and the page must say so.

    An answer that silently used a substrate the caller never gave is the costly
    error the whole design exists to avoid, so the value has to be visible on
    the page and not only in the diagnostics an ordinary visitor never opens.

    What changed is the label, not the visibility. The caller wrote "on brick",
    so the page says the answer was given for brick *as they said it* - and does
    not file it under a heading reading "Assumed", which is what it used to do
    and which makes a system that listened look like one that guessed.
    """
    status, body = get(server, "/", q="How much water does Solo need on brick")

    assert status == 200
    assert "brick (substrate), as you said" in body, body
    assert "<h3>Assumed</h3>" not in body, "a value the caller gave was called a guess"


def test_the_page_reports_a_genuine_assumption_under_a_heading_that_says_so(server):
    """The other half: something nobody said still has to be flagged as assumed.

    An uncued inside/outside is answered for both options, and that *is* an
    assumption - so it keeps the heading the stated values lost.
    """
    status, body = get(server, "/ask", q="What is Solo used for on brick")
    payload = json.loads(body)
    part = payload["parts"][0]

    assert status == 200
    assert {"slot": "substrate", "value": "brick", "provenance": "stated"} in part["facts"]
    assumed = [f for f in part["facts"] if f["provenance"] == "assumed"]
    if assumed:                                    # per-option depends on the ask
        assert part["assumptions"], part


# ------------------------------------------------------- the diagnostics view


def test_the_diagnostics_view_names_the_step_the_reason_and_the_score(server):
    """The audit surface. Without it a path is a claim rather than a record."""
    status, body = get(server, "/", q="How much water does Solo need on brick",
                       v="1")

    assert status == 200
    assert "path      extract" in body, body
    assert "step      7" in body, "the router step is what makes the path explicable"
    assert "one document answers a factual question" in body
    assert "top score 1.0" in body
    # Escaped on the way out, because the slot dictionary is repr'd into HTML.
    assert "substrate" in body and "brick" in body
    assert "generated" not in body, "Extract runs no model and must claim no time"
    assert "checked" in body, "the checkbox has to come back on, or -v is one-shot"


def test_the_diagnostics_view_reports_how_long_generation_took(compose_server):
    """Compose is the only path a model runs on, and latency is the open question.

    Decision 7 leaves live-demonstration-versus-transcript resting on a measured
    warm compose, so the number has to reach the page rather than only the log.
    """
    status, body = get(compose_server, "/", q=COMPOSE_QUESTION, v="1")

    assert status == 200
    assert "path      compose" in body, body
    assert "step      8" in body
    assert "generated 0.01s" in body


def test_the_diagnostics_view_names_the_checks_that_refused_an_answer(
        compose_server, monkeypatch):
    """A refusal without its reason is indistinguishable from a broken system.

    Over-refusal is measured by which check fired, so the page has to print the
    failures rather than the bare word 'refused'.
    """
    # A figure that appears in no cited passage. Check 2 is what catches it, and
    # the diagnostics view is where an operator reads that it did.
    monkeypatch.setattr(ollama, "generate",
                        lambda *_a, **_k: ("Solo covers 50 m2 per bag [1].", 0.01))
    status, body = get(compose_server, "/", q=COMPOSE_QUESTION, v="1")

    assert status == 200
    assert 'class="tag refused">refused' in body, body
    assert "checks that failed:" in body
    assert "check 2" in body, "the check that fired was not named"
    assert "generated 0.01s" in body, "a failed compose still cost generation time"


# ----------------------------------------------------------- the HTML boundary


def test_a_question_containing_markup_is_escaped_rather_than_served(server):
    """The page echoes the question into an input: the one XSS boundary here.

    CLAUDE.md asks for boundary risks to be tested rather than reviewed, and
    this is the only place in the system where caller-supplied text is written
    into markup.
    """
    payload = "<script>alert('solo')</script>"
    status, body = get(server, "/", q=payload)

    assert status == 200
    assert payload not in body, "the question came back as live markup"
    assert "&lt;script&gt;" in body
    # The value sits inside a quoted attribute, so the quote characters have to
    # be escaped too - otherwise the attribute can be closed early and an event
    # handler appended after it.
    assert "alert(&#x27;solo&#x27;)" in body


def test_an_answer_carries_the_same_correlation_id_in_its_header_and_its_body(
        server):
    """The id is what turns a reported problem into a line in the log."""
    url = f"{server}/ask?" + urllib.parse.urlencode(
        {"q": "How much water does Solo need per bag"})
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        header = response.headers["X-Correlation-Id"]

    assert header, "no correlation id was returned"
    assert payload["correlation_id"] == header, "two ids for one request"


# ---------------------------------------------------- more of the error paths


def test_an_empty_question_is_answered_with_no_parts_rather_than_an_error(server):
    """A bookmarked /ask with the query dropped is a plausible arrival."""
    status, body = get(server, "/ask")

    assert status == 200, body
    assert json.loads(body)["parts"] == []


def test_an_unreachable_model_is_reported_on_the_page_rather_than_traced_back(
        server, monkeypatch):
    """The JSON branch is covered above; the page has its own error card."""
    def unavailable(*_a, **_k):
        raise ollama.OllamaUnavailable("Ollama is not answering. Run `ollama serve`.")

    monkeypatch.setattr(ollama, "embed_one", unavailable)
    status, body = get(server, "/", q="How much water does Solo need per bag")

    assert status == 200, "the page still renders; the card carries the reason"
    assert "ollama serve" in body
    assert "Traceback" not in body


def test_an_index_rebuilt_with_another_model_stops_the_page_answering(
        server, tmp_path):
    """Querying an index with the wrong embedding model returns confident nonsense.

    The engine re-verifies the release on every request rather than only at
    startup, so a rebuild underneath a running page has to stop it answering.
    This is the one way `IndexMismatch` reaches the request handler at all.
    """
    build_repo(tmp_path, embedding_model="some-other-embedding:1b",
               snapshot_id="snap-ui-rebuilt", revision="b").close()
    status, body = get(server, "/ask", q="How much water does Solo need per bag")

    assert status == 503, body
    assert "some-other-embedding:1b" in body, "the operator needs both model names"
    assert "Traceback" not in body


# ----------------------------------------------------------------- main()


def test_main_binds_serves_the_page_and_stops_cleanly(db, bound, capsys):
    """`python -m assistant.ui` is the whole demonstration, start to finish."""
    with serving(["--db", db, "--port", "0", "--no-browser"], bound) as (base, code):
        status, body = get(base, "/", q="How much water does Solo need per bag")
        assert status == 200, body
        assert "5-6 litres" in body

    assert code == [0], "main() did not return after the server was stopped"
    printed = capsys.readouterr().out
    # The port actually bound, not the one requested. They differ whenever the
    # port is left to the operating system, and the printed address was built
    # from the request — so it advertised http://127.0.0.1:0/ and the browser
    # it opens went nowhere.
    assert f"Lime Green technical assistant on {base}/" in printed, printed
    assert "/ask?q=" in printed, "the JSON endpoint is not advertised"


def test_main_states_the_index_the_models_and_the_threshold_on_the_page(db, bound):
    """The header is the honesty line: which release and which models answered."""
    with serving(["--db", db, "--port", "0", "--no-browser"], bound) as (base, _code):
        _status, body = get(base, "/")

    assert "2 documents" in body and "2 passages" in body, body
    assert "index built 2026-01-01" in body
    assert ollama.EMBED_MODEL in body and ollama.GENERATION_MODEL in body
    assert "abstention threshold 0.45" in body


def test_main_opens_a_browser_at_the_address_it_printed(db, bound, monkeypatch):
    """The address printed and the address opened must be the same one."""
    opened, fired = [], threading.Event()

    def record(address: str) -> None:
        opened.append(address)
        fired.set()

    monkeypatch.setattr(ui.webbrowser, "open", record)
    with serving(["--db", db, "--port", "0"], bound) as (base, _code):
        # Waited for rather than slept past: the timer is real, and letting the
        # test end first would fire it after the patch had been undone, which
        # would open a browser window on whoever ran the suite.
        assert fired.wait(30), "the browser was never opened"

    assert opened == [f"{base}/"], opened


def test_a_container_bind_still_prints_an_address_that_can_be_opened(
        db, bound, capsys):
    """`--host 0.0.0.0` is the documented container bind, and `http://0.0.0.0/`
    is not an address anyone can open. What is printed has to be the loopback."""
    with serving(["--db", db, "--host", "0.0.0.0", "--port", "0", "--no-browser"],
                 bound) as (_base, _code):
        printed = capsys.readouterr().out

    assert "on http://127.0.0.1:" in printed, printed
    assert "http://0.0.0.0:" not in printed


def test_ctrl_c_stops_the_server_and_says_so_rather_than_tracing_back(
        db, bound, monkeypatch, capsys):
    """Ctrl-C is how the demonstration ends. It is a clean exit, not a failure."""
    def interrupted(_self, *_args, **_kwargs):
        raise KeyboardInterrupt

    # Ctrl-C surfaces out of the blocking accept loop. There is no portable way
    # to deliver a real SIGINT to this process on Windows, so it is raised at
    # the point the operating system would raise it.
    monkeypatch.setattr(ui.ThreadingHTTPServer, "serve_forever", interrupted)
    code = ui.main(["--db", db, "--port", "0", "--no-browser"])
    captured = capsys.readouterr()

    assert code == 0
    assert "stopped" in captured.out
    assert "Traceback" not in captured.err


def test_main_refuses_to_start_on_an_index_built_by_another_model(
        tmp_path, monkeypatch, capsys):
    """Refusing to boot is the point: a mismatched index answers plausibly wrong."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    build_repo(tmp_path, embedding_model="some-other-embedding:1b").close()

    code = ui.main(["--db", str(tmp_path / "index" / "knowledge.db"),
                    "--port", "0", "--no-browser"])
    err = capsys.readouterr().err

    assert code == 1
    assert "some-other-embedding:1b" in err, err
    assert ollama.EMBED_MODEL in err, "the operator needs both names to act on this"
    assert "Traceback" not in err


def test_the_default_server_will_not_serve_staff_material_however_it_is_asked(
        db, bound):
    """`?a=staff` on a public instance was self-service promotion. It must not be."""
    with serving(["--db", db, "--port", "0", "--no-browser"], bound) as (base, _code):
        _status, body = get(base, "/ask", q="What is the internal margin on Solo",
                            a="staff")

    payload = json.loads(body)
    cited = [source["url"] for part in payload["parts"]
             for source in part["sources"]]
    text = " ".join(part["text"] for part in payload["parts"])

    assert "42 per cent" not in text, "staff evidence reached a public caller"
    assert not any(url.startswith("fixture://") for url in cited), cited


def test_allow_audience_reaches_retrieval_rather_than_being_parsed_and_lost(
        db, bound):
    """A flag that parses and does nothing is worse than no flag at all."""
    with serving(["--db", db, "--port", "0", "--no-browser",
                  "--allow-audience", "staff"], bound) as (base, _code):
        _status, body = get(base, "/ask", q="What is the internal margin on Solo")

    payload = json.loads(body)
    cited = [source["url"] for part in payload["parts"]
             for source in part["sources"]]
    assert any(url.startswith("fixture://") for url in cited), cited


# ----------------------------------------------------- conversation state


class Browser:
    """A caller that keeps its cookie, which is the whole of the session layer.

    `urllib` does not keep one, so every other test in this file starts a fresh
    conversation on each request — which is the single-turn behaviour these
    tests exist to move past, and why they need something that behaves like a
    browser rather than like `get()`.
    """

    def __init__(self, base: str) -> None:
        self.base = base
        self.cookie = ""

    def _open(self, path: str, params: dict):
        url = self.base + path + ("?" + urllib.parse.urlencode(params)
                                  if params else "")
        request = urllib.request.Request(url)
        if self.cookie:
            request.add_header("Cookie", self.cookie)
        response = urllib.request.urlopen(request, timeout=30)
        self.cookie = response.headers.get("Set-Cookie", "").split(";")[0]
        return response

    def ask(self, question: str = "", **params) -> tuple[int, str]:
        with self._open("/", {"q": question, **params} if question or params
                        else {}) as response:
            return response.status, response.read().decode("utf-8")

    def json(self, question: str, **params) -> dict:
        with self._open("/ask", {"q": question, **params}) as response:
            return json.loads(response.read().decode("utf-8"))

    @property
    def session_id(self) -> str:
        return self.cookie.split("=", 1)[1]


def _serve(tmp_path, monkeypatch, *, cache: bool, audiences=("public",)):
    """A real threaded server over two public documents, with its own session store.

    Two documents so the router reaches Compose, which is where a recommendation
    is actually composed and where a carried substrate shows up as a stated
    assumption. `cache` is a parameter rather than a default because the answer
    cache is keyed on the question text and not on the carried slots — see
    `test_the_answer_cache_does_not_serve_one_conversation_from_another` for
    what that costs and where the fix belongs.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: between(0, 1))
    monkeypatch.setattr(ollama, "generate", quoting)

    build_repo(tmp_path, second_product=True).close()
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    Handler.assistant = Assistant(repo, cache=cache)
    Handler.meta = "test"
    Handler.audiences = audiences
    Handler.sessions = SessionStore()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        repo.close()


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """A conversational server. The answer cache is off; see `_serve`."""
    yield from _serve(tmp_path, monkeypatch, cache=False)


@pytest.fixture
def cached_chat(tmp_path, monkeypatch):
    """The same server in the configuration `main()` actually ships."""
    yield from _serve(tmp_path, monkeypatch, cache=True)


@pytest.fixture
def staff_chat(tmp_path, monkeypatch):
    """A server started for staff, so that a request has something to narrow."""
    yield from _serve(tmp_path, monkeypatch, cache=False,
                      audiences=("public", "staff"))


ASK = "What plaster should I use on my wall"


def test_a_session_cookie_is_issued_and_is_not_readable_by_script(chat):
    """The cookie names the session, so nothing on the page may touch it."""
    with urllib.request.urlopen(chat + "/", timeout=30) as response:
        issued = response.headers.get("Set-Cookie", "")

    assert issued.startswith(f"{SESSION_COOKIE}="), issued
    assert "HttpOnly" in issued
    assert "SameSite=Lax" in issued
    assert "Path=/" in issued


def test_the_same_session_comes_back_rather_than_a_new_one_each_request(chat):
    browser = Browser(chat)
    browser.ask(ASK)
    first = browser.cookie

    browser.ask("How much water does Solo need")

    assert browser.cookie == first, "every turn started a new conversation"


def test_a_malformed_cookie_header_starts_a_conversation_rather_than_a_500(chat):
    """The Cookie header is attacker-controlled, and `SimpleCookie` raises on it."""
    request = urllib.request.Request(chat + "/")
    request.add_header("Cookie", "=====")

    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        assert response.headers.get("Set-Cookie", "").startswith(SESSION_COOKIE)


def test_an_ask_back_answered_in_the_next_turn_is_answered_for_that_substrate(chat):
    """Decision 10's recorded cost, closed end to end over a real server.

    The ask-back is the one place this system asks a question instead of
    answering one, and single-turn it was a dead end: the reply "brick" arrived
    as a new question about brick. Here the substrate is uncued, step 5 asks
    what the wall is built of, the person answers in one word, and the question
    they originally asked is the one that gets answered — with brick used and
    stated, rather than assumed in silence.
    """
    browser = Browser(chat)

    first = browser.json(ASK)
    assert first["parts"][0]["path"] == "ask back", first
    assert "wall built of" in first["parts"][0]["text"]

    second = browser.json("brick")

    assert second["answered"] == ASK, "the pending question was not resumed"
    part = second["parts"][0]
    assert part["path"] != "ask back", "it asked the same question twice"
    assert part["diagnostics"]["slots"]["substrate"] == "brick"
    assert second["session_slots"] == {"substrate": "brick"}


def test_a_correction_in_a_later_turn_beats_what_the_session_remembered(chat):
    """"Actually it's stone." A session that argues back is worse than no session."""
    browser = Browser(chat)
    browser.json(ASK)
    browser.json("brick")

    browser.json("actually it is stone")
    again = browser.json(ASK)

    assert again["session_slots"]["substrate"] == "stone"
    assert again["parts"][0]["diagnostics"]["slots"]["substrate"] == "stone"
    assert again["parts"][0]["path"] != "ask back"


def test_a_reply_that_names_no_substrate_is_a_new_question_not_an_answer(chat):
    """Only the fact step 5 asked for resumes the pending question.

    And a turn that changes the subject retires it: the person moved on, and a
    question resurrected three turns later by a stray "brick" would be a worse
    surprise than asking again. Nothing is lost by retiring it — the substrate
    is remembered either way, so re-asking now gets an answer rather than a
    second ask-back.
    """
    browser = Browser(chat)
    browser.json(ASK)

    aside = browser.json("How much water does Solo need")

    assert aside["answered"] == "How much water does Solo need", \
        "an unrelated question was answered as if it were the missing detail"
    assert Handler.sessions.pending(browser.session_id) == ""


def test_a_page_with_no_question_still_belongs_to_the_conversation(chat):
    """Reloading the bare page must not quietly end a pending ask-back."""
    browser = Browser(chat)
    browser.ask(ASK)
    before = browser.cookie

    status, body = browser.ask()

    assert status == 200
    assert browser.cookie == before
    assert "Earlier in this conversation" in body, body
    assert Handler.sessions.pending(browser.session_id) == ASK


def test_the_page_shows_the_conversation_so_far(chat):
    """A follow-up has to read as a follow-up rather than as an isolated answer."""
    browser = Browser(chat)
    browser.ask(ASK)

    _status, body = browser.ask("brick")

    assert "Earlier in this conversation" in body, body
    assert escaped_in(body, ASK), "the earlier question was not shown back"


def test_the_first_page_of_a_conversation_shows_no_transcript(chat):
    """An empty history must print nothing rather than an empty heading."""
    _status, body = Browser(chat).ask()

    assert "Earlier in this conversation" not in body, body


def test_the_conversation_shown_back_is_escaped_like_everything_else(chat):
    """The transcript is a second place a typed question reaches the page."""
    browser = Browser(chat)
    browser.ask("<script>alert(1)</script> what plaster")

    _status, body = browser.ask("brick")

    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body


def test_a_second_browser_does_not_inherit_the_first_browsers_wall(chat):
    """A fresh caller starts uncued, whatever anybody else has established."""
    first = Browser(chat)
    first.json(ASK)
    first.json("brick")

    fresh = Browser(chat).json(ASK)

    assert fresh["parts"][0]["path"] == "ask back", \
        "a new visitor was answered from somebody else's wall"
    assert fresh["session_slots"] == {}


def test_concurrent_conversations_never_see_each_others_slots(chat):
    """The real access pattern: one ThreadingHTTPServer, several browsers at once."""
    substrates = ["brick", "stone", "cob", "lath"] * 2

    def converse(substrate: str) -> dict:
        browser = Browser(chat)
        browser.json(ASK)
        browser.json(substrate)
        return browser.json(ASK)

    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(converse, substrates))

    for substrate, reply in zip(substrates, replies):
        assert reply["session_slots"] == {"substrate": substrate}, reply
        assert reply["parts"][0]["diagnostics"]["slots"]["substrate"] == substrate


def test_a_session_cannot_widen_the_audience_a_later_turn_is_served(staff_chat):
    """The session holds slots. Rights are resolved per request, on every request.

    A conversation that saw staff material once and then narrows itself must
    narrow for real — otherwise the session rather than the request decides what
    a caller may read, which is decision 14's cache leak wearing a cookie.
    """
    browser = Browser(staff_chat)
    staff = browser.json("What is the internal margin on Solo", a="staff")
    assert any(source["url"].startswith("fixture://")
               for part in staff["parts"] for source in part["sources"]), staff

    public = browser.json("What is the internal margin on Solo", a="public")

    text = " ".join(part["text"] for part in public["parts"])
    cited = [source["url"] for part in public["parts"]
             for source in part["sources"]]
    assert "42 per cent" not in text, "the session carried staff evidence forward"
    assert not any(url.startswith("fixture://") for url in cited), cited
    assert "audience" not in public["session_slots"]


def test_the_answer_cache_does_not_serve_one_conversation_from_another(cached_chat):
    """Found as a defect by the work that needed it, fixed, and kept as a guard.

    Two conversations, the same words, different walls. The cache key was the
    question text, the audience set, the snapshot, the generation model and the
    chunking version — and not what the caller had been taken to have said
    earlier. So the second browser was answered from the first browser's brick,
    which is exactly the silent assumption about somebody's wall that decision
    10 exists to prevent.

    It also defeated multi-turn through its own cache: resuming a pending
    question re-asks the same words, so the stored *ask-back* came back instead
    of the answer the new substrate had finally made possible. The carried
    slots are now in the key.
    """
    first = Browser(cached_chat)
    first.json(ASK)
    first.json("brick")

    second = Browser(cached_chat)
    second.json(ASK)
    second.json("stone")

    reply = second.json(ASK)
    assert reply["parts"][0]["diagnostics"]["slots"]["substrate"] == "stone", reply


# ------------------------------------------------- the refusal, as a page reads


def test_a_refusal_leads_with_a_sentence_and_folds_the_passage_away(server):
    """The reviewed complaint: a wall of datasheet where a sentence would do.

    The refusal used to open with "the closest published passage is this" and
    600 characters of raw chunk, which on a public page buries the one thing the
    visitor needed - that this is not published, and who to ask. The passage is
    still on the page, and still complete: it is inside a disclosure rather than
    in front of the summary.
    """
    status, body = get(server, "/", q="What is the pot life of Solo in hours")

    assert status == 200
    assert "I could not find published Lime Green guidance" in body, body
    assert "Show source passage" in body, "the evidence was dropped, not demoted"
    # Prose first, evidence second - the whole of the change.
    assert body.index("I could not find published") < body.index("Show source passage")
    assert "0800 538 5746" in body, "a refusal without a way forward"


def test_the_folded_passage_still_carries_the_published_words_and_its_source(server):
    """A presentation change must not quietly become a reduction."""
    status, body = get(server, "/", q="What is the pot life of Solo in hours")
    inner = body.split("Show source passage", 1)[1]

    assert status == 200
    assert "Source passage" in inner
    assert html.escape("5-6 litres of clean water per 25 kg sack",
                       quote=True) in inner, inner[:600]
    assert "<h3>Sources</h3>" in body, "the refusal cited nothing"
    assert SOLO in body


def test_the_diagnostics_view_opens_the_passage_it_would_otherwise_fold(server):
    """A reader auditing a refusal should not have to click to see the evidence."""
    _status, folded = get(server, "/", q="What is the pot life of Solo in hours")
    _status, opened = get(server, "/", q="What is the pot life of Solo in hours",
                          v="1")

    assert "<details class='passage'>" in folded
    assert "<details class='passage' open>" in opened


def test_the_json_surface_carries_provenance_structured_rather_than_as_prose(server):
    """A channel adapter should not have to parse our sentences back apart."""
    status, body = get(server, "/ask", q="How much water does Solo need on brick")
    part = json.loads(body)["parts"][0]

    assert status == 200
    assert part["facts"] == [{"slot": "substrate", "value": "brick",
                              "provenance": "stated"}]
    assert part["assumptions"] == []
    assert part["disclosure"] == ""
    # `text` on the JSON surface stays whole, disclosure included, because a
    # consumer that wants the evidence should not have to ask for it twice.
    assert "as you said" in part["text"]
