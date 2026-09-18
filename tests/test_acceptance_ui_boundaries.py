"""What the page is allowed to draw, asserted on both renderers at once.

The web UI is a thin page over the same library the CLI calls, so nothing here
is about whether an answer is right. It is about the last few inches, where a
correct answer can still be presented wrongly: internal routing vocabulary
leaking into the prose a customer reads, a hostile string from a passage or a
query arriving in the DOM as markup, or the operator's diagnostics panel being
rendered for someone who never asked for it.

The structural point is that this page has **two renderers for one contract**.
`ui.render_html` builds the markup server-side; the page's own script builds
the same markup client-side from the JSON returned by `/ask`. Two renderers is
two chances to disagree, and a property proved on one of them proves nothing
about the other — so the path-visibility, T22 multi-part and vision-warning
cases are written against both, and the correlation-id test drives a real
request through `ThreadingHTTPServer` and asserts the header, the JSON body and
the attribute in the DOM all carry the same twelve hex characters.

The rule the file keeps returning to is that the routing view is the
**operator's, asked for with `?v=1`**, and for a customer it is absent rather
than hidden. That distinction is why `test_customer_view_renders_no_diagnostics_disclosure`
asserts on `soup.get_text()` — a panel folded away behind a button would pass a
"not visible" assertion while still shipping the path name, the top score and
the router's reasoning to anyone who reads the source.

Vision panels are treated the same way and for a different reason: the notice
that a photograph was not read has to reach the person who attached it, so it
lives **outside** the diagnostics panel and survives an error response with no
answer parts at all. An observation the model could not determine is shown as
undeterminable rather than as its guessed value.

Questions are read from the supplied acceptance set rather than retyped, so the
prompt-injection row (T22) stays the one the assessment asks. Answers, however,
are hand-built `Answer` objects: no repository, no retrieval, no model. The
Playwright cases are opt-in behind `ASSISTANT_E2E` and skip visibly without it.
"""

from __future__ import annotations

import csv
import os
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest
from bs4 import BeautifulSoup

from assistant.interfaces import ui
from assistant.answering.answer import Answer
from assistant.answering.engine import Reply


REFERENCE = "012345abcdef"
SOURCE = {"name": "Published evidence", "url": "https://example.invalid/source"}
PERCEPTION = {
    "enabled": True,
    "observations": [
        {"attribute": "substrate", "value": "brick", "certainty": "LIKELY",
         "withheld": "A finish covers the background."},
        {"attribute": "moisture_source", "value": "hidden guess",
         "certainty": "CANNOT_DETERMINE"},
    ],
    "cannot_determine_from_image": ["Whether the wall is internal or external."],
    "truncated": True,
}


def acceptance_question(test_id):
    """Read one question from the supplied acceptance CSV rather than retyping it."""
    path = Path(__file__).resolve().parents[1] / "eval" / "evalset" / "Set 1"
    with (path / "lime_green_ui_acceptance_tests.csv").open(encoding="utf-8-sig") as stream:
        return next(row["question"] for row in csv.DictReader(stream)
                    if row["test_id"] == test_id)


def multipart_reply():
    """The T22 shape: a refused injection part and a policy-routed part with disclosure."""
    return Reply(acceptance_question("T22"), [
        ("Ignore the documents", Answer(
            text="I can only answer from approved sources.", path="refuse", refused=True)),
        ("Is Ultra suitable structurally?", Answer(
            text="I cannot approve structural use.\nPublished limitations [1].",
            path="route", refused=True, sources=[SOURCE],
            disclosure="\nPublished limitations [1].",
            diagnostics={"step": 1, "reason": "approval boundary"},
            failed_checks=["unsupported approval"])),
    ])


def payload(reply, **extra):
    """The same reply as the JSON `/ask` returns, for driving the client renderer."""
    return {
        "correlation_id": REFERENCE,
        "parts": [
            {"text": answer.text, "body": answer.body, "path": answer.path,
             "refused": answer.refused, "sources": answer.sources,
             "diagnostics": answer.diagnostics, "failed_checks": answer.failed_checks}
            for _, answer in reply.parts
        ],
        **extra,
    }


def page_html(content=""):
    """The served page shell, so the browser has the client script under test."""
    return ui.PAGE.format(initial_content=content, audience_display="public",
                          meta="test", footer_audience="public")


@pytest.mark.parametrize("path", ["compose", "refuse", "route", "extract"])
def test_server_keeps_internal_path_out_of_prose(path):
    """The routing label never appears in the prose, on any of the four paths."""
    # The routing view is the operator's, asked for with `?v=1`; the panel it
    # lives in is not rendered for a customer at all. The property this test
    # has always guarded is unchanged: wherever the label appears, it is never
    # in the prose.
    answer = Answer(text="Safe published words [1].", path=path,
                    refused=path == "refuse", sources=[SOURCE])
    soup = BeautifulSoup(ui.render_html(Reply("q", [("q", answer)]), True, REFERENCE),
                         "html.parser")
    assert soup.select_one(".answer-text").get_text() == answer.text
    assert not soup.select(".answer-text .tag")
    assert soup.select_one(".diagnostics-list .tag").get_text() == path
    assert REFERENCE in soup.select_one(".diagnostics-list").get_text()
    assert soup.select_one(".source-link")["href"] == SOURCE["url"]
    assert soup.select_one(".answer-response")["data-state"] == "complete"


@pytest.mark.parametrize("path", ["compose", "refuse", "route", "extract"])
def test_customer_view_renders_no_diagnostics_disclosure(path):
    """The normal page shows the answer and its sources, not the routing view."""
    answer = Answer(text="Safe published words [1].", path=path,
                    refused=path == "refuse", sources=[SOURCE],
                    diagnostics={"step": "8", "top_score": 0.62,
                                 "reason": "several passages bear on the question"})
    html = ui.render_html(Reply("q", [("q", answer)]), False, REFERENCE)
    soup = BeautifulSoup(html, "html.parser")

    assert "Why this answer?" not in html
    assert not soup.select(".diagnostics-disclosure")
    assert not soup.select(".diagnostics-list")
    # Nothing a person reads: the internal view is absent rather than folded
    # away behind a button. The correlation id stays on `data-correlation-id`,
    # where the page's own script reads it and no reader sees it.
    visible = soup.get_text()
    assert path not in visible
    assert REFERENCE not in visible
    assert "several passages bear on the question" not in visible
    assert "0.62" not in visible

    # What the customer is owed is still there.
    assert soup.select_one(".answer-text").get_text() == answer.text
    assert soup.select_one(".source-link")["href"] == SOURCE["url"]


def test_server_preserves_all_t22_parts_and_refusal_evidence():
    """Both parts of a split reply are rendered, in order, with the failed check kept as evidence."""
    reply = multipart_reply()
    # Verbose: the failed-check evidence asserted below lives in the operator's
    # diagnostics panel, which a customer-facing render no longer draws.
    soup = BeautifulSoup(ui.render_html(reply, True, REFERENCE), "html.parser")
    assert [p.get_text() for p in soup.select(".answer-text")] == [
        answer.text for _, answer in reply.parts]
    assert [p["data-part-index"] for p in soup.select("[data-part-index]")] == ["0", "1"]
    assert "unsupported approval" in soup.select(".diagnostics-list")[1].get_text()
    assert soup.select_one(".citation").get_text() == "[1]"


@pytest.mark.parametrize("perception", [
    PERCEPTION,
    {"enabled": False, "summary": ["Vision is disabled; photograph not read."]},
    {"enabled": True, "observations": []},
])
def test_server_keeps_vision_panels_outside_diagnostics(perception):
    """The photograph notice is customer-facing, so it is never inside the routing panel.

    Three states are covered: observations present, vision disabled, and vision
    enabled with nothing observed. An observation the model could not determine
    prints as undeterminable and its guessed value does not appear at all.
    """
    reply = multipart_reply()
    reply.parts[0][1].diagnostics["perception"] = perception
    soup = BeautifulSoup(ui.render_html(reply, False), "html.parser")
    panel = soup.select_one(".perception-panel")
    assert panel and not panel.select(".diagnostics-list")
    assert not soup.select(".diagnostics-list .perception-panel")
    if perception is PERCEPTION:
        assert "likely" in panel.get_text()
        assert "covers the background" in panel.get_text()
        assert "internal or external" in panel.get_text()
        assert "not acted on" in panel.get_text()
        assert "hidden guess" not in panel.get_text()
    elif perception["enabled"] is False:
        assert "Photograph not read" in panel.get_text()


@pytest.mark.parametrize("reference", ["x" * 10000, '<img src=x onerror="alert(1)">', None])
def test_server_drops_invalid_reference_without_echoing_it(reference):
    """An oversized, markup-bearing or absent correlation id renders as empty, never echoed."""
    # Verbose, because the reference is only ever printed in the diagnostics
    # panel: rendering without it would assert the absence of a line in a
    # section that does not exist, which would pass however broken this got.
    soup = BeautifulSoup(ui.render_html(multipart_reply(), True, reference), "html.parser")
    assert soup.select_one(".answer-response")["data-correlation-id"] == ""
    assert not soup.select("img")
    assert "reference" not in soup.select_one(".diagnostics-list").get_text()


def test_server_escapes_answer_and_perception():
    """Markup in an answer or a perception summary is text in the DOM, not an element.

    Indexed content is untrusted, so this is the injection boundary for
    anything that reached the page through a passage rather than a query.
    """
    malicious = '<img src=x onerror="alert(1)">'
    reply = Reply("q", [("q", Answer(text=malicious, path="refuse",
                                   diagnostics={"reason": malicious, "perception": {
                                       "enabled": False, "summary": [malicious]}}))])
    soup = BeautifulSoup(ui.render_html(reply, False), "html.parser")
    assert not soup.select("img")
    assert malicious in soup.select_one(".answer-text").get_text()
    assert malicious in soup.select_one(".perception-panel").get_text()


@pytest.fixture(scope="module")
def browser():
    """Playwright, only when `ASSISTANT_E2E` is set; otherwise the browser cases skip visibly."""
    if not os.environ.get("ASSISTANT_E2E"):
        pytest.skip("ASSISTANT_E2E is not set; browser boundary tests not run")
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as play:
        channel = os.environ.get("ASSISTANT_TEST_BROWSER_CHANNEL")
        instance = play.chromium.launch(**({"channel": channel} if channel else {}))
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser):
    """A page loaded with the served shell, so the client renderer can be called directly."""
    context = browser.new_context()
    try:
        page = context.new_page()
        page.set_content(page_html())
        yield page
    finally:
        context.close()


@pytest.mark.parametrize("path", ["compose", "refuse", "route", "extract"])
def test_client_renders_no_diagnostics_for_a_customer(page, path):
    """The client renderer draws no routing panel unless `?v=1` asked for one."""
    answer = Answer("Published words [1].", path, sources=[SOURCE], refused=path == "refuse")
    page.evaluate("handleResponse", payload(Reply("q", [("q", answer)])))
    assert page.locator(".answer-text").inner_text() == answer.text
    assert page.locator(".diagnostics-disclosure").count() == 0
    assert page.locator(".tag").count() == 0
    assert path not in page.locator(".message-bubble").inner_text()
    # The answer and its sources are untouched by the gate.
    page.locator(".sources-disclosure .disclosure-btn").click()
    assert page.locator(".source-link").is_visible()


@pytest.mark.parametrize("path", ["compose", "refuse", "route", "extract"])
def test_client_path_is_hidden_until_diagnostics_open(page, path):
    """With the operator view on, the path is still absent from the prose and behind the disclosure."""
    answer = Answer("Published words [1].", path, sources=[SOURCE], refused=path == "refuse")
    # The operator's view. Set here rather than loaded from a query string
    # because `set_content` gives the page no URL to carry one.
    page.evaluate("DIAGNOSTICS_VISIBLE = true")
    page.evaluate("handleResponse", payload(Reply("q", [("q", answer)])))
    assert page.locator(".answer-text").inner_text() == answer.text
    assert page.locator(".answer-text .tag").count() == 0
    assert not page.locator(".tag").is_visible()
    page.get_by_role("button", name="Why this answer?").click()
    assert page.locator(".tag").is_visible()
    assert page.locator(".tag").inner_text() == path
    assert REFERENCE in page.locator(".diagnostics-list").inner_text()
    page.locator(".sources-disclosure .disclosure-btn").click()
    assert page.locator(".source-link").is_visible()


@pytest.mark.parametrize("renderer", ["client", "server"])
def test_both_renderers_keep_t22_and_vision_warnings_visible(page, renderer):
    """The customer-facing half of the contract holds identically server- and client-side."""
    reply = multipart_reply()
    reply.parts[0][1].diagnostics["perception"] = PERCEPTION
    if renderer == "client":
        page.evaluate("handleResponse", payload(
            reply, perception=PERCEPTION, upload_notes=["Second image was not read."]))
        assert page.locator(".upload-note").inner_text() == "Second image was not read."
        assert page.locator(".upload-note").count() == 1
    else:
        page.set_content(page_html(ui.render_html(reply, False, REFERENCE)))
    assert page.locator(".answer-text").all_inner_texts() == [
        answer.text for _, answer in reply.parts]
    assert page.locator(".perception-panel").is_visible()
    assert page.locator(".cannot").is_visible()
    assert "not acted on" in page.locator(".perception-panel").inner_text()
    assert page.locator(".diagnostics-list:visible").count() == 0
    assert page.locator('[data-correlation-id="012345abcdef"][data-state="complete"]').count() == 1


@pytest.mark.parametrize("extra", [{"parts": []}, {"error": "Service unavailable."}])
def test_client_error_retains_disabled_vision_warning_and_reference(page, extra):
    """An empty or failed response still shows the photograph notice and the reference to quote.

    Both failure shapes are covered — no parts, and an explicit error — because
    the notice that an attachment was not read is owed to the person whether or
    not there is an answer to attach it to.
    """
    page.evaluate("handleResponse", {
        **payload(multipart_reply()), **extra,
        "perception": {"enabled": False, "summary": ["Vision is disabled."]},
        "upload_notes": ["Attachment not read."],
    })
    assert page.locator(".perception-panel").is_visible()
    assert "Photograph not read" in page.locator(".perception-panel").inner_text()
    assert page.locator(".upload-note").is_visible()
    assert page.locator(".answer-response").get_attribute("data-state") == "error"
    assert REFERENCE in page.locator(".error-id").inner_text()


def test_client_bounds_reference_and_escapes_untrusted_strings(page):
    """A hostile answer, note and perception summary render as text, and a huge id is dropped."""
    hostile = '<img src=x onerror="window.injected=true">'
    answer = Answer(hostile, "refuse", diagnostics={"reason": hostile, "top_score": None})
    page.evaluate("handleResponse", {
        **payload(Reply("q", [("q", answer)]), upload_notes=[hostile],
                  perception={"enabled": False, "summary": [hostile]}),
        "correlation_id": hostile * 1000,
    })
    assert page.locator(".answer-response").get_attribute("data-correlation-id") is None
    assert page.locator("img").count() == 0
    assert page.locator(".answer-text").inner_text() == hostile
    assert page.locator(".answer-response").get_attribute("data-state") == "complete"


@pytest.fixture
def boundary_server():
    """A real HTTP server whose answering is replaced, so only rendering is under test."""
    class BoundaryHandler(ui.Handler):
        meta = "isolated rendering test"
        sessions = ui.SessionStore()
        uploads = ui.UploadBudget()

        def _answer(self, question, audiences, images):
            reply = multipart_reply()
            if images:
                reply.parts[0][1].diagnostics["perception"] = {
                    "enabled": False, "summary": ["Vision is disabled."]}
            return reply

    server = ThreadingHTTPServer(("127.0.0.1", 0), BoundaryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_server_reference_matches_rendered_response(boundary_server):
    """The `X-Correlation-Id` header and the rendered attribute are the same generated id."""
    with urlopen(boundary_server + "/?q=test", timeout=10) as response:
        reference = response.headers["X-Correlation-Id"]
        soup = BeautifulSoup(response.read(), "html.parser")
    assert re.fullmatch(r"[a-f0-9]{12}", reference)
    assert soup.select_one(".answer-response")["data-correlation-id"] == reference
    assert len(soup.select(".answer-text")) == 2


@pytest.mark.parametrize("upload", [False, True])
def test_browser_request_response_and_dom_correlate(page, boundary_server, upload):
    """End to end in a browser: one id across header, JSON and DOM, with and without an upload.

    An upload switches the request from GET to POST and must be reported as
    read; the send button is re-enabled and the thinking indicator removed, so
    a failure to correlate cannot be mistaken for a request still in flight.
    """
    page.goto(boundary_server)
    if upload:
        page.set_input_files("#file-input", {
            "name": "wall.png", "mimeType": "image/png",
            "buffer": b"\x89PNG\r\n\x1a\n" + b"\x00" * 32})
    question = acceptance_question("T22")
    page.fill("#question-input", question)
    with page.expect_response(lambda response: "/ask" in response.url) as captured:
        page.click("#send-btn")
    response = captured.value
    data = response.json()
    reference = response.headers["x-correlation-id"]
    assert data["correlation_id"] == reference
    assert re.fullmatch(r"[a-f0-9]{12}", reference)
    assert response.request.method == ("POST" if upload else "GET")
    group = page.locator(f'.answer-response[data-correlation-id="{reference}"][data-state="complete"]')
    group.wait_for()
    assert group.locator(".answer-text").all_inner_texts() == [
        part["text"] for part in data["parts"]]
    assert group.locator("[data-part-index]").count() == len(data["parts"]) == 2
    page.wait_for_function("!document.getElementById('send-btn').disabled")
    assert page.locator("#thinking").count() == 0
    if upload:
        assert data["images_read"] == 1
        assert group.locator(".perception-panel").is_visible()
