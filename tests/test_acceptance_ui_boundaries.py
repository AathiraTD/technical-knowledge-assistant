"""Rendering boundaries, not an answer-quality grader or a live-model test."""

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

from assistant import ui
from assistant.answer import Answer
from assistant.engine import Reply


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
    path = Path(__file__).resolve().parents[1] / "eval" / "evalset"
    with (path / "lime_green_ui_acceptance_tests.csv").open(encoding="utf-8-sig") as stream:
        return next(row["question"] for row in csv.DictReader(stream)
                    if row["test_id"] == test_id)


def multipart_reply():
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
    return ui.PAGE.format(initial_content=content, audience_display="public",
                          meta="test", footer_audience="public")


@pytest.mark.parametrize("path", ["compose", "refuse", "route", "extract"])
def test_server_keeps_internal_path_out_of_prose(path):
    answer = Answer(text="Safe published words [1].", path=path,
                    refused=path == "refuse", sources=[SOURCE])
    soup = BeautifulSoup(ui.render_html(Reply("q", [("q", answer)]), False, REFERENCE),
                         "html.parser")
    assert soup.select_one(".answer-text").get_text() == answer.text
    assert not soup.select(".answer-text .tag")
    assert soup.select_one(".diagnostics-list .tag").get_text() == path
    assert REFERENCE in soup.select_one(".diagnostics-list").get_text()
    assert soup.select_one(".source-link")["href"] == SOURCE["url"]
    assert soup.select_one(".answer-response")["data-state"] == "complete"


def test_server_preserves_all_t22_parts_and_refusal_evidence():
    reply = multipart_reply()
    soup = BeautifulSoup(ui.render_html(reply, False, REFERENCE), "html.parser")
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
    soup = BeautifulSoup(ui.render_html(multipart_reply(), False, reference), "html.parser")
    assert soup.select_one(".answer-response")["data-correlation-id"] == ""
    assert not soup.select("img")
    assert "reference" not in soup.select_one(".diagnostics-list").get_text()


def test_server_escapes_answer_and_perception():
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
    context = browser.new_context()
    try:
        page = context.new_page()
        page.set_content(page_html())
        yield page
    finally:
        context.close()


@pytest.mark.parametrize("path", ["compose", "refuse", "route", "extract"])
def test_client_path_is_hidden_until_diagnostics_open(page, path):
    answer = Answer("Published words [1].", path, sources=[SOURCE], refused=path == "refuse")
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
    with urlopen(boundary_server + "/?q=test", timeout=10) as response:
        reference = response.headers["X-Correlation-Id"]
        soup = BeautifulSoup(response.read(), "html.parser")
    assert re.fullmatch(r"[a-f0-9]{12}", reference)
    assert soup.select_one(".answer-response")["data-correlation-id"] == reference
    assert len(soup.select(".answer-text")) == 2


@pytest.mark.parametrize("upload", [False, True])
def test_browser_request_response_and_dom_correlate(page, boundary_server, upload):
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
