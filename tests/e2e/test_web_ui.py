"""What only a browser can prove about the web surface.

Eight behaviours, chosen because each one is invisible to every other test in
this repository. The cookie is HttpOnly, so the page's own script cannot read it
and neither can a test that only calls the library. The upload body is assembled
by the browser from a file input, so a hand-written multipart in a unit test
proves the parser and not the path. A citation is a link or it is not, and that
is a fact about rendered DOM.

None of these asserts whether an answer is *correct*. `eval/run.py` owns answer
quality, on the library, where decision 15 deliberately keeps it; a suite that
graded answers through a browser would be slower, flakier and duplicated.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from .conftest import ROOT, ROUTED_QUESTION, ask

# The smallest thing that is really a PNG: an 8-byte signature and a valid IHDR.
# Built here rather than committed, because the upload boundary sniffs content
# and a fixture file would only prove that the same bytes survive a round trip.
ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")


def session_cookie(page) -> str:
    for cookie in page.context.cookies():
        if cookie["name"] == "tka_session":
            return cookie["value"]
    return ""


def test_the_session_cookie_is_httponly_and_survives_a_follow_up(page):
    """One conversation across two turns, and a cookie no script can touch.

    The HttpOnly assertion is not decoration. It is the property that makes the
    session a server fact rather than a page fact, and every other test here
    depends on it being true.
    """
    page.goto("/")
    ask(page, ROUTED_QUESTION)
    first = session_cookie(page)
    assert first, "no session cookie was set"

    ask(page, ROUTED_QUESTION)

    assert session_cookie(page) == first, "the follow-up started a new session"
    jar = [c for c in page.context.cookies() if c["name"] == "tka_session"]
    assert jar[0]["httpOnly"] is True


def test_new_chat_actually_starts_a_new_conversation(page):
    """The button has to reach the server, because only the server can do this.

    `newChat()` clears the cookie with `document.cookie`, and a cookie set
    HttpOnly is precisely the cookie script cannot clear -- the browser ignores
    the write. So the page reloads, the old cookie goes back up with the next
    request, and the conversation the person thought they had ended carries its
    substrate, location and product into the next question.
    """
    page.goto("/")
    ask(page, ROUTED_QUESTION)
    before = session_cookie(page)

    page.click(".new-chat-btn")
    page.wait_for_load_state("networkidle")
    ask(page, ROUTED_QUESTION)

    assert session_cookie(page) != before, (
        "New chat left the old session in place; the next question will still "
        "inherit the previous conversation's slots")


def test_an_uploaded_photograph_reaches_the_server(page):
    """The browser assembles the multipart body; this proves it arrives.

    Asserted on `images_read` rather than on anything the vision model says,
    because perception needs a model and costs minutes, and the question here is
    whether the bytes crossed the boundary at all -- which is the half that was
    broken.
    """
    page.goto("/")
    page.set_input_files("#file-input", {
        "name": "wall.png", "mimeType": "image/png", "buffer": ONE_PIXEL_PNG})

    with page.expect_response(lambda r: "/ask" in r.url) as caught:
        page.fill("#question-input", "This is the wall I am working on.")
        page.click("#send-btn")
    payload = caught.value.json()

    assert payload["images_read"] == 1, (
        f"the server read {payload['images_read']} images; "
        f"upload notes were {payload.get('upload_notes')}")


def test_a_photograph_and_the_text_turn_after_it_are_one_conversation(page):
    """An upload is a POST and a plain question is a GET; both are one session."""
    page.goto("/")
    page.set_input_files("#file-input", {
        "name": "wall.png", "mimeType": "image/png", "buffer": ONE_PIXEL_PNG})
    with page.expect_response(lambda r: "/ask" in r.url):
        page.fill("#question-input", "This is the wall I am working on.")
        page.click("#send-btn")
    after_upload = session_cookie(page)

    ask(page, ROUTED_QUESTION)

    assert session_cookie(page) == after_upload, (
        "the text turn after a photograph started a new conversation")


def test_an_attachment_that_is_not_an_image_is_refused_and_said_so(page):
    """Refused by content, and the caller is told rather than quietly ignored."""
    page.goto("/")
    page.set_input_files("#file-input", {
        "name": "wall.png",                  # the name says PNG; the bytes do not
        "mimeType": "image/png",
        "buffer": b"this is not a picture, whatever it is called"})

    with page.expect_response(lambda r: "/ask" in r.url) as caught:
        page.fill("#question-input", "This is the wall I am working on.")
        page.click("#send-btn")
    payload = caught.value.json()

    assert payload["images_read"] == 0
    assert payload["upload_notes"], "the attachment was dropped without saying so"
    assert "not a recognised image" in " ".join(payload["upload_notes"])


def test_a_cited_source_renders_as_a_working_link(page):
    """A citation the reader cannot follow is not a citation.

    Asked as a document request, which the manifest answers with names, dates
    and links and without the model: the question here is whether a source
    becomes an anchor with an absolute href, not whether generation cites well.
    """
    page.goto("/")
    ask(page, "Where is the datasheet for Lime Green Ultra?")

    links = page.locator(".source-link")
    if links.count() == 0:
        pytest.skip("this question returned no sources to render")
    href = links.first.get_attribute("href")
    assert href and href.startswith("http"), f"source link href was {href!r}"


def test_a_public_caller_cannot_ask_its_way_into_staff_material(page):
    """The audience set narrows from what the server allows and never widens.

    Enforced in `resolve`, but asserted through the URL because that is where a
    caller would actually try it, and because a regression here is a leak rather
    than a bug.
    """
    page.goto("/?q=What+is+the+coverage%3F&audience=staff")

    footer = page.inner_text("#footer-audience").lower()
    assert "staff" not in footer
    assert "public" in footer


def test_the_correlation_id_the_browser_gets_reads_a_real_trace(page):
    """The whole observability chain, end to end and across process boundaries.

    The browser is handed an id in a header; the operator pastes that id into
    `python -m assistant.trace` and gets the stages of that answer back. Until
    this held, a reported problem could only be investigated by reproducing it.
    """
    page.goto("/")
    with page.expect_response(lambda r: "/ask" in r.url) as caught:
        page.fill("#question-input", ROUTED_QUESTION)
        page.click("#send-btn")
    response = caught.value

    correlation = response.headers.get("x-correlation-id", "")
    assert correlation, "no X-Correlation-Id header came back"
    assert response.json()["correlation_id"] == correlation

    read = subprocess.run(
        [sys.executable, "-m", "assistant.trace", correlation, "--json"],
        cwd=ROOT, capture_output=True, text=True)
    assert read.returncode == 0, f"the trace could not be read: {read.stderr[-300:]}"
    spans = json.loads(read.stdout)
    assert spans, "the id came back from the browser but named no spans"
    assert {s["name"] for s in spans} & {"answer", "part"}
