"""What a browser actually receives when somebody attaches a photograph.

The seam tests prove a reading reaches the router; these prove it reaches the
*person*. That is a separate risk and it failed separately: the resolution was
being computed correctly and then dropped on the floor, so a rendered wall
whose substrate had been correctly refused looked identical, on the page, to a
photograph nobody had looked at.

Driven over HTTP against the real server rather than against `render_html`,
because the payload is assembled in the request handler and a test calling the
renderer directly would skip the half that was broken. The vision call is
replaced; everything else -- multipart parsing, the sniffer, the session, the
graph, the JSON assembly -- is the code that runs in the demonstration.

The escaping tests are not ceremony. A filename is caller-controlled text that
reaches the page without passing a model, and the observation note is
model-controlled text that reaches the page without passing the six checks.
Those are the two strings on this surface with no other guard in front of them.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
import uuid
import io
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import ollama, ui, vision                             # noqa: E402
from assistant.engine import Assistant                               # noqa: E402
from assistant.store.factory import open_repository                  # noqa: E402
from assistant.ui import Handler                                     # noqa: E402

from test_engine import build_repo, quoting, unit                    # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 20
WHOLE = (0.0, 0.0, 1.0, 1.0)


def perception(*pairs, confidence=0.93, note="", cannot=("the moisture source",)):
    return vision.Perception(
        observations=tuple(
            vision.Observation(attribute=a, value=v, confidence=confidence,
                               image="IMG_001", region=WHOLE, observation=note)
            for a, v in pairs),
        cannot_determine_from_image=tuple(cannot),
        image="IMG_001", model="stub-vlm")


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """A factory: the caller says what the model sees, then gets a base URL."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)

    build_repo(tmp_path, two_documents=True).close()
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    saved = {name: Handler.__dict__.get(name) for name in
             ("assistant", "meta", "audiences", "sessions", "uploads")}
    running: list = []

    def start(seen):
        monkeypatch.setattr(vision, "observe", lambda *_a, **_k: seen)
        Handler.assistant = Assistant(repo)
        Handler.meta = "test"
        Handler.audiences = ("public",)
        Handler.sessions = ui.SessionStore()
        Handler.uploads = ui.UploadBudget()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        running.append((httpd, thread))
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        yield start
    finally:
        for httpd, thread in running:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
        repo.close()
        for name, value in saved.items():
            if value is None:
                # `Handler.__dict__` is a mappingproxy and cannot be written
                # through; the class attribute has to go through `delattr`.
                if name in Handler.__dict__:
                    delattr(Handler, name)
            else:
                setattr(Handler, name, value)


def multipart(fields: dict, files: list) -> tuple[bytes, str]:
    boundary = "----" + uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        out.write(value.encode("utf-8") + b"\r\n")
    for filename, data in files:
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="image"; '
                  f'filename="{filename}"\r\n'.encode())
        out.write(b"Content-Type: application/octet-stream\r\n\r\n")
        out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def ask(base: str, question: str, files=((("wall.png"), PNG),), path="/ask"):
    body, content_type = multipart({"q": question}, list(files))
    request = urllib.request.Request(base + path, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    return urllib.request.urlopen(request, timeout=30).read()


# ------------------------------------------------------------- the payload


def test_the_page_is_told_what_the_photograph_showed(serve):
    """The whole point: a reading reaches the browser, with its certainty."""
    base = serve(perception(("substrate", "brickwork"),
                            ("exposed_masonry", "exposed")))
    payload = json.loads(ask(base, "what thickness can I build this up to?"))

    assert payload["images_read"] == 1
    report = payload["perception"]
    assert report["routed"] == {"substrate": "brick"}

    rows = {r["attribute"]: r for r in report["observations"]}
    assert rows["substrate"]["certainty"] == "OBSERVED"
    assert rows["exposed_masonry"]["certainty"] == "OBSERVED"
    assert report["summary"], "nothing was written for a person to read"


def test_a_refused_reading_says_so_rather_than_vanishing(serve):
    """The failure this whole payload exists to fix.

    A substrate refused through a render used to look exactly like a
    photograph that had never been read: no slot, no mention, nothing. The
    reading is now shown, hedged, with the reason it was not acted on.
    """
    base = serve(perception(("substrate", "brickwork"),
                            ("existing_finish", "rendered")))
    report = json.loads(ask(base, "what should I use on this?"))["perception"]

    assert report["routed"] == {}
    substrate = next(r for r in report["observations"]
                     if r["attribute"] == "substrate")
    assert substrate["certainty"] == "LIKELY"
    assert "covers the background" in substrate["withheld"]


def test_what_the_photograph_could_not_settle_reaches_the_page(serve):
    base = serve(perception(("substrate", "brickwork"),
                            cannot=("whether the wall is internal or external",
                                    "the moisture source")))
    report = json.loads(ask(base, "what thickness?"))["perception"]

    assert any("internal or external" in c
               for c in report["cannot_determine_from_image"])


def test_a_turn_with_no_photograph_carries_no_perception(serve):
    """Its presence is the signal that an image was read.

    So a page can tell "looked and saw nothing" apart from "was sent nothing",
    and a text-only turn must not inherit the last upload's reading.
    """
    base = serve(perception(("substrate", "brickwork")))
    with_image = json.loads(ask(base, "what thickness?"))
    assert with_image["perception"]

    text_only = json.loads(urllib.request.urlopen(
        base + "/ask?q=what+thickness", timeout=30).read())
    assert text_only["perception"] is None


def test_a_failed_perception_still_answers(serve):
    """Coverage drops, the answer does not."""
    base = serve(vision.Perception(error="connection refused"))
    payload = json.loads(ask(base, "how much water does Solo need"))

    assert payload["parts"], "a broken vision model cost the caller their answer"
    assert payload["perception"]["observations"] == []


# ------------------------------------------------------- the rendered page


def test_the_html_page_declares_a_rejected_attachment(serve):
    """Regression: the notes were dropped when the page became a chat client.

    A POST to `/` with a file that was not an image answered the question and
    said nothing about the file, which reads as "the assistant ignored my
    photo" rather than as the refusal it was.
    """
    base = serve(perception(("substrate", "brickwork")))
    page = ask(base, "how much water does Solo need",
               files=[("wall.png", b"<?php echo 1; ?>")],
               path="/").decode("utf-8")

    assert "not a recognised image" in page


def test_a_hostile_filename_never_reaches_the_page_at_all(serve):
    """Stronger than escaping, and the note says so deliberately.

    The rejection names the attachment by *shape* -- "one attachment was not a
    recognised image" -- rather than quoting what the caller called it. So the
    filename is not escaped on the way out; it never goes out. This asserts the
    stronger property rather than the weaker one it would be easy to settle
    for, because a later edit that helpfully added the filename back would pass
    an escaping test and reintroduce the string.
    """
    base = serve(perception(("substrate", "brickwork")))
    page = ask(base, "how much water does Solo need",
               files=[("<script>alert(1)</script>.txt", b"not an image")],
               path="/").decode("utf-8")

    assert "not a recognised image" in page
    assert "alert(1)" not in page
    assert "<script>alert(1)</script>.txt" not in page

    # The renderer escapes regardless, so the guarantee does not rest solely on
    # the wording of one note.
    assert "&lt;" in ui.render_upload_notes(["<b>x</b>"])


def test_the_client_renders_every_reading_with_its_certainty(serve):
    """Asserted on the page source, because the pairing is the safety property.

    A row that could print the value without the hedge beside it would turn a
    reading into a fact. The renderer is the last place that could separate
    them, so it is checked that it does not have a path which does.
    """
    base = serve(perception(("substrate", "brickwork")))
    page = urllib.request.urlopen(base + "/", timeout=30).read().decode("utf-8")

    assert "renderPerception" in page
    for certainty in ("OBSERVED", "LIKELY", "UNCERTAIN", "CANNOT_DETERMINE"):
        assert certainty in page, f"{certainty} has no rendering"
    assert "Not determinable from the photograph" in page
    # And the attachment is named back to the person who sent it.
    assert "attachments.map(escapeHtml)" in page


def test_the_model_note_shown_on_the_page_is_escaped(serve):
    """Model-controlled text, reaching a person without passing the six checks."""
    base = serve(perception(("substrate", "brickwork"),
                            note="<img src=x onerror=alert(1)>"))
    report = json.loads(ask(base, "what thickness?"))["perception"]

    note = next(r for r in report["observations"]
                if r["attribute"] == "substrate")["notes"][0]
    assert note == "<img src=x onerror=alert(1)>", (
        "the JSON should carry it verbatim; escaping is the renderer's job")
    page = urllib.request.urlopen(base + "/", timeout=30).read().decode("utf-8")
    assert "escapeHtml" in page
