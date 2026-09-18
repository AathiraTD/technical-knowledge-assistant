"""The join between a photograph and the router, and the provenance it needs.

`assistant/vision.py` has its own suite and is not re-tested here; what is
tested is everything that happens *at the seam*, which is where the risk
actually lives. A perception module that is correct in isolation still produces
a system that lies if the value it resolves is printed as something the caller
said, persisted as though they had said it, or served out of a cache to
somebody who never sent an image.

Five properties, and each one is a way the seam could be wrong rather than a
way it could be incomplete:

1. **Precedence.** Question over photograph over session. A caller who uploads
   a picture of a stone wall and types "it's brick" is correcting the image, and
   a photograph attached now is newer than a slot carried from two turns ago.
2. **Provenance prints, and prints in the right group.** An observed value gets
   its own sentence and never appears under the heading reading "Assumed",
   because a reading from an image is neither something stated nor a guess.
3. **Nothing leaks into the session.** Vision fills slots for the turn its image
   belongs to and no further. This is review §3.4 and it is the property most
   easily lost by accident, because the router's merged slot view looks exactly
   like something worth remembering.
4. **Filling slots does not license diagnosis.** A cause question with a
   photograph attached still hands off to a person. Perception is not judgement.
5. **Failure reduces coverage, not safety.** An unreachable model leaves the
   router exactly where no photograph would have.

Plus the upload boundary, which is a different kind of risk and is tested
against a real server rather than against the library, because a guard that
only holds when something else parses the request first is not a guard.

Ollama is never reached. The vision call is replaced by a function returning a
`Perception` the real resolver then has to accept or reject on its own rules, so
the vocabulary check, the band and the region are all genuinely exercised — a
stub returning a finished slot dict would prove nothing about what may cross.
"""

from __future__ import annotations

import io
import socket
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant.answering import vision
from assistant.infrastructure import ollama
from assistant.interfaces import ui
from assistant.answering.answer import (  # noqa: E402
    Provenance,
    SlotFact,
)
from assistant.cache import AnswerCache                             # noqa: E402
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.answering.router import (  # noqa: E402
    Path_,
)
from assistant.store.factory import open_repository                 # noqa: E402
from assistant.interfaces.ui import (  # noqa: E402
    Handler,
    MAX_IMAGES_PER_SESSION,
    sniff,
)

from test_engine import build_repo, quoting, unit                   # noqa: E402

# A one-pixel PNG is the smallest thing `sniff` will accept, and every test that
# needs "an image" needs only that: nothing in this system decodes a pixel, so a
# real photograph would test the same code path more slowly.
PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 20)
NOT_AN_IMAGE = b"<?php echo 1; ?>"

WHOLE_IMAGE = (0.0, 0.0, 1.0, 1.0)


def perception(slot: str, value: str, confidence: float = 0.95,
               region=WHOLE_IMAGE) -> vision.Perception:
    """One confident, auditable observation, as the model would have returned it."""
    return vision.Perception(
        observations=(vision.Observation(attribute=slot, value=value,
                                         confidence=confidence, image="IMG_001",
                                         region=region),),
        cannot_determine_from_image=("the moisture source",),
        image="IMG_001", model="test-vision")


def sees(monkeypatch, *perceptions):
    """Replace the vision call, leaving the resolver and its rules in place."""
    queue = list(perceptions)

    def fake_observe(_image, image_id: str = "", **_kwargs):
        return queue.pop(0) if queue else vision.Perception(error="nothing left")

    monkeypatch.setattr(vision, "observe", fake_observe)


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    """The assembled assistant over a two-document store, with no model at all.

    Two documents rather than one so the router can reach Compose, and a
    quoting stand-in for generation so an answer actually prints: the seam is
    only observable on a path that renders facts.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo)
    finally:
        repo.close()


def facts_of(reply) -> dict:
    """Every slot fact across the reply, by slot. One part per question here."""
    return {fact.slot: fact
            for _part, answer in reply.parts for fact in answer.facts}


# ------------------------------------------------------------- 1 precedence


def test_the_question_beats_the_photograph(assistant, monkeypatch):
    """Typing "it's brick" over a picture of stone is a correction, not a conflict.

    The person is looking at the wall and the model is looking at an image of
    it, so the sentence wins — and it must win on provenance as well as on
    value, because reporting a stated substrate as something read off a
    photograph is the same misattribution in the other direction.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    reply = assistant.ask("how much water does Solo need on a brick wall",
                          images=[PNG])
    substrate = facts_of(reply)["substrate"]
    assert substrate.value == "brick"
    assert substrate.provenance is Provenance.STATED


def test_the_photograph_beats_an_earlier_turn(assistant, monkeypatch):
    """A picture attached now is newer evidence than a slot from two turns ago."""
    sees(monkeypatch, perception("substrate", "stone"))
    reply = assistant.ask("how much water does Solo need",
                          carried={"substrate": "cob"}, images=[PNG])
    substrate = facts_of(reply)["substrate"]
    assert substrate.value == "stone"
    assert substrate.provenance is Provenance.OBSERVED


def test_an_earlier_turn_still_wins_when_nothing_was_photographed(assistant):
    """The existing behaviour, unchanged: no images means no origins at all."""
    reply = assistant.ask("how much water does Solo need",
                          carried={"substrate": "cob"})
    substrate = facts_of(reply)["substrate"]
    assert substrate.value == "cob"
    assert substrate.provenance is Provenance.CARRIED


def test_only_confirmed_attributes_cross_the_seam(assistant, monkeypatch):
    """Below the top band, and without a region, a reading fills nothing.

    Two refusals in one, because both are the resolver's and the seam must not
    route around either: an uncertain confidence is not evidence, and a claim
    with no region is not auditable. Either way the slot stays uncued, which is
    where the system already is when no photograph was sent.
    """
    sees(monkeypatch,
         perception("substrate", "stone", confidence=0.6),
         perception("substrate", "stone", region=None))
    reply = assistant.ask("how much water does Solo need", images=[PNG, PNG])
    assert "substrate" not in facts_of(reply)


def test_a_product_name_cannot_arrive_as_a_substrate(assistant, monkeypatch):
    """The vision model naming a product fills no slot, at full confidence.

    This is `vision.resolve`'s guarantee rather than the seam's, and it is
    asserted here anyway: the seam is the only place it could be lost, by
    reaching into `attributes` for something the resolver deliberately refused.
    """
    sees(monkeypatch, perception("substrate", "Lime Green Solo"))
    reply = assistant.ask("how much water does Solo need", images=[PNG])
    assert "substrate" not in facts_of(reply)


# ---------------------------------------------------- 2 how provenance prints


def test_observed_renders_its_own_sentence():
    """The phrase names the photograph, and names it once."""
    fact = SlotFact("substrate", "brick", Provenance.OBSERVED)
    assert fact.sentence == "brick (substrate), from the photograph you sent"
    assert fact.stated is False


def test_an_observed_value_is_printed_but_never_as_an_assumption(
        assistant, monkeypatch):
    """Its own line in the answer, and nothing under the "Assumed" heading.

    The heading is the point. Every surface prints `assumptions` under a word
    saying the system guessed, and a reading from an image is not a guess — but
    neither is it something the caller said, so it cannot go in the "Answered
    for … as you said" sentence either. Three groups, and this asserts that the
    value landed in the third rather than in either of the other two.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    reply = assistant.ask("how much water does Solo need", images=[PNG])
    _part, answer = reply.parts[0]

    assert "stone (substrate), from the photograph you sent" in answer.text
    assert answer.observed == ["stone (substrate), from the photograph you sent"]
    assert answer.assumptions == []
    assert "as you said" not in answer.text
    assert "as you told me earlier" not in answer.text


def test_a_genuine_assumption_is_still_an_assumption(assistant, monkeypatch):
    """Regrouping must not have emptied the group it was grouped out of.

    A per-option answer is the one thing this system really does assume, and it
    still has to arrive under the heading that says so — otherwise the fix for
    mislabelling observations would have mislabelled assumptions instead.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    reply = assistant.ask("which plaster should I use on this wall", images=[PNG])
    _part, answer = reply.parts[0]
    if answer.diagnostics.get("slots", {}).get("location"):
        pytest.skip("the question cued a location, so nothing was assumed")
    assumed = [fact for fact in answer.facts
               if fact.provenance is Provenance.ASSUMED]
    for fact in assumed:
        assert fact.sentence in answer.assumptions


# --------------------------------------------------- 3 nothing reaches memory


def test_a_vision_slot_is_not_written_back_into_the_session(
        assistant, monkeypatch):
    """Review §3.4: perception lives for the turn its image belongs to.

    Asserted against the page's own fold-back, because that is where the leak
    would happen: `diagnostics["slots"]` is the router's merged view and holds
    the observed value indistinguishably from a stated one. Persisting it would
    make a model's reading of an image outlive the image, and be reported on
    every later turn as something the caller told us.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    store = ui.SessionStore()
    session_id = store.open()

    handler = Handler.__new__(Handler)
    handler.sessions = store
    handler.session_id = session_id
    reply = assistant.ask("how much water does Solo need", images=[PNG])
    handler._remember("how much water does Solo need", reply)

    # Named, not compared whole. This asserted `== {}` while `substrate` was the
    # only slot a turn like this could produce, so the empty dict stood in for
    # "the observed value did not persist". It stopped standing for that when
    # `product` joined the carried slots: the question says Solo, so Solo is
    # carried, and it is carried because the caller wrote it rather than because
    # a model saw it. The invariant is about the observed slot, so the assertion
    # now says which slot it means.
    assert "substrate" not in store.carried(session_id)


def test_a_slot_the_caller_also_stated_is_still_remembered(
        assistant, monkeypatch):
    """The exclusion is by provenance, not by "a photograph was attached".

    Someone who uploads an image and types "it's brick" has stated a substrate,
    and dropping it because an image happened to be in the same request would
    lose a fact the person actually gave — over-correcting the leak into a
    conversation that forgets what it was told.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    store = ui.SessionStore()
    session_id = store.open()

    handler = Handler.__new__(Handler)
    handler.sessions = store
    handler.session_id = session_id
    question = "how much water does Solo need on a brick wall"
    handler._remember(question, assistant.ask(question, images=[PNG]))

    # The stated value, and specifically not the seen one. Compared by slot
    # rather than whole for the reason given above: the question also names a
    # product, and this test is not about which products are remembered.
    assert store.carried(session_id)["substrate"] == "brick"


# ------------------------------------------------------- 4 no new answer route


def test_a_cause_question_with_a_photograph_still_hands_off(
        assistant, monkeypatch):
    """Filling slots from an image does not license a diagnosis.

    Router step 3 sends a cause question to the hand-off because judging a wall
    is a liability decision the technical team makes, and that reason is
    untouched by the assistant being able to see the wall. The photograph may
    settle what the wall is built of; it may not settle what is wrong with it.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    reply = assistant.ask("what is causing the damp patches on this wall",
                          images=[PNG])
    _part, answer = reply.parts[0]
    assert answer.path in (Path_.DIAGNOSIS.value, Path_.REFUSE.value)
    assert "technical team" in answer.text


def test_vision_cannot_fill_the_cause_slot_at_all():
    """`cause_asked` is absent from the enum the model is constrained to."""
    assert "cause_asked" not in vision.VISION_SLOTS
    schema = vision.observation_schema()
    enum = schema["properties"]["observations"]["items"]["properties"]["attribute"]["enum"]
    assert "cause_asked" not in enum


# ---------------------------------------------- 5 failure reduces coverage only


def test_a_failed_perception_leaves_the_router_where_no_photograph_would(
        assistant, monkeypatch):
    """An unreachable model must change nothing about the answer.

    Asserted as an equality against the same question asked with no image at
    all, rather than against a hard-coded path, so it keeps holding if the
    routing of that question ever changes for some other reason.
    """
    without = assistant.ask("which plaster should I use")

    sees(monkeypatch, vision.Perception(error="the vision call failed: refused"))
    assistant.cache.clear()
    with_a_broken_camera = assistant.ask("which plaster should I use",
                                         images=[PNG])

    assert ([a.path for _q, a in with_a_broken_camera.parts]
            == [a.path for _q, a in without.parts])
    assert ([a.text for _q, a in with_a_broken_camera.parts]
            == [a.text for _q, a in without.parts])


def test_the_seam_adds_no_exception_handler_of_its_own(assistant, monkeypatch):
    """`observe()` owns the failure, so a raising stand-in must escape.

    This is the inverse of the test above and it is the one that keeps the
    property honest. If the join grew a `try` of its own, a genuinely broken
    vision stage would be indistinguishable from an absent photograph and the
    `error` the `Perception` carries would never reach anyone.
    """
    def explode(*_a, **_k):
        raise RuntimeError("a real failure, not a degraded one")

    monkeypatch.setattr(vision, "observe", explode)
    with pytest.raises(RuntimeError):
        assistant.ask("how much water does Solo need", images=[PNG])


# ------------------------------------------------------------------ the cache


def test_the_same_value_from_two_origins_is_two_cache_keys():
    """Because it is two different printed sentences, so two different answers.

    Sharing the key would serve "as you told me earlier in this conversation"
    to somebody who had told us nothing — the provenance defect reintroduced
    through a cache, which is a correctness failure rather than a stale entry.
    """
    common = ("what plaster", ("public",), "snap", "gen-model", "chunk-v1",
              {"substrate": "brick"})
    carried_key = AnswerCache.key(*common)
    observed_key = AnswerCache.key(*common,
                                   {"substrate": Provenance.OBSERVED})
    assert carried_key != observed_key


def test_the_cache_still_hits_for_two_identical_ordinary_questions():
    """The extra element must not have turned every key into a miss."""
    common = ("what plaster", ("public",), "snap", "gen-model", "chunk-v1",
              {"substrate": "brick"})
    assert AnswerCache.key(*common) == AnswerCache.key(*common, None)


def test_a_photographed_answer_is_not_served_to_a_caller_without_one(
        assistant, monkeypatch):
    """The same property, end to end through the engine rather than the key.

    The first call establishes the entry, the second asks the identical words
    with the substrate carried from a session instead. Two different sentences
    have to print, which they can only do if the two calls missed each other.
    """
    sees(monkeypatch, perception("substrate", "stone"))
    photographed = assistant.ask("how much water does Solo need", images=[PNG])
    carried = assistant.ask("how much water does Solo need",
                            carried={"substrate": "stone"})

    assert "from the photograph you sent" in photographed.parts[0][1].text
    assert "from the photograph you sent" not in carried.parts[0][1].text
    assert "as you told me earlier" in carried.parts[0][1].text


# ----------------------------------------------------------- the CLI surface


def test_the_cli_passes_repeated_image_flags_through(tmp_path, monkeypatch):
    """`--image` is repeatable and reaches `ask` as a list, in order."""
    from assistant.interfaces import cli

    build_repo(tmp_path, two_documents=True).close()
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)

    seen = {}
    real_ask_turn = Assistant.ask_turn

    def record(self, turn, *args, **kwargs):
        # The CLI answers through the graph now, so the images arrive inside a
        # `TurnInput` rather than as an `ask()` keyword. What the test is for is
        # unchanged: `--image` is repeatable and the files reach the engine in
        # the order they were given.
        seen["images"] = list(turn.images)
        return real_ask_turn(self, turn, *args, **kwargs)

    monkeypatch.setattr(Assistant, "ask_turn", record)
    monkeypatch.setattr(vision, "observe",
                        lambda *_a, **_k: vision.Perception(error="not run"))

    code = cli.main(["-q", "how much water does Solo need",
                     "--db", str(tmp_path / "index" / "knowledge.db"),
                     "--image", "one.png", "--image", "two.png"])
    assert code == 0
    assert seen["images"] == ["one.png", "two.png"]


# -------------------------------------------------------- the upload boundary


def test_sniffing_identifies_by_content_and_refuses_everything_else():
    """The allowlist is by signature, and a filename is not a signature."""
    assert sniff(PNG) == "image/png"
    assert sniff(b"\xff\xd8\xff\xe0 whatever") == "image/jpeg"
    assert sniff(b"GIF89a....") == "image/gif"
    assert sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    # A RIFF container that is not a WEBP shares its first four bytes, which is
    # why the check looks twelve bytes in rather than at the prefix.
    assert sniff(b"RIFF\x00\x00\x00\x00WAVEfmt ") == ""
    assert sniff(NOT_AN_IMAGE) == ""
    assert sniff(b"") == ""


def test_the_session_budget_stops_charging_at_the_cap():
    """A per-caller counter, bounded in both directions."""
    budget = ui.UploadBudget(limit=3, tracked=2)
    assert budget.take("a", 2) == 2
    assert budget.take("a", 2) == 1              # one left of the three
    assert budget.take("a", 2) == 0
    # A different caller is unaffected by the first one's spend.
    assert budget.take("b", 3) == 3
    # And the store of counters cannot grow without limit.
    budget.take("c", 1)
    assert len(budget._counts) == 2


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A real threaded server, because the guards are on the request itself.

    Testing `_read_upload` against a hand-made handler would test the parsing
    and skip the thing that matters: that an oversized or untyped body is
    refused with a status code before it is read, by the server that actually
    receives it.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    monkeypatch.setattr(vision, "observe",
                        lambda *_a, **_k: perception("substrate", "stone"))
    # These tests are about the *enabled* path: no provider is injected, so the
    # request goes through the default one, and `ASSISTANT_VISION_DEMO` decides
    # whether that exists. Switched off the turn would take decision 16's
    # hand-off instead, which `tests/test_vision_flag.py` covers.
    monkeypatch.setenv(vision.VISION_DEMO_FLAG, "1")

    build_repo(tmp_path, two_documents=True).close()
    repo = open_repository(str(tmp_path / "index" / "knowledge.db"), dsn="",
                           thread_safe=True)
    saved = {name: Handler.__dict__.get(name) for name in
             ("assistant", "meta", "audiences", "sessions", "uploads")}
    Handler.assistant = Assistant(repo)
    Handler.meta = "test"
    Handler.audiences = ("public",)
    Handler.sessions = ui.SessionStore()
    Handler.uploads = ui.UploadBudget()

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
        for name, value in saved.items():
            if value is None:
                if name in Handler.__dict__:
                    delattr(Handler, name)
            else:
                setattr(Handler, name, value)


def multipart(fields: dict, files: list) -> tuple[bytes, str]:
    """A multipart body, built by hand so the test controls every byte.

    `files` are `(filename, bytes)` pairs. The filename is included precisely
    because the server must be shown to ignore it.
    """
    boundary = "----" + uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                  .encode())
        out.write(value.encode("utf-8") + b"\r\n")
    for filename, data in files:
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="image"; '
                  f'filename="{filename}"\r\n'.encode())
        out.write(b"Content-Type: application/octet-stream\r\n\r\n")
        out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def post(base: str, body: bytes, content_type: str, path: str = "/ask",
         headers: dict | None = None):
    request = urllib.request.Request(base + path, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    return urllib.request.urlopen(request, timeout=30)


def test_an_uploaded_photograph_fills_a_slot_and_says_where_it_came_from(server):
    """The whole seam, over HTTP, on the surface the demonstration uses."""
    import json
    body, content_type = multipart({"q": "how much water does Solo need"},
                                   [("wall.png", PNG)])
    payload = json.loads(post(server, body, content_type).read())
    assert payload["images_read"] == 1
    facts = {f["slot"]: f for part in payload["parts"] for f in part["facts"]}
    assert facts["substrate"]["value"] == "stone"
    assert facts["substrate"]["provenance"] == "observed"
    # And it did not land in the session, which is the property §3.4 names.
    # By slot, not whole: `product` is carried now and is read from the question
    # text, never from the image. What must not appear here is the substrate the
    # model saw.
    assert "substrate" not in payload["session_slots"]


def test_an_oversized_body_is_refused_before_it_is_read(server):
    """The declared length decides, so nothing large is ever buffered.

    Spoken to the socket for the same reason as the test below it: the
    refusal deliberately does *not* drain an oversized body — reading twelve
    megabytes to say "too large" is the work the cap exists to avoid — so the
    connection closes, and a client still writing sees a reset instead of the
    status. Declaring the length and sending no body is the same request as
    far as the guard is concerned, and it cannot race.
    """
    host, port = server.removeprefix("http://").split(":")
    _, content_type = multipart({"q": "hello"}, [("wall.png", PNG)])
    request = "\r\n".join((
        "POST /ask HTTP/1.1",
        f"Host: {host}:{port}",
        f"Content-Type: {content_type}",
        f"Content-Length: {ui.MAX_UPLOAD_BYTES + 1}",
        "", "",
    ))
    with socket.create_connection((host, int(port)), timeout=30) as client:
        client.sendall(request.encode("ascii"))
        status = client.makefile("rb").readline().decode("latin-1")

    assert "413" in status, status


def test_a_body_with_no_length_is_refused(server):
    """Reading until the connection closes is the unbounded read, so: no.

    Spoken to the socket rather than through `urllib`, and the reason is the
    server's own correctness. A body of undeclared length cannot be drained —
    draining it *is* the unbounded read — so the refusal closes the connection
    instead. A client still writing when that happens sees a reset rather than
    the status, which made this test fail about one run in six against a server
    doing exactly the right thing. Sending the headers and no body removes the
    race entirely: there is nothing left to write when the refusal arrives, so
    a failure here is the guard going missing rather than the clock.
    """
    host, port = server.removeprefix("http://").split(":")
    _, content_type = multipart({"q": "hello"}, [])
    request = "\r\n".join((
        "POST /ask HTTP/1.1",
        f"Host: {host}:{port}",
        f"Content-Type: {content_type}",
        "Transfer-Encoding: chunked",
        "", "",                      # the blank line that ends the headers
    ))
    with socket.create_connection((host, int(port)), timeout=30) as client:
        client.sendall(request.encode("ascii"))
        status = client.makefile("rb").readline().decode("latin-1")

    assert "411" in status, status


def test_a_body_that_is_not_multipart_is_refused(server):
    """Declared type checked before the body is parsed."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server, b'{"q": "hello"}', "application/json")
    assert caught.value.code == 415


def test_a_file_that_is_not_an_image_is_dropped_and_declared(server):
    """Never passed on, and never silently ignored either.

    Both halves matter. A PHP file named `wall.png` must not reach the model,
    and a request whose attachment vanished without a word would answer as
    though nothing had been sent — the quiet failure this codebase refuses.
    """
    import json
    body, content_type = multipart({"q": "how much water does Solo need"},
                                   [("wall.png", NOT_AN_IMAGE)])
    payload = json.loads(post(server, body, content_type).read())
    assert payload["images_read"] == 0
    assert any("not a recognised image" in note
               for note in payload["upload_notes"])
    facts = {f["slot"] for part in payload["parts"] for f in part["facts"]}
    assert "substrate" not in facts


def test_the_uploaded_filename_reaches_nothing(server):
    """A caller-chosen string is not evidence and is not echoed back.

    The name here is a traversal attempt with a script tag in it, which covers
    the two things a filename could do if it were used: reach a path, or reach
    the page. It is used for neither, so it appears in no response at all.
    """
    import json
    hostile = "../../<script>alert(1)</script>.png"
    body, content_type = multipart({"q": "how much water does Solo need"},
                                   [(hostile, PNG)])
    raw = post(server, body, content_type).read().decode("utf-8")
    assert "script" not in raw
    assert ".." not in raw
    assert json.loads(raw)["images_read"] == 1


def test_a_session_may_not_send_photographs_without_limit(server):
    """Perception is minutes of CPU per image, so the count is capped per caller.

    The cookie is carried between requests, which is what makes this a session
    cap rather than a per-request one — and the answer keeps coming: reaching
    the cap stops the images being read, never the question being answered.
    """
    import json
    from http.cookies import SimpleCookie

    cookie = ""
    last = None
    for _attempt in range(MAX_IMAGES_PER_SESSION + 2):
        body, content_type = multipart({"q": "how much water does Solo need"},
                                       [("wall.png", PNG)])
        response = post(server, body, content_type,
                        headers={"Cookie": cookie} if cookie else {})
        jar = SimpleCookie(response.headers.get("Set-Cookie", ""))
        morsel = jar.get(ui.SESSION_COOKIE)
        if morsel:
            cookie = f"{ui.SESSION_COOKIE}={morsel.value}"
        last = json.loads(response.read())

    assert last["images_read"] == 0
    assert any("as many photographs" in note for note in last["upload_notes"])
    assert last["parts"], "the question is still answered after the cap"


def test_more_images_than_one_request_may_carry_are_trimmed(server):
    """A per-request cap as well as a per-session one, and it says so."""
    import json
    files = [("wall.png", PNG)] * (ui.MAX_IMAGES_PER_REQUEST + 3)
    body, content_type = multipart({"q": "how much water does Solo need"}, files)
    payload = json.loads(post(server, body, content_type).read())
    assert payload["images_read"] == ui.MAX_IMAGES_PER_REQUEST
    assert any("Only the first" in note for note in payload["upload_notes"])


def test_a_post_to_an_unknown_path_is_a_404(server):
    """The upload path is not a second router around the one that exists."""
    body, content_type = multipart({"q": "hello"}, [])
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server, body, content_type, path="/upload")
    assert caught.value.code == 404


def test_a_post_with_no_photograph_answers_like_a_get(server):
    """The upload form is still a way to ask an ordinary question."""
    import json
    body, content_type = multipart({"q": "how much water does Solo need"}, [])
    payload = json.loads(post(server, body, content_type).read())
    assert payload["images_read"] == 0
    assert payload["parts"]


def test_a_multipart_declaration_with_no_parts_is_refused(server):
    """The header may say multipart while the body is not one.

    Checked after parsing as well as before it, because the declared type is a
    claim by the caller and the parse is the only thing that can contradict it.
    A body that turns out to hold no parts at all is not a form, and guessing at
    what was meant is how a parser ends up accepting whatever it was handed.
    """
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server, b"not a form at all",
             "multipart/form-data; boundary=----nothing")
    assert caught.value.code == 415


def test_a_form_part_with_no_name_and_no_filename_is_ignored(server):
    """Neither a field nor an attachment, so it is neither read nor stored."""
    import json
    boundary = "----" + uuid.uuid4().hex
    body = (f"--{boundary}\r\n".encode()
            + b"Content-Disposition: form-data\r\n\r\n"
            + b"anonymous\r\n"
            + f"--{boundary}\r\n".encode()
            + b'Content-Disposition: form-data; name="q"\r\n\r\n'
            + b"how much water does Solo need\r\n"
            + f"--{boundary}--\r\n".encode())
    payload = json.loads(
        post(server, body, f"multipart/form-data; boundary={boundary}").read())
    assert payload["question"] == "how much water does Solo need"
    assert payload["images_read"] == 0


def test_the_page_tells_a_person_what_happened_to_their_attachment(server):
    """The same notes the JSON carries, on the surface a person is looking at.

    Reported above the answer rather than inside it: what happened to an upload
    is a fact about the request, not about the published material, and putting
    it in the answer would put an unsourced sentence on a page where every other
    sentence carries a citation.
    """
    body, content_type = multipart({"q": "how much water does Solo need"},
                                   [("wall.png", NOT_AN_IMAGE)])
    page = post(server, body, content_type, path="/").read().decode("utf-8")
    assert "not a recognised image" in page
