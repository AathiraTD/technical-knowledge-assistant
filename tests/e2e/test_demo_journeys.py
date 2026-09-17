"""The journeys the demonstration walks, in the order it walks them.

Eight journeys, A to H. Each one is a thing a person will do in front of a
panel, and the reason to automate them is narrow and worth stating: a
demonstration fails in ways a unit test cannot see. The library can route
correctly while the citation renders as plain text; the session can carry a
substrate while `New chat` leaves the cookie in place; the upload boundary can
accept a photograph while the page gives no sign it did. Every failure of that
kind is invisible to `eval/run.py`, which drives the library, and to
`tests/test_ui_server.py`, which drives HTTP without a browser.

**These do not grade answers.** `eval/run.py` owns whether an answer is right,
on the library, where decision 15 deliberately keeps it. What these assert is
the route the deterministic router took, the slots the conversation carried,
and what rendered -- facts that are stable across runs because decision 7's G4
note records that the same question asked twice produces different prose and
identical evidence. An assertion on wording here would be an assertion on the
one thing this system does not promise.

**Speed is a design constraint, not an afterthought.** An uncached compose
costs tens of seconds to minutes on this hardware. Every journey that is really
about the browser picks a question the policy gate or the router answers
without generation, so the smoke suite runs in about a minute:

    python -m pytest tests/e2e -m smoke     # pre-flight, no generation
    python -m pytest tests/e2e              # pre-demo, everything

Only journey B is marked `slow`, because multi-turn context retention cannot be
demonstrated without a real answer to carry context into.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from .conftest import (
    CERTIFICATION_QUESTION,
    ONE_PIXEL_PNG,
    QUANTITY_QUESTION,
    ROOT,
    ROUTED_QUESTION,
    SELECTION_QUESTION,
    answer_text,
    ask,
    ask_json,
    expand,
    route_of,
)

# Perception runs before routing, so a request carrying an image costs a vision
# model call even when the question itself never reaches retrieval. On a
# processor with no graphics card that is minutes; the default Playwright
# timeout of thirty seconds fails the test while the server is still working.
PERCEPTION_TIMEOUT = 300_000
COMPOSE_TIMEOUT = 420_000
ROUTED_TIMEOUT = 60_000


# --------------------------------------------------------------------- A. RAG

@pytest.mark.smoke
def test_a_factual_question_answers_with_a_citation_and_openable_sources(page):
    """Journey A. The first thing anyone will ask, and the three things it owes.

    A quantity question, because router step 6 sends it to extract: a real
    retrieval over the coverage and pack-size passages, printed verbatim by code
    with the sum refused. That is a genuine grounded answer with a genuine
    citation and it costs no generation, which is what makes this the journey to
    open a demonstration with.

    Three assertions, and each is a different promise. The marker is the claim
    that a sentence is sourced; the source list is the claim being checkable;
    the link is the claim being followable back to Lime Green's own document.
    """
    page.goto("/")
    payload = ask_json(page, QUANTITY_QUESTION, timeout=ROUTED_TIMEOUT)

    assert route_of(payload) == "extract", (
        f"expected the extract path, got {route_of(payload)!r}")
    assert payload["parts"][0]["sources"], "an answer came back citing nothing"

    assert page.locator(".citation").count() > 0, "no [n] marker rendered"

    panel = expand(page, "Sources")
    links = panel.locator(".source-link")
    assert links.count() > 0, "the sources panel opened on nothing"
    href = links.first.get_attribute("href")
    assert href and href.startswith("http"), f"source link href was {href!r}"


@pytest.mark.smoke
def test_the_page_says_which_index_is_answering(page):
    """Journey A, the part a panel asks about second.

    A demonstration that cannot name the index it is answering from is a
    demonstration of nothing in particular. The footer carries the document and
    passage counts, the embedding and generation models and the abstention
    threshold, so the claim on the slide can be read off the page.
    """
    page.goto("/")
    meta = page.inner_text("#index-meta")
    assert "documents" in meta and "passages" in meta, meta
    assert "threshold" in meta, meta


# ------------------------------------------------------------- B. multi-turn

@pytest.mark.slow
def test_a_second_turn_inherits_the_wall_and_the_product_from_the_first(page):
    """Journey B. The conversation decision 20 rebuilt the turn as a graph for.

    Two turns, and the second says neither what the wall is nor which product it
    means. If the substrate and the product do not survive the turn boundary,
    the follow-up is answered about a different building -- which is the failure
    the hand-rolled orchestration actually had, and the reason `assistant/graph.py`
    exists.

    Asserted on `session_slots`, not on prose. The conversation state is what
    carries; the wording that carries it is the thing decision 7's G4 note
    records as not reproducible.

    Marked slow and given seven minutes because the first turn genuinely
    composes: several passages from several documents bear on whether Ultra
    suits a solid brick wall internally and at what thickness, so step 8 fires
    and the model runs. There is no honest way to demonstrate context retention
    without paying for one real answer.
    """
    page.goto("/")

    first = ask_json(
        page,
        "I have an old solid brick wall and want to improve its insulation. "
        "Would Lime Green Ultra be suitable internally, and what thickness "
        "can it be applied at?",
        timeout=COMPOSE_TIMEOUT)

    carried = first["session_slots"]
    assert carried.get("substrate"), (
        f"turn 1 established no substrate; slots were {carried}")
    assert carried.get("location") == "internal", (
        f"turn 1 lost 'internally'; slots were {carried}")

    second = ask_json(page, "How much would I need for 30 m² at 25 mm?",
                      timeout=COMPOSE_TIMEOUT)

    after = second["session_slots"]
    assert after.get("substrate") == carried.get("substrate"), (
        "the wall changed between turns: "
        f"{carried.get('substrate')!r} became {after.get('substrate')!r}")
    assert after.get("location") == "internal", (
        f"the second turn lost the location; slots were {after}")

    # The quantity question names no product, so the only way it can be about
    # Ultra is by carrying it. A calculation that quietly switched product is
    # the expensive version of this failure.
    assert "ultra" in json.dumps(second).lower(), (
        "the follow-up lost the product the conversation was about")

    # Router step 6: a quantity is extracted from the published coverage and
    # pack-size passages and the multiplication is refused, which is the
    # behaviour the calculation restriction exists to produce.
    assert route_of(second) in ("extract", "compose"), route_of(second)


@pytest.mark.smoke
def test_a_follow_up_stays_in_one_conversation_without_a_model(page):
    """Journey B, the half that needs no generation, so smoke can cover it.

    The substrate is stated outright and the follow-up carries it. Everything
    about the turn boundary that could break -- the cookie, the session store,
    the slot merge -- is exercised here; only the composed answer is not.
    """
    page.goto("/")

    first = ask_json(page, "I have an internal brick wall.", timeout=ROUTED_TIMEOUT)
    assert first["session_slots"].get("substrate") == "brick", first["session_slots"]

    second = ask_json(page, SELECTION_QUESTION, timeout=ROUTED_TIMEOUT)
    assert second["session_slots"].get("substrate") == "brick", (
        "the wall did not survive the turn; slots were "
        f"{second['session_slots']}")


# ------------------------------------------------------------------ C. image

@pytest.mark.smoke
def test_an_attached_photograph_is_accepted_and_shown_as_attached(page):
    """Journey C. The upload boundary, and the page admitting it happened.

    Two failures live here and they look identical from the outside. The bytes
    can fail to cross the boundary, which is what `images_read` catches. Or they
    can cross it and leave no trace on the page, which is what the attachment
    chip catches -- and which is the one a person demonstrating this actually
    hits, because the file input is `display:none` and is cleared on send.

    Asserted on `images_read` rather than on anything the vision model says:
    perception costs minutes, and whether a one-pixel PNG yields a useful
    observation is not a question about the browser.
    """
    page.goto("/")
    page.set_input_files("#file-input", {
        "name": "wall.png", "mimeType": "image/png", "buffer": ONE_PIXEL_PNG})

    # Before sending: the page says what is attached.
    attached = page.locator(".attached-file")
    assert attached.count() == 1, "attaching a photograph showed nothing"
    assert "wall.png" in attached.first.inner_text()

    payload = ask_json(page, ROUTED_QUESTION, timeout=PERCEPTION_TIMEOUT)

    assert payload["images_read"] == 1, (
        f"the server read {payload['images_read']} images; "
        f"upload notes were {payload.get('upload_notes')}")
    assert not payload["upload_notes"], (
        f"a valid PNG produced complaints: {payload['upload_notes']}")

    # After sending: the person's own bubble records what went with the question.
    assert "wall.png" in page.locator(".message.user").last.inner_text()


@pytest.mark.smoke
def test_an_attachment_that_is_not_an_image_says_so_on_the_page(page):
    """Journey C, the failure half. Refused by content, and reported.

    Fast for a reason that is itself the behaviour: nothing recognised as an
    image means nothing to perceive, so the vision model is never called. A
    regression that started accepting these bytes would show up here as a
    timeout before it showed up as a wrong assertion.

    The note is asserted on the rendered page rather than only in the payload,
    because an upload dropped in silence is the failure mode -- the JSON having
    said so is no use to the person looking at the screen.
    """
    page.goto("/")
    page.set_input_files("#file-input", {
        "name": "wall.png",                   # the name says PNG; the bytes do not
        "mimeType": "image/png",
        "buffer": b"this is not a picture, whatever it is called"})

    payload = ask_json(page, ROUTED_QUESTION, timeout=ROUTED_TIMEOUT)

    assert payload["images_read"] == 0
    assert "not a recognised image" in " ".join(payload["upload_notes"])

    note = page.locator(".upload-note")
    assert note.count() > 0, "the attachment was refused without telling anyone"
    assert "not a recognised image" in note.first.inner_text()


# --------------------------------------------------------------- D. ask-back

@pytest.mark.smoke
def test_a_selection_question_asks_back_and_then_resumes_the_original(page):
    """Journey D. The load-bearing slot, and the cycle closing on it.

    "Which plaster should I use?" cannot be answered without knowing the wall,
    and decision 10 makes substrate the slot that asks back rather than assuming
    -- a recommendation on an assumed wall being the costly error the whole
    design exists to avoid.

    The half worth automating is the second one. Asking back is easy; resuming
    is where the hand-rolled version failed, and `interrupt()`/`Command(resume=)`
    in `assistant/graph.py` is what replaced it. So this asserts the answer to
    the ask-back is treated as an answer -- the substrate lands in the session
    and the turn stops asking -- rather than as a new question about the word
    "brick".
    """
    page.goto("/")

    asked = ask_json(page, SELECTION_QUESTION, timeout=ROUTED_TIMEOUT)
    assert route_of(asked) == "ask back", (
        f"a question with no substrate took {route_of(asked)!r}")
    assert "wall built of" in answer_text(asked).lower(), answer_text(asked)[:300]

    resumed = ask_json(page, "brick", timeout=COMPOSE_TIMEOUT)

    assert resumed["session_slots"].get("substrate") == "brick", (
        "the reply to the ask-back was not recorded as the substrate; "
        f"slots were {resumed['session_slots']}")
    assert route_of(resumed) != "ask back", (
        "the assistant asked the same question twice")


# ----------------------------------------------------------------- E. safety

@pytest.mark.smoke
@pytest.mark.parametrize("question, must_mention", [
    (CERTIFICATION_QUESTION, "0800 538 5746"),
    ("My gable wall has a 10 mm crack, is the house safe?", "0800 538 5746"),
])
def test_a_judgement_the_company_will_not_make_is_handed_over_not_answered(
        page, question, must_mention):
    """Journey E. Two questions that must never reach retrieval.

    Compliance sign-off and structural safety are both on the policy gate, and
    both for the same reason: they are judgements Lime Green's technical team
    makes and stands behind, not facts the corpus publishes. The gate fires
    before slot detection and before retrieval, so no amount of phrasing reaches
    the model with them.

    The contact line is asserted because a refusal that hands over nothing is
    not the designed outcome -- and because the number is harvested from the
    crawled contact page into the manifest rather than typed into a prompt,
    which is what stops the assistant inventing one.
    """
    page.goto("/")
    payload = ask_json(page, question, timeout=ROUTED_TIMEOUT)

    assert route_of(payload) == "route", (
        f"a policy question took {route_of(payload)!r} instead of routing")
    assert must_mention in answer_text(payload), answer_text(payload)[:400]
    assert not payload["parts"][0]["sources"], (
        "a routed referral cited retrieved evidence, which it never retrieves")


@pytest.mark.smoke
def test_a_public_caller_cannot_ask_its_way_into_staff_material(page):
    """Journey E, the audience half.

    The audience set narrows from what the server was started with and never
    widens. Asserted through the URL because that is where a caller would
    actually try it, and because a regression here is a leak rather than a bug.

    The parameter is `a`. An earlier version of this sent `audience=staff`, a
    name the server does not read at all: it passed, and proved nothing, because
    an attempt at elevation the code never sees is not an attempt at elevation.
    """
    page.goto("/?q=How+much+does+a+bag+cost%3F&a=staff")

    footer = page.inner_text("#footer-audience").lower()
    assert "staff" not in footer
    assert "public" in footer


# --------------------------------------------------------------- F. new chat

@pytest.mark.smoke
def test_new_chat_forgets_the_wall_the_previous_conversation_established(page):
    """Journey F. The button has to reach the server, because only it can do this.

    The cookie is HttpOnly, so the `document.cookie` write the old `newChat()`
    attempted was discarded by the browser in silence: the page reloaded, the
    old cookie went back up with the next request, and the conversation someone
    thought they had ended carried its substrate into the next question. `/new`
    mints a new session server-side instead.

    Asserted on the slots rather than only on the cookie, because the cookie
    changing is the mechanism and the wall being forgotten is the promise.
    """
    page.goto("/")
    established = ask_json(page, "I have an internal brick wall.",
                           timeout=ROUTED_TIMEOUT)
    assert established["session_slots"].get("substrate") == "brick"

    page.click(".new-chat-btn")
    page.wait_for_load_state("networkidle")

    after = ask_json(page, SELECTION_QUESTION, timeout=ROUTED_TIMEOUT)
    assert not after["session_slots"].get("substrate"), (
        "New chat left the previous conversation's wall in place; slots were "
        f"{after['session_slots']}")
    assert route_of(after) == "ask back", (
        "the new conversation answered from the old one's substrate")


@pytest.mark.smoke
def test_new_chat_clears_what_the_page_was_showing(page):
    """Journey F, the visible half. A new conversation looks like one."""
    page.goto("/")
    ask(page, ROUTED_QUESTION, timeout=ROUTED_TIMEOUT)
    assert page.locator(".message").count() > 0

    page.click(".new-chat-btn")
    page.wait_for_load_state("networkidle")

    assert page.locator(".landing").count() == 1, (
        "New chat left the previous conversation on screen")
    assert page.locator(".message.assistant .tag").count() == 0


# ----------------------------------------------------------------- G. reload

@pytest.mark.smoke
def test_a_reload_keeps_the_conversation_even_though_the_page_clears(page):
    """Journey G. What survives a refresh, stated as the design actually is.

    This is the journey most likely to be misread in a demonstration, so it
    asserts the real behaviour rather than a flattering version of it. Two
    different things are going on and only one of them persists:

    * The **conversation** is a server fact. It is named by an HttpOnly cookie
      and holds the slots earlier turns established, so a reload continues it --
      the wall is still known and the next question inherits it.
    * The **transcript** is a page fact. Answers are rendered into the DOM by
      the page's own script, and nothing re-renders them, so a reload shows the
      landing panel again.

    So the honest claim is "the assistant remembers, the screen does not", and
    that is what this pins. A future change that server-renders the transcript
    would break the second assertion, which is the right thing for it to do.
    """
    page.goto("/")
    ask_json(page, "I have an internal brick wall.", timeout=ROUTED_TIMEOUT)
    before = [c for c in page.context.cookies() if c["name"] == "tka_session"]

    page.reload()
    page.wait_for_load_state("networkidle")

    after = [c for c in page.context.cookies() if c["name"] == "tka_session"]
    assert after[0]["value"] == before[0]["value"], (
        "a reload started a new conversation")
    assert page.locator(".landing").count() == 1, (
        "the transcript re-rendered; this suite's claim about reload is stale "
        "and should be updated rather than this assertion deleted")

    resumed = ask_json(page, SELECTION_QUESTION, timeout=ROUTED_TIMEOUT)
    assert resumed["session_slots"].get("substrate") == "brick", (
        "the conversation did not survive the reload; slots were "
        f"{resumed['session_slots']}")


# ------------------------------------------------------------------ H. trace

@pytest.mark.smoke
def test_the_correlation_id_the_browser_gets_reads_a_real_trace(page):
    """Journey H. The observability chain, end to end and across processes.

    The browser is handed an id in a header; the operator pastes that id into
    `python -m assistant.trace` and gets the stages of that answer back. Until
    this held, a reported problem could only be investigated by reproducing it.

    This is the journey that answers "why this answer?" from the other side --
    not the disclosure panel on the page, which shows the reader the step and
    the score, but the log an engineer reads afterwards.
    """
    page.goto("/")
    with page.expect_response(lambda r: "/ask" in r.url,
                              timeout=ROUTED_TIMEOUT) as caught:
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


@pytest.mark.smoke
def test_why_this_answer_opens_and_names_the_step_that_decided(page):
    """Journey H, the half the audience sees.

    The diagnostics disclosure is the demonstration's answer to "how do you know
    it did not make that up": it names the router step, the reason and the top
    score, on the page, for the answer being looked at.
    """
    page.goto("/")
    ask(page, QUANTITY_QUESTION, timeout=ROUTED_TIMEOUT)

    panel = expand(page, "Why this answer?")
    keys = panel.locator(".diag-key").all_inner_texts()
    assert "step" in " ".join(keys).lower(), keys
