"""The image demo, end to end: a wall, a quantity, a photograph, a symptom.

`tests/test_vision.py` owns perception in isolation and `tests/test_vision_seam.py`
owns the join between a photograph and the router. What neither covers is the
thing a panel will actually be shown -- **a conversation** -- and the failures
that only exist across turns: a photograph of one wall filling a slot on
another, an observation surviving a subject change, a reading printed as
something the customer said, a recommendation that moved because a model looked
at pixels.

Two rules shape every test here.

**No model, no network.** The vision call is replaced, but the *resolver* is
not: the double returns `Perception` objects and the real `vision.resolve()`
decides what may cross. A stub handing back a finished slot dict would prove
nothing about the vocabulary check, the band, the region or the substrate gate,
which are the four things standing between a model's guess and a
recommendation. The live counterpart is `eval/vision_eval.py`, which is opt-in
because it costs minutes per image.

**The assertions are about what must not happen.** That a photograph of
brickwork yields `substrate=brick` is pleasant and is not what protects
anybody. That it yields no `location`, that it cannot overrule a person, that
it does not survive into a question about a different wall, and that a symptom
in frame never becomes a cause in the answer -- those are the properties worth
a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant.answering import vision
from assistant.infrastructure import ollama
from assistant.answering.answer import (  # noqa: E402
    Provenance,
)
from assistant.turn.conversation import ConversationState, FactStatus, TurnInput
from assistant.answering.engine import (  # noqa: E402
    Assistant,
)
from assistant.indexing.index import (  # noqa: E402
    CHUNKING_VERSION,
)
from assistant.knowledge.model import Caveat, Chunk, Document, DocumentVersion, Snapshot
from assistant.knowledge.store import SQLiteKnowledgeRepository               # noqa: E402

from test_engine import DIMS, quoting, unit                         # noqa: E402

ULTRA = "https://example.invalid/ultra"
ULTRA_PAGE = "https://example.invalid/ultra-page"
SOLO = "https://example.invalid/solo"

CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}

WHOLE = (0.0, 0.0, 1.0, 1.0)


# ------------------------------------------------------------------ a corpus

def _chunk(url, index, section, content, product, axis, doc_type="datasheet",
           authority=1) -> Chunk:
    return Chunk(canonical_url=url, version=1, chunk_index=index,
                 section=section, content=content, product=product,
                 document_type=doc_type, authority=authority,
                 source_date="2024-07-01", embedding=unit(axis))


def _version(url) -> DocumentVersion:
    return DocumentVersion(canonical_url=url, version=1, content_hash="h1",
                           source_path="cache/x.pdf", first_seen_at="2026-01-01",
                           fetched_at="2026-01-01T00:00:00Z",
                           checked_at="2026-01-01T00:00:00Z")


@pytest.fixture
def repo(tmp_path):
    """A small store that can actually support an Ultra recommendation.

    Deliberately says the things the evidence gate looks for -- a substrate the
    sheet names, a thickness, a coverage figure -- and deliberately says
    nothing about *why* a surface dries patchily. Turn four of the scenario
    depends on that absence: the corpus not answering is what makes a refusal
    the correct behaviour rather than a shortfall.
    """
    documents = [
        Document(canonical_url=ULTRA, title="Ultra datasheet",
                 document_type="datasheet", authority=1, product="Ultra",
                 link_text="Ultra Datasheet"),
        Document(canonical_url=ULTRA_PAGE, title="Ultra product page",
                 document_type="product_page", authority=2, product="Ultra",
                 link_text="Ultra"),
        Document(canonical_url=SOLO, title="Solo datasheet",
                 document_type="datasheet", authority=1, product="Solo",
                 link_text="Solo Datasheet"),
    ]
    chunks = [
        _chunk(ULTRA, 0, "Description",
               "Ultra is an insulating lime plaster for internal walls. It is "
               "suitable for solid brick, stone and block masonry backgrounds.",
               "Ultra", 0),
        _chunk(ULTRA, 1, "Thickness",
               "Apply Ultra at a thickness of between 10 and 40mm, building up "
               "in coats of no more than 25mm.", "Ultra", 1),
        _chunk(ULTRA, 2, "Coverage",
               "Ultra covers approximately 1.5 m2 per 25 kg bag at 25mm "
               "thickness.", "Ultra", 2),
        _chunk(ULTRA_PAGE, 0, "Uses",
               "Ultra is used internally on brick and stone to improve thermal "
               "performance while remaining breathable.", "Ultra", 3,
               doc_type="product_page", authority=2),
        _chunk(SOLO, 0, "Mixing",
               "Mix Solo with 5-6 litres of clean water per 25 kg sack.",
               "Solo", 4),
    ]
    snapshot = Snapshot(
        snapshot_id="snap-vision", created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=len(documents),
        chunk_count=len(chunks),
        notes={"products": ["Ultra", "Solo"], "colours": ["York"],
               "merchants": ["The Lime Centre"], "contact": CONTACT})

    made = SQLiteKnowledgeRepository(tmp_path / "index" / "knowledge.db")
    made.publish(documents, [_version(d.canonical_url) for d in documents],
                 chunks, snapshot,
                 [Caveat(ULTRA, "temperature",
                         "Do not apply below 5 degrees C.", "Mixing")])
    try:
        yield made
    finally:
        made.close()


@pytest.fixture
def assistant(repo, monkeypatch):
    # Every question embeds to the same axis, so retrieval is stable and the
    # test is about state rather than about similarity.
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))
    monkeypatch.setattr(ollama, "generate", quoting)
    return Assistant(repo, source="test")


# ----------------------------------------------------------- a vision double

def observation(attribute, value, confidence=0.93, region=WHOLE,
                note="") -> vision.Observation:
    return vision.Observation(attribute=attribute, value=value,
                              confidence=confidence, image="IMG_001",
                              region=region, observation=note)


class Camera:
    """A `VisionProvider` that runs the **real** resolver over stated readings.

    The double is the model, not the rules. `observe()` builds a `Perception`
    exactly as `_decode` would have and then calls `vision.resolve`, so the
    vocabulary check, the confidence band, the region requirement, the conflict
    rule and the substrate gate all run for real. A double that returned a
    finished `{"substrate": "brick"}` would let every one of those regress
    without a single test going red.
    """

    def __init__(self, *observations, cannot=("the moisture source",),
                 truncated=False):
        self.observations = observations
        self.cannot = cannot
        self.truncated = truncated
        self.calls = 0
        self.images_seen: list = []

    def observe(self, images):
        self.calls += 1
        self.images_seen.append(list(images))
        perception = vision.Perception(
            observations=tuple(self.observations),
            cannot_determine_from_image=tuple(self.cannot),
            image="IMG_001", model="stub-vlm", truncated=self.truncated)
        return vision.resolve([perception])


def turn(question, index=1, images=(), session="demo", history=""):
    made = TurnInput(raw_question=question, turn_index=index,
                     images=tuple(images), session_id=session)
    object.__setattr__(made, "history", history)
    return made


def perception_of(reply) -> dict:
    """The photograph report the turn attached, or `{}` if there was none."""
    for _question, answer in reply.parts:
        found = answer.diagnostics.get("perception")
        if found:
            return found
    return {}


def text_of(reply) -> str:
    return " ".join(a.text or a.body or "" for _q, a in reply.parts).lower()


def certainties(report) -> dict:
    return {o["attribute"]: o["certainty"] for o in report["observations"]}


# =====================================================================
#  The four-turn scenario
# =====================================================================

def test_the_demo_scenario_carries_the_wall_and_adds_the_photograph(assistant):
    """Ultra on internal brick, a quantity, a photograph, then a symptom.

    One test rather than four because the thing being asserted is continuity,
    and a continuity property split across four tests is four tests that each
    set up the state they meant to inherit.
    """
    state = ConversationState()
    camera = Camera(
        observation("substrate", "brickwork", 0.94,
                    note="coursed brickwork with recessed joints"),
        observation("exposed_masonry", "exposed", 0.92),
        observation("existing_finish", "none", 0.88),
        observation("location", "internal", 0.95),
        observation("staining", "patchy", 0.71),
        cannot=("whether the wall is internal or external",
                "the moisture source", "the mortar mix"))

    # -- turn 1: the wall, the objective and the product ------------------
    first, state = assistant.ask_turn(turn(
        "I have an old solid brick wall and want to improve its insulation. "
        "Would Lime Green Ultra be suitable internally?", 1), state)

    assert state.active()["substrate"] == "brick"
    assert state.active()["location"] == "internal"
    assert not perception_of(first), "no photograph was sent on turn one"

    # -- turn 2: the quantity, on the same wall ---------------------------
    second, state = assistant.ask_turn(turn(
        "How much would I need for 30 m2 at 25 mm?", 2), state)

    assert state.active()["substrate"] == "brick", "the wall did not change"
    assert state.active()["location"] == "internal"
    assert state.active().get("product") == "ultra", (
        "the product the whole conversation is about was dropped")

    # -- turn 3: the photograph -------------------------------------------
    third, state = assistant.ask_turn(turn(
        "This is the wall I'm talking about. What can you reliably identify "
        "from the photo, and does anything here change the guidance?", 3,
        images=["wall.jpg"]), state, vision=camera)

    assert camera.calls == 1, "the photograph never reached the perception stage"
    report = perception_of(third)
    assert report, "the turn read a photograph and reported nothing"

    # The readings are typed and each carries its own certainty.
    seen = certainties(report)
    assert seen["substrate"] == "OBSERVED"
    assert seen["exposed_masonry"] == "OBSERVED"
    assert seen["staining"] == "LIKELY", (
        "a 0.71 reading was reported as firmly as a 0.94 one")

    # The person's own words still hold the slots, and the photograph agreeing
    # with them does not restate them as something a model saw.
    assert state.active()["substrate"] == "brick"
    assert state.facts["substrate"].current.provenance in (
        Provenance.STATED, Provenance.CARRIED)
    assert state.facts["substrate"].current.status is FactStatus.ACTIVE

    # `location` was read at 0.95 and did not route. See `CONTEXT_SLOTS`.
    assert "location" not in report["routed"]
    assert state.facts["location"].current.provenance in (
        Provenance.STATED, Provenance.CARRIED)

    # What the photograph could not settle is carried, not silently dropped.
    assert any("internal or external" in c
               for c in report["cannot_determine_from_image"])

    # -- turn 4: the symptom ----------------------------------------------
    fourth, state = assistant.ask_turn(turn(
        "There are also patchy areas after drying. What could be causing "
        "that?", 4), state)

    # The wall survives a question about it.
    assert state.active()["substrate"] == "brick"
    assert state.active()["location"] == "internal"

    # And no cause is invented -- from the corpus, which does not discuss
    # patchy drying, or from the photograph, which showed patchy staining and
    # therefore makes this the exact turn where a model would be tempted.
    answer = text_of(fourth)
    for invented in ("rising damp", "penetrating damp", "salt attack",
                     "because the wall", "caused by"):
        assert invented not in answer, (
            f"a cause was stated from evidence that does not support it: "
            f"{invented!r}")


def test_the_photograph_does_not_move_the_recommendation_on_its_own(assistant):
    """Turn three must not change the guidance because a model looked at pixels.

    The scenario's own wording invites it -- "does anything here change the
    guidance?" -- and the honest answer, when the photograph agrees with what
    was already said, is no.
    """
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.94),
                    observation("exposed_masonry", "exposed", 0.9))

    before, state = assistant.ask_turn(turn(
        "I have an internal solid brick wall. Would Ultra suit it?", 1), state)
    after, state = assistant.ask_turn(turn(
        "This is the wall. Does anything change?", 2, images=["wall.jpg"]),
        state, vision=camera)

    assert [p for p in before.paths] == [p for p in after.paths], (
        "the path changed although the photograph only confirmed what was said")
    assert state.facts["substrate"].current.value == "brick"
    assert not state.facts["substrate"].contradicted_by, (
        "agreement was recorded as a contradiction")


# =====================================================================
#  Image-assisted selection, and the minimum question
# =====================================================================

def test_an_image_only_selection_asks_for_the_one_thing_it_cannot_see(assistant):
    """A photograph plus "what should I use here?" -- the demo's hardest turn.

    The image settles the substrate. It does not settle whether the wall is
    inside or outside, and for an insulation job that decides the product. So
    the assistant asks, and asks for *that* -- not for the substrate it can
    already see, and not for a list.
    """
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.93),
                    cannot=("whether the wall is internal or external",))

    reply, state = assistant.ask_turn(turn(
        "What product should I use here to insulate this?", 1,
        images=["wall.jpg"]), state, vision=camera)

    asked = text_of(reply)
    assert "inside or an outside wall" in asked, (
        f"the missing fact was not asked for; got: {asked[:300]}")
    assert "built of underneath" not in asked, (
        "it asked for the substrate the photograph had already shown it")

    # And it did not quietly recommend something while asking.
    assert "ultra" not in asked or "suitable" not in asked


def test_the_answer_to_that_question_completes_the_selection(assistant):
    """The ask-back resumes the original question rather than answering "inside"."""
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.93))

    _paused, state = assistant.ask_turn(turn(
        "What product should I use here to insulate this?", 1,
        images=["wall.jpg"]), state, vision=camera)
    resumed, state = assistant.ask_turn(turn("inside", 2), state)

    answered = resumed.parts[0][1].diagnostics.get("resumed_question", "")
    assert "what product should i use" in answered.lower(), (
        f"the resumed turn answered the word typed, not the question asked: "
        f"{answered!r}")
    assert state.active()["substrate"] == "brick", (
        "the substrate the photograph established was lost across the pause")


def test_a_photograph_that_shows_nothing_usable_asks_for_both(assistant):
    """The poor-light case: perception fails, and the flow is unchanged.

    Everything below the top band leaves its slot uncued, which is exactly
    where the caller would have been having sent no photograph at all.
    """
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.41),
                    observation("texture", "rough", 0.38),
                    cannot=("the substrate", "whether the wall is internal "
                            "or external"))

    reply, _state = assistant.ask_turn(turn(
        "What should I use on this wall?", 1, images=["dark.jpg"]),
        state, vision=camera)

    report = perception_of(reply)
    assert report["routed"] == {}
    assert certainties(report)["substrate"] == "CANNOT_DETERMINE"
    assert "built of underneath" in text_of(reply), (
        "a photograph that settled nothing was treated as though it had")


# =====================================================================
#  Provenance, and what must not leak
# =====================================================================

def test_a_reading_is_never_printed_as_something_the_customer_said(assistant):
    """The misattribution that matters: a guess laundered into testimony."""
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.9))

    reply, state = assistant.ask_turn(turn(
        "What thickness should I build this up to inside?", 1,
        images=["wall.jpg"]), state, vision=camera)

    facts = {f.slot: f for _q, a in reply.parts for f in a.facts}
    assert facts["substrate"].provenance is Provenance.OBSERVED
    assert "as you said" not in facts["substrate"].sentence
    assert "photograph" in facts["substrate"].sentence
    assert state.facts["substrate"].current.image_ref == "IMG_001", (
        "the reading lost the image it came from")


def test_a_new_wall_does_not_inherit_the_old_walls_photograph(assistant):
    """The sharpest leak, because the image has scrolled away.

    A slot filled from a photograph of one wall, surviving onto a question
    about another, is invisible: nobody can see what it was filled from.
    """
    state = ConversationState()
    camera = Camera(observation("substrate", "stonework", 0.95),
                    observation("exposed_masonry", "exposed", 0.9))

    _first, state = assistant.ask_turn(turn(
        "What should I use on this wall inside?", 1, images=["wall.jpg"]),
        state, vision=camera)
    assert state.active()["substrate"] == "stone"

    _second, state = assistant.ask_turn(turn(
        "Now I have another wall outside. What should I use on that?", 2),
        state)

    assert "substrate" not in state.active(), (
        "the first wall's photographed substrate reached the second wall")
    assert state.observations == (), (
        "observations of one wall survived into a case about another")
    assert state.active().get("location") == "external", (
        "what this turn itself stated was thrown away with the old case")


def test_a_new_conversation_starts_with_no_photograph(assistant):
    """New chat means new chat. The checkpoint is dropped, not filtered."""
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.9))

    _reply, _state = assistant.ask_turn(
        turn("What should I use here inside?", 1, images=["wall.jpg"],
             session="chat-one"), state, vision=camera)

    assistant.forget("chat-one")
    fresh = assistant.conversation_state("chat-one")

    assert fresh.active() == {}
    assert fresh.observations == ()


def test_a_later_turn_with_no_photograph_does_not_report_the_earlier_one(assistant):
    """A checkpointed channel keeps what it holds until something clears it.

    Reporting turn three's reading on turn four would tell somebody the
    assistant had just looked at an image it was not sent -- the same class of
    error as the trace that accumulated across a conversation.
    """
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.9))

    first, state = assistant.ask_turn(turn(
        "What should I use here inside?", 1, images=["wall.jpg"]),
        state, vision=camera)
    assert perception_of(first)

    second, _state = assistant.ask_turn(turn(
        "And what thickness?", 2), state)

    assert not perception_of(second), (
        "a turn with no photograph reported the previous turn's reading")
    assert camera.calls == 1, "perception ran on a turn with no image"


# =====================================================================
#  What a photograph may never decide
# =====================================================================

def test_a_symptom_in_frame_does_not_become_a_cause_in_the_answer(assistant):
    """Decision 16, on the path where a model is most tempted to help."""
    state = ConversationState()
    camera = Camera(observation("symptom", "white crystalline deposits", 0.94),
                    observation("staining", "white deposits", 0.94),
                    observation("substrate", "brickwork", 0.9),
                    observation("exposed_masonry", "exposed", 0.9),
                    cannot=("the moisture source",))

    reply, _state = assistant.ask_turn(turn(
        "Why has my wall gone like this?", 1, images=["salts.jpg"]),
        state, vision=camera)

    answer = text_of(reply)
    assert "rising damp" not in answer
    assert "penetrating damp" not in answer
    # The photograph is acknowledged rather than silently ignored.
    assert perception_of(reply), "the photograph was read but never reported"


def test_a_render_hides_its_background_and_the_substrate_is_refused(assistant):
    """The substrate gate, through the whole stack rather than in the resolver.

    A model that reports a covering finish and no exposed masonry has said, in
    its own two readings, that the background is concealed. A substrate claim
    then contradicts it -- and the conversation must ask rather than proceed on
    a wall nobody has seen.
    """
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("existing_finish", "rendered", 0.93),
                    observation("damaged_finish", "sound", 0.9),
                    cannot=("what is behind the render",))

    reply, state = assistant.ask_turn(turn(
        "What should I use to insulate this wall inside?", 1,
        images=["rendered.jpg"]), state, vision=camera)

    report = perception_of(reply)
    assert report["routed"] == {}, (
        "a substrate was claimed through a sound render")
    assert "substrate" not in state.active()

    substrate = next(o for o in report["observations"]
                     if o["attribute"] == "substrate")
    assert substrate["certainty"] == "LIKELY"
    assert "covers the background" in substrate["withheld"], (
        "the reading was dropped without saying why")
    assert "built of underneath" in text_of(reply)


def test_a_truncated_reading_is_reported_and_never_acted_on(assistant):
    """A looping model's observations are not evidence, and say so."""
    state = ConversationState()
    camera = Camera(observation("substrate", "brickwork", 0.95),
                    observation("exposed_masonry", "exposed", 0.95),
                    truncated=True)

    reply, state = assistant.ask_turn(turn(
        "What should I use on this inside?", 1, images=["wall.jpg"]),
        state, vision=camera)

    report = perception_of(reply)
    assert report["truncated"] is True
    assert report["routed"] == {}
    assert "substrate" not in state.active()
    assert certainties(report)["substrate"] == "LIKELY"


def test_perception_failing_leaves_the_conversation_where_it_was(assistant):
    """An unreachable model reduces coverage and changes nothing else."""
    class Broken:
        calls = 0

        def observe(self, images):
            Broken.calls += 1
            return vision.resolve([vision.Perception(error="connection refused")])

    state = ConversationState()
    reply, state = assistant.ask_turn(turn(
        "I have an internal brick wall. What thickness of Ultra?", 1,
        images=["wall.jpg"]), state, vision=Broken())

    assert Broken.calls == 1
    assert state.active()["substrate"] == "brick", "the typed facts were lost"
    assert perception_of(reply)["observations"] == []
