"""The merge rules: who wins, what is kept, and what stays unsettled.

`assistant/conversation.py` has one interesting function and everything else is
containers. `merge_facts` decides what a conversation believes after a turn, and
each of its three rules exists because the alternative is a specific, nameable
harm:

* a correction that accumulates beside the thing it corrects answers about a
  wall that does not exist;
* a photograph that overwrites a typed sentence attributes a model's guess to
  the person who is going to act on the answer;
* an inherited value that overwrites the present one makes self-correction
  impossible.

So the rules are tested as behaviour rather than as a merge, and the audit trail
each one leaves is tested too — a rule that produced the right current value and
threw away what it replaced would pass a weaker suite than this one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.answering.answer import (  # noqa: E402
    Provenance,
)
from assistant.conversation import (                               # noqa: E402
    MAX_FACTS_PER_TURN, OBSERVABLE_SLOTS, ConversationState, FactHistory,
    FactStatus, SessionFact, TurnInput, merge_facts,
)


def stated(slot, value, turn=1):
    return SessionFact(slot, value, Provenance.STATED, turn)


def carried(slot, value, turn=1):
    return SessionFact(slot, value, Provenance.CARRIED, turn)


def observed(slot, value, turn=1, confidence=0.9, image="IMG_001"):
    return SessionFact(slot, value, Provenance.OBSERVED, turn,
                       confidence=confidence, image_ref=image)


def merged(*turns):
    """Apply several turns in order, as a conversation would."""
    state: dict[str, FactHistory] = {}
    for facts in turns:
        state = merge_facts(state, facts)
    return state


# ------------------------------------------------- rule 1: the person, now


def test_a_correction_supersedes_rather_than_accumulating():
    """"Actually the wall is stone" is the scenario this module exists for."""
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": stated("substrate", "stone", 4)})

    assert state["substrate"].current.value == "stone"
    assert state["substrate"].current.status is FactStatus.ACTIVE


def test_the_superseded_value_is_retained_for_audit():
    """An answer given while brick was believed still has to be explicable.

    The same argument decision 18 makes for document versions: a value that is
    no longer served is not a value that was never true.
    """
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": stated("substrate", "stone", 4)})

    history = state["substrate"]
    assert [f.value for f in history.superseded] == ["brick"]
    assert history.superseded[0].status is FactStatus.SUPERSEDED
    assert history.superseded[0].source_turn == 1, "the turn that said it"


def test_restating_the_same_value_does_not_manufacture_history():
    """Saying "brick" twice is one fact, not a correction of itself."""
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": stated("substrate", "brick", 3)})

    assert state["substrate"].superseded == ()
    assert state["substrate"].current.source_turn == 3, "the turn is refreshed"


def test_a_correction_settles_a_slot_a_photograph_had_disputed():
    """Once the person says stone, the photograph saying stone is not a dispute."""
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": observed("substrate", "stone", 3)},
                   {"substrate": stated("substrate", "stone", 4)})

    assert state["substrate"].current.status is FactStatus.ACTIVE
    assert state["substrate"].contradicted_by == ()


# ------------------------------------- rule 2: a photograph informs, never rules


def test_a_photograph_does_not_overwrite_what_a_person_said():
    """The rule the whole vision seam rests on.

    A model reading pixels is not a witness. Promoting its reading over a typed
    sentence would launder an inference into testimony, and the person would
    then be shown "brick, as you told me" about a wall they never described.
    """
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": observed("substrate", "stone", 3)})

    assert state["substrate"].current.value == "brick"
    assert state["substrate"].current.provenance is Provenance.STATED


def test_a_disagreeing_photograph_leaves_the_slot_unsettled():
    """Not silently resolved either way: unsettled, so the gate can ask."""
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": observed("substrate", "stone", 3)})

    assert state["substrate"].current.status is FactStatus.CONFLICTING
    assert state["substrate"].unsettled


def test_the_disagreeing_observation_is_retained_not_discarded():
    """It is what the clarifying question has to quote back."""
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": observed("substrate", "stone", 3,
                                          image="IMG_007")})

    disputed = state["substrate"].contradicted_by
    assert [f.value for f in disputed] == ["stone"]
    assert disputed[0].image_ref == "IMG_007", "a person can check the evidence"


def test_a_photograph_fills_a_slot_nobody_has_spoken_for():
    """Informing is the whole point; only overruling is forbidden."""
    state = merged({"substrate": observed("substrate", "brick", 2)})

    assert state["substrate"].current.value == "brick"
    assert state["substrate"].current.provenance is Provenance.OBSERVED


def test_a_photograph_that_agrees_does_not_downgrade_the_persons_fact():
    """Corroboration must not rewrite testimony as observation.

    If agreement promoted the slot to OBSERVED, a later answer would tell
    somebody their own stated substrate came "from the photograph you sent",
    which is both wrong and an invitation to correct something never in doubt.
    """
    state = merged({"substrate": stated("substrate", "brick", 1)},
                   {"substrate": observed("substrate", "brick", 3)})

    assert state["substrate"].current.provenance is Provenance.STATED
    assert state["substrate"].current.status is FactStatus.ACTIVE


def test_a_later_photograph_may_replace_an_earlier_one():
    """Two images, no person: the newer reading is simply better evidence."""
    state = merged({"substrate": observed("substrate", "brick", 2)},
                   {"substrate": observed("substrate", "stone", 3)})

    assert state["substrate"].current.value == "stone"


# ----------------------------------------- rule 3: history under the present


def test_an_inherited_fact_does_not_overwrite_the_present():
    state = merged({"substrate": stated("substrate", "stone", 4)},
                   {"substrate": carried("substrate", "brick", 1)})

    assert state["substrate"].current.value == "stone"


def test_an_inherited_fact_fills_a_gap():
    state = merged({"product": carried("product", "Ultra", 1)})

    assert state["product"].current.value == "Ultra"


def test_inheriting_turns_stated_into_carried_and_leaves_observed_alone():
    """"As you said" is only true in the turn they said it.

    An observation does not become testimony by ageing, so it keeps its
    provenance and keeps being printed as something the photograph showed.
    """
    state = ConversationState(facts=merged(
        {"substrate": stated("substrate", "brick", 1),
         "location": observed("location", "internal", 1)}))

    nxt = state.inherit()
    assert nxt["substrate"].current.provenance is Provenance.CARRIED
    assert nxt["location"].current.provenance is Provenance.OBSERVED


# ------------------------------------------------------------ the state object


def test_active_excludes_a_conflicting_slot():
    """A caller reading bare strings cannot be told "disputed" any other way.

    So it is not told "brick" either. The requirement gate sees the slot as
    absent, asks, and the person settles it.
    """
    state = ConversationState(facts=merged(
        {"substrate": stated("substrate", "brick", 1),
         "location": stated("location", "internal", 1)},
        {"substrate": observed("substrate", "stone", 3)}))

    assert state.active() == {"location": "internal"}
    assert state.unsettled() == ["substrate"]


def test_active_is_the_shape_every_existing_caller_expects():
    """`session.carried()` has always returned dict[str, str]."""
    state = ConversationState(facts=merged(
        {"substrate": stated("substrate", "brick", 1)}))

    assert state.active() == {"substrate": "brick"}
    assert all(isinstance(v, str) for v in state.active().values())


def test_provenance_is_reported_per_slot_for_the_answer_to_print():
    state = ConversationState(facts=merged(
        {"substrate": stated("substrate", "brick", 1),
         "location": observed("location", "internal", 1)}))

    assert state.provenance_of() == {"substrate": Provenance.STATED,
                                     "location": Provenance.OBSERVED}


# ------------------------------------------------------------- topic reset


def test_a_change_of_subject_forgets_what_the_photograph_showed():
    """One wall's image must not fill a slot on a different case."""
    state = ConversationState(facts=merged(
        {"substrate": observed("substrate", "brick", 1)}))
    state.observations = (observed("substrate", "brick", 1),)

    state.supersede_observations()

    assert "substrate" not in state.facts
    assert state.observations == ()


def test_a_change_of_subject_keeps_what_a_person_said():
    """A person can be asked again; an image that has scrolled away cannot."""
    state = ConversationState(facts=merged(
        {"substrate": stated("substrate", "brick", 1)}))

    state.supersede_observations()

    assert state.active() == {"substrate": "brick"}


def test_a_topic_change_resettles_a_slot_the_gone_photograph_disputed():
    """The dispute leaves with the evidence for it."""
    state = ConversationState(facts=merged(
        {"substrate": stated("substrate", "brick", 1)},
        {"substrate": observed("substrate", "stone", 3)}))
    assert state.unsettled() == ["substrate"]

    state.supersede_observations()

    assert state.unsettled() == []
    assert state.active() == {"substrate": "brick"}


# -------------------------------------------------------------- the bounds


def test_a_turn_cannot_write_unbounded_facts():
    """A conversation is a handful of facts about one building."""
    flood = {f"slot{i}": stated(f"slot{i}", "x", 1)
             for i in range(MAX_FACTS_PER_TURN * 3)}

    assert len(merge_facts({}, flood)) == MAX_FACTS_PER_TURN


# --------------------------------------------------- what cannot be observed


def test_a_product_can_never_come_from_a_photograph():
    """Decision 16.1's one absolute rule, pinned at the state layer too.

    `vision.VISION_SLOTS` already refuses to emit one, and this is the second
    lock: even a caller that hand-built an OBSERVED product fact would be
    writing a slot this set does not sanction, and the SELECT gate reads this
    set rather than trusting its input.
    """
    assert "product" not in OBSERVABLE_SLOTS


def test_the_observable_slots_cover_what_16_1_asks_a_vlm_to_report():
    expected = {"substrate", "existing_finish", "defects", "staining",
                "cracks", "texture", "exposed_masonry", "previous_render"}

    assert expected <= OBSERVABLE_SLOTS


# ------------------------------------------------------------- the raw turn


def test_a_turn_input_holds_the_question_and_nothing_prepended():
    """The type exists to make concatenation impossible to do by accident."""
    turn = TurnInput(raw_question="What plaster should I use?", turn_index=2)

    assert turn.raw_question == "What plaster should I use?"
    assert "\n\n" not in turn.raw_question
