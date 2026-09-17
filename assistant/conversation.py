"""Conversation state as typed, provenance-bearing facts.

`assistant/session.py` carries four slots as bare strings. That was enough while
the only question was "what is this wall made of", and it stopped being enough
the moment three things became true at once: a photograph can now propose a
value, a later turn can correct an earlier one, and a recommendation has to be
able to say *why* it believed something. A bare string answers none of those. It
cannot distinguish a substrate the caller typed from one a vision model guessed,
it cannot represent "the person said brick and the photograph says stone", and
it has no room for the turn a value came from.

So a slot value becomes a `SessionFact`, and the merge that folds a new turn into
an old state becomes `merge_facts` — a pure function with three rules, each of
which exists because getting it wrong is a specific, nameable failure:

**A current-turn statement supersedes everything.** "Actually the wall is stone"
must replace brick, not accumulate beside it. The superseded fact is retained in
the slot's history rather than deleted, because an answer given while brick was
believed still has to be explicable — the same argument decision 18 makes for
document versions.

**A vision observation never overwrites something a person said.** This is the
rule the whole seam rests on. A model reading pixels is not a witness; promoting
its reading over a typed sentence launders an inference into testimony. When the
two disagree the *person's* fact stays current and is marked `CONFLICTING`, and
the observation is retained alongside it. A conflicting slot then reads as
missing to the requirement gate, so the system asks rather than picks a winner.

**An inherited fact never overwrites a current one.** Merge order is history
under the present, which is what `assistant/router.py` already does with
`carried` and what makes self-correction work at all.

**What this module deliberately does not do.** It never parses an assistant
answer. Nothing here reads generated prose, and no constructor takes one: a
`SessionFact` can only be built from something the caller wrote, something a
`VisionProvider` observed, or something deterministic code derived. Previous
assistant output is not a source of truth, and the way to keep it from becoming
one is to give it nowhere to enter.

`Provenance` is imported from `assistant/answer.py` rather than redefined. It
already carries exactly the four states this needs, with matching semantics —
``STATED`` is the current user, ``CARRIED`` is the same person in an earlier
turn, ``OBSERVED`` is a photograph, ``ASSUMED`` is the system choosing — and the
renderer's phrase table is keyed on it. A parallel vocabulary would mean two
enums to keep in step and a translation layer between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from .answer import Provenance

# How much a single turn may add. A conversation is a handful of facts about one
# building; anything past this is either an attack or a bug, and both are better
# refused than absorbed.
MAX_FACTS_PER_TURN = 24

# Slots whose value decides a recommendation and which therefore may be filled
# from a photograph. `product` is deliberately absent, here as in
# `vision.VISION_SLOTS`: a model that could name a product from an image would
# be turning a guess into a commercial recommendation, which is the one thing
# decision 16.1 forbids outright.
OBSERVABLE_SLOTS = frozenset({
    "substrate", "location", "exposure", "symptom", "existing_finish",
    "defects", "staining", "cracks", "texture", "exposed_masonry",
    "previous_render", "moisture_evidence",
})


class FactStatus(str, Enum):
    """Whether a fact is believed, replaced, or disputed.

    ``CONFLICTING`` is the member that earns this enum. Without it a
    disagreement between what somebody typed and what a photograph shows has
    only two resolutions, and both are wrong: silently keep the typed value and
    the image was pointless, or silently take the image and the system has
    overruled a person on the strength of an uncalibrated score. A third state
    lets the requirement gate treat the slot as unsettled and ask, which is what
    an advisor would do.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class SessionFact:
    """One slot value, where it came from, and when.

    Frozen because a fact is a record of something that happened. Correcting a
    wall means adding a newer fact and superseding this one, not editing it —
    editing would destroy the history that makes an earlier answer explicable.
    """

    slot: str
    value: str
    provenance: Provenance
    source_turn: int = 0
    status: FactStatus = FactStatus.ACTIVE
    # Only ever set for ``OBSERVED`` facts, and never compared against a
    # threshold outside `assistant/vision.py`. Decision 16.1: a vision model
    # reporting 0.91 is not right 91% of the time, so this travels for the
    # audit trail and the band decision is made where the calibration argument
    # lives.
    confidence: float = 1.0
    # Which image produced it, for an ``OBSERVED`` fact. The audit equivalent of
    # a citation: it names the evidence a person can check with their own eyes.
    image_ref: str = ""

    @property
    def from_person(self) -> bool:
        """Did a human assert this, in any turn? A photograph answers no."""
        return self.provenance in (Provenance.STATED, Provenance.CARRIED)

    def inherited(self) -> "SessionFact":
        """The same fact, seen from a later turn.

        ``STATED`` becomes ``CARRIED`` — still the person's own words, no longer
        in the sentence being answered. Everything else keeps its provenance,
        because an observation does not become testimony by ageing.
        """
        if self.provenance is Provenance.STATED:
            return replace(self, provenance=Provenance.CARRIED)
        return self


@dataclass(frozen=True)
class FactHistory:
    """Everything known about one slot: what is believed, and what was.

    One channel rather than two. The alternative — a `facts` dict beside a
    separate `superseded` list — puts the supersession rule in whichever node
    happens to write both, and a node that forgot the second write would drop
    the audit trail silently. Holding the history against the slot makes the
    rule unbypassable: there is no way to replace a value except through
    `merge_facts`, and no way through it that discards what it replaced.
    """

    current: SessionFact
    superseded: tuple[SessionFact, ...] = ()
    # Observations that disagree with `current`. Retained rather than discarded,
    # because "the photograph suggests stone" is exactly what the clarifying
    # question needs to quote back.
    contradicted_by: tuple[SessionFact, ...] = ()

    def __post_init__(self) -> None:
        """Normalise the sequences, because a checkpoint does not preserve type.

        JSON has one sequence and Python has two, so a tuple written into a
        checkpoint comes back as a list. `merge_facts` then does
        `history.superseded + (fact,)` and raises `can only concatenate list
        (not "tuple") to list` -- three turns into a conversation, only for a
        slot that had actually been superseded, and only once the checkpointer
        became the thing that carried state.

        Coercing here rather than at each use keeps the invariant with the type
        that promises it: a `FactHistory` holds tuples, wherever it was built.
        """
        object.__setattr__(self, "superseded", tuple(self.superseded or ()))
        object.__setattr__(self, "contradicted_by",
                           tuple(self.contradicted_by or ()))

    @property
    def unsettled(self) -> bool:
        return self.current.status is FactStatus.CONFLICTING


@dataclass(frozen=True)
class NewCase:
    """A reducer instruction: this turn is about a different wall.

    A plain dictionary cannot say "forget the rest", and it must not be able to
    -- `merge_facts` is the single place supersession is enforced and a node
    that could bypass it by assigning the channel directly would be a hole in
    the guarantee. So starting a case is an explicit value the reducer
    recognises, rather than an assignment the reducer never sees.

    `keep` is what this turn itself stated. It survives because the person said
    it in the sentence that opened the new case: "now I have another wall
    outside" establishes `location=external` *about the new wall*, and throwing
    it away with the old case would mean immediately asking about something they
    just told us.
    """

    keep: dict = field(default_factory=dict)
    reason: str = ""


def merge_facts(old: dict[str, FactHistory],
                new: dict | NewCase) -> dict[str, FactHistory]:
    """Fold this turn's facts into the conversation's. The three rules, in code.

    This is the reducer the state graph runs on the `facts` channel, which means
    every write to conversation state passes through it — a node cannot set a
    slot without the supersession and conflict rules applying. That is the point
    of putting it here rather than in a node: the guarantee is structural rather
    than a convention a later node has to remember.

    **It has to be total over what a channel can hand it**, and finding that out
    was worth the bug. A graph applies its reducer to the *initial* state as well
    as to every node's return, so the first call receives a whole prior
    conversation -- a `dict[str, FactHistory]` -- while every later call receives
    one turn's `dict[str, SessionFact]`. A version assuming the second shape
    wrapped each restored `FactHistory` inside another one, and the corruption
    surfaced three layers away as `'FactHistory' object has no attribute
    'status'`. Accepting either shape is the fix; rejecting one would have put
    the normalisation in the caller, which is the convention-instead-of-structure
    this function exists to avoid.

    Pure and free of I/O, so it is testable without a graph, a model or a store,
    and so the same function serves whichever runner is wiring the nodes.
    """
    if isinstance(new, NewCase):
        # A different wall. Nothing from the old one may inform this one, so the
        # accumulated facts are dropped rather than merged under -- which is the
        # whole bug this instruction exists to fix. The old case is not lost:
        # `ConversationState.open_case` files it in `history` for the transcript
        # and the audit trail before this runs.
        return merge_facts({}, dict(new.keep))

    out = dict(old or {})
    for slot, incoming in list((new or {}).items())[:MAX_FACTS_PER_TURN]:
        # Either shape. A restored history keeps its own past; a bare fact
        # starts one.
        restored = incoming if isinstance(incoming, FactHistory) else None
        fact = restored.current if restored is not None else incoming
        history = out.get(slot)

        if history is None:
            out[slot] = (restored if restored is not None
                         else FactHistory(current=fact))
            continue

        prior = history.current

        # Rule 1 — the person, speaking now, wins over anything.
        if fact.provenance is Provenance.STATED:
            if prior.value == fact.value and prior.from_person:
                # Restated, not corrected. Refresh the turn, keep the history.
                out[slot] = replace(history, current=fact)
            else:
                out[slot] = FactHistory(
                    current=fact,
                    superseded=history.superseded + (
                        replace(prior, status=FactStatus.SUPERSEDED),),
                    # A correction settles the slot, so a previous disagreement
                    # is no longer live.
                    contradicted_by=(),
                )
            continue

        # Rule 2 — a photograph may inform, never overrule.
        if fact.provenance is Provenance.OBSERVED:
            if not prior.from_person:
                out[slot] = replace(history, current=fact)
            elif prior.value == fact.value:
                # Agreement. The person's fact stands and is now corroborated;
                # promoting it to OBSERVED would *lose* information.
                out[slot] = replace(
                    history, contradicted_by=history.contradicted_by)
            elif prior.source_turn == fact.source_turn:
                # Said and photographed in the same breath: "here is a picture,
                # it's brick". They are correcting the image, not disagreeing
                # with themselves, and asking which they meant would be
                # ignoring what they just took the trouble to type. The person
                # wins outright and the reading is kept beside it.
                out[slot] = replace(history,
                                    contradicted_by=history.contradicted_by + (fact,))
            else:
                # An older statement against a new photograph. Here the
                # disagreement is real -- the wall may have been described from
                # memory, or the picture may be of a different part of it -- and
                # neither source can be preferred without asking.
                out[slot] = replace(
                    history,
                    current=replace(prior, status=FactStatus.CONFLICTING),
                    contradicted_by=history.contradicted_by + (fact,),
                )
            continue

        # Rule 3 — history fills a gap, never overwrites the present.
        if prior.status is FactStatus.ACTIVE and prior.from_person:
            continue
        out[slot] = replace(history, current=fact)

    return out


def merge_observations(old, new):
    """Accumulate what images showed -- and drop it all when the case changes.

    The `facts` channel and this one both have to understand `NewCase`, and
    finding that out cost a failing test: `case_boundary` returned an empty list
    for the observations, an append reducer appended nothing, and the previous
    wall's observation survived a boundary its facts had not. A photograph of
    one wall filling a slot on another is the sharpest form of the leak, because
    the image has scrolled away and nobody can see what the slot was filled
    from.

    So the reset is expressed the same way in both places rather than being
    something this reducer has to be told separately.
    """
    if isinstance(new, NewCase):
        return []
    return list(old or []) + list(new or [])


@dataclass
class Case:
    """One technical subject: a wall, a job, an elevation.

    A conversation is not one case. Somebody finishes asking about the internal
    brick wall and says "now I have another wall outside", and everything
    established about the first wall is now actively misleading about the
    second. Carrying it is not a small inaccuracy -- it produces a confident,
    cited recommendation for a wall that does not exist, printed as "brick, as
    you told me earlier", which attributes the invention to the customer.

    Cases are retained rather than discarded when a new one opens, because the
    transcript still shows the earlier answers and they still have to be
    explicable. What a retired case may not do is be *read* -- routing and
    retrieval see only the current one.
    """

    case_id: str
    facts: dict[str, FactHistory] = field(default_factory=dict)
    observations: tuple[SessionFact, ...] = ()
    pending: str = ""

    def __post_init__(self) -> None:
        # Same reason as `FactHistory.__post_init__`: a checkpoint round-trip
        # turns tuples into lists.
        self.observations = tuple(self.observations or ())
    opened_at_turn: int = 0
    # The words that opened it, for the audit trail. A case boundary is a
    # judgement, and a judgement nobody can inspect is a judgement nobody can
    # correct.
    opened_because: str = ""


# Words that introduce a *different* subject. Deliberately short, deliberately
# paired with a subject noun below, and deliberately not the whole mechanism:
# `understanding.TurnUnderstanding.new_subject` carries the model's reading of
# the same question, and either is enough.
#
# **Both directions of error are not equal, and the asymmetry decides the
# design.** A false positive opens a case that did not need opening: the person
# is asked for the substrate again, which costs them one reply. A false negative
# carries brick onto a stone wall and recommends a product for it. So this errs
# towards opening, and the model's signal is allowed to open a case on its own.
_INTRODUCERS = (
    "another", "a second", "second", "a different", "different", "other",
    "next", "separate", "new", "also have", "also got", "as well as",
)

# What is being introduced. Without this, "a different question" and "another
# thing" would open cases, and a case boundary on every conversational filler is
# its own kind of broken.
_SUBJECTS = (
    "wall", "walls", "room", "rooms", "job", "project", "property", "building",
    "house", "elevation", "ceiling", "floor", "gable", "chimney", "extension",
    "surface", "area",
)

# A correction is the opposite of a new subject: it is the *same* wall, described
# better. "Actually it is stone" must supersede brick on the current case, never
# open a second one -- getting that backwards would lose the correction into a
# fresh case and keep answering from the value the person just corrected.
_CORRECTIONS = (
    "actually", "sorry", "i meant", "i mean", "correction", "my mistake",
    "to correct", "rather than", "not brick", "not stone", "instead it",
)


def opens_a_new_case(question: str, model_says: bool = False) -> tuple[bool, str]:
    """Does this turn introduce a different wall? Returns (yes, why).

    Deterministic signals first, the model's reading second, and a correction
    marker vetoing both. The veto is checked first because "actually, I have
    another wall" is vanishingly rare next to "actually it is stone", and
    treating a correction as a new case is the worse of the two mistakes.
    """
    lowered = f" {question.lower().strip()} "

    # An explicit introducer paired with a subject noun is the strongest signal
    # available and is honoured even alongside a correction marker. "Actually, I
    # have another wall" is both an apology and a new wall, and reading only the
    # apology loses the wall.
    for introducer in _INTRODUCERS:
        if f" {introducer} " not in lowered:
            continue
        for subject in _SUBJECTS:
            if f" {subject} " in lowered or f" {subject}." in lowered:
                return True, f"{introducer!r} + {subject!r}"

    # The veto applies to the weaker signal only. A model reading "actually it is
    # stone" as a new subject is making the specific mistake that would lose a
    # correction into a fresh case and go on answering from the value the person
    # had just corrected -- so on this one the deterministic marker outranks it.
    correction = next((m for m in _CORRECTIONS if m in lowered), "")
    if correction:
        return False, f"correction marker {correction!r}: same wall, restated"

    if model_says:
        return True, "the understanding stage read this as a new subject"
    return False, ""


@dataclass(frozen=True)
class TurnInput:
    """One message, exactly as it arrived. Never merged with anything.

    The separation this type exists to enforce is the one
    `assistant/engine.py` used to break: `question = context + "\\n\\n" + question`
    put an earlier answer's prose in front of the policy gate, the slot
    detector and the embedder, so a previous mention of "cost" routed an
    unrelated question to the price referral and a previous mention of plaster
    put a substrate on a question that named none.

    `raw_question` is what the person typed and nothing else. The transcript is
    shown back to them by the page and reaches generation as a separate
    argument, after routing and retrieval have already been decided.
    """

    raw_question: str
    turn_index: int = 0
    images: tuple = ()
    audiences: tuple[str, ...] = ("public",)
    session_id: str = ""
    # The reply to an ask-back, when this turn is answering one.
    resumes: str = ""
    # What the persisted trace calls this turn. Defaults to the index, but a
    # driver that labels its conversations -- the evaluation harness does --
    # passes its own, so a recorded trace can be read back as that scenario
    # rather than as turn "2" of something unnamed.
    turn_id: str = ""
    # The id the surface already gave this request. A web response carries it in
    # `X-Correlation-Id`, so it has to be the id the trace is written under or
    # the header points at nothing -- which is the whole of "paste this id and
    # read what happened".
    correlation_id: str = ""


class ConversationState:
    """What a conversation knows, organised by the case it is about.

    The container changed shape when cross-case contamination was measured. It
    used to hold one flat set of facts for the whole conversation, and that is
    correct only while a conversation is about one wall. It very often is not:
    somebody finishes with the internal brick wall and says "now I have another
    wall outside", and every fact about the first wall is now actively
    misleading about the second.

    So the facts belong to a `Case`, and a conversation holds one current case
    and a history of retired ones. The history exists for the same reason
    decision 18 retains superseded document versions: the earlier answers are
    still in the transcript and still have to be explicable. What a retired case
    may not do is be read -- `facts`, `active()` and everything routing and
    retrieval consume see the current case only.

    `facts`, `observations` and `pending` remain readable and writable as though
    they were still fields, because dozens of call sites and tests use them that
    way and a rename would have been churn rather than a change.
    """

    def __init__(self, facts: dict | None = None,
                 observations: tuple = (), pending: str = "",
                 current: "Case | None" = None,
                 history: tuple = (), turn_index: int = 0) -> None:
        """Written out rather than generated, so the old keywords keep working.

        `facts`, `observations` and `pending` became properties of the current
        case, and a `@dataclass` cannot have a field and a property of the same
        name. Every caller and test that says `ConversationState(facts=...)`
        would have had to change -- churn that would have obscured the actual
        change, which is that a conversation now has cases at all.
        """
        self.current = current or Case(case_id="case-1")
        if facts:
            self.current.facts = dict(facts)
        if observations:
            self.current.observations = tuple(observations)
        if pending:
            self.current.pending = pending
        self.history: tuple[Case, ...] = tuple(history)
        self.turn_index = turn_index

    def __repr__(self) -> str:
        return (f"ConversationState(case={self.current.case_id!r}, "
                f"facts={sorted(self.current.facts)}, "
                f"retired={len(self.history)}, turn={self.turn_index})")

    # -- the old field names, delegating to the current case ----------------

    @property
    def facts(self) -> dict[str, FactHistory]:
        return self.current.facts

    @facts.setter
    def facts(self, value: dict) -> None:
        self.current.facts = dict(value or {})

    @property
    def observations(self) -> tuple[SessionFact, ...]:
        return self.current.observations

    @observations.setter
    def observations(self, value) -> None:
        self.current.observations = tuple(value or ())

    @property
    def pending(self) -> str:
        return self.current.pending

    @pending.setter
    def pending(self, value: str) -> None:
        self.current.pending = value or ""

    @property
    def case_id(self) -> str:
        return self.current.case_id

    # -- reading -------------------------------------------------------------

    def active(self) -> dict[str, str]:
        """Slots a later turn may inherit, as bare strings.

        The shape `assistant/session.py` has always returned and every existing
        caller expects, so adding provenance and cases underneath costs no call
        site. Two exclusions: a conflicting slot, because the requirement gate
        must see it as unsettled and a caller reading only strings cannot be
        told the difference any other way; and every retired case, because that
        is the contamination this structure exists to prevent.
        """
        return {slot: h.current.value
                for slot, h in self.current.facts.items()
                if h.current.status is FactStatus.ACTIVE}

    def unsettled(self) -> list[str]:
        """Slots where a photograph and a person disagree."""
        return sorted(slot for slot, h in self.current.facts.items()
                      if h.unsettled)

    def provenance_of(self) -> dict[str, Provenance]:
        """The origin of each active slot, for `Decision.origins`."""
        return {slot: h.current.provenance
                for slot, h in self.current.facts.items()
                if h.current.status is FactStatus.ACTIVE}

    def inherit(self) -> dict[str, FactHistory]:
        """This case as the next turn sees it: stated becomes carried."""
        return {slot: replace(h, current=h.current.inherited())
                for slot, h in self.current.facts.items()}

    # -- writing -------------------------------------------------------------

    def open_case(self, turn_index: int, because: str,
                  keep: dict | None = None) -> Case:
        """Retire the current case and start a new one.

        `keep` is what *this* turn stated, and it carries over because the
        person said it in the sentence that opened the case: "now I have another
        wall outside" establishes `location=external` about the new wall.
        Dropping it would mean asking immediately about something they had just
        told us, which is how a safety mechanism becomes an annoyance and then
        gets removed.

        Everything else is left behind. Not deleted -- the retired case is
        appended to `history`, where the transcript and an audit can still reach
        it, and where routing and retrieval cannot.
        """
        retired = self.current
        self.history = self.history + (retired,)
        self.current = Case(
            case_id=f"case-{len(self.history) + 1}",
            facts=merge_facts({}, dict(keep or {})),
            opened_at_turn=turn_index,
            opened_because=because,
        )
        return self.current

    def supersede_observations(self) -> None:
        """Forget what images showed, without retiring the case.

        Weaker than `open_case` and kept for the narrower situation it suits: the
        subject has not changed, but the photographs no longer apply -- a new
        upload replacing an old one, or an explicit "ignore that picture".

        Facts a person stated survive, because a person can be asked again; an
        observation cannot be re-checked once the image has gone, and carrying it
        is how one wall's photograph fills a slot on another.
        """
        self.current.observations = ()
        for slot, history in list(self.current.facts.items()):
            if history.current.provenance is Provenance.OBSERVED:
                del self.current.facts[slot]
            elif history.contradicted_by:
                self.current.facts[slot] = replace(
                    history,
                    current=replace(history.current, status=FactStatus.ACTIVE),
                    contradicted_by=())
