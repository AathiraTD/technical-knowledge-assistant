"""The turn as a state machine: nodes around the functions that already exist.

Every node in this file is a thin adapter. `understand_turn` calls
`assistant/understanding.py`; `retrieve_candidates` calls the existing
`Retriever`; `assess_evidence` calls `assistant/candidates.py`; the printing
nodes call the existing `AnswerEngine`. **No domain truth lives here.** The
repository, retrieval, the six checks, the policy gate, the calculations and the
audience filter are unchanged and are called, not absorbed — which is the
condition under which adopting an orchestration library is safe at all.

What the graph adds is the thing `assistant/session.py` plus `assistant/ui.py`
were doing informally and getting wrong: an explicit order, a single place where
conversation state is merged, and a pause that can be resumed. The bugs that
motivated it are all state-machine bugs — transcript contaminating routing,
assistant output becoming a slot, an ask-back value used once and dropped, state
surviving invisibly across a reload.

**Two deliberate configurations, both of which are the reason this import is
acceptable in a system whose decision 4 says "no LangChain".**

*Telemetry is off, in code.* `langsmith` is a hard dependency of
`langchain-core`, which is a hard dependency of `langgraph`, and it is a client
for a hosted tracing service. A single environment variable would otherwise
export whole conversations to a third party, which this system's privacy posture
forbids. The variables are set here, before the import, rather than documented
in a runbook — a posture that depends on nobody setting an env var is not a
posture. `assistant/observability.py` remains the only telemetry path.

*Serialisation is explicit.* The checkpointer refuses to deserialise unknown
types in a future version, and defaulting to permissive would mean a checkpoint
could rehydrate arbitrary classes. The domain types are named.
"""

from __future__ import annotations

import os

# Every environment variable that can turn hosted tracing on. `langsmith`
# resolves a name like "TRACING_V2" against both the LANGSMITH_ and LANGCHAIN_
# prefixes, so both spellings of each have to be closed.
_TRACING_VARS = (
    "LANGSMITH_TRACING", "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2",
    "LANGSMITH_OTEL_ENABLED", "LANGCHAIN_OTEL_ENABLED",
)


def _disable_hosted_tracing() -> None:
    """Make hosted tracing unreachable, whatever the parent environment says.

    This used to be `os.environ.setdefault(...)`, and the reasoning behind that
    -- do not override an operator who has deliberately chosen otherwise -- was
    wrong for this variable in this system. It left a measured hole:
    `LANGCHAIN_TRACING_V2=true` with credentials present re-enabled outbound
    export of conversation state, because `setdefault` by definition does not
    override. A CI image, a shell profile or a colleague's `.env` was enough.

    `langsmith.utils.tracing_is_enabled` resolves in this order:

        1. the `tracing_context` context variable
        2. whether a run tree is already open
        3. `langsmith._internal._context._GLOBAL_TRACING_ENABLED`
        4. the environment

    So closing the environment alone closes the weakest of the four. Two layers
    are shut here:

    **Assignment, not `setdefault`, on every spelling** -- closes (4), and must
    happen before `langchain_core` is imported, because it reads these at import
    time to decide whether to install its hooks.

    **The process-global fallback** -- closes (3), which outranks the
    environment entirely, so it holds even if something later rewrites a
    variable. It is a private name and is therefore set defensively and never
    depended on alone; the guarantee this function actually offers is asserted
    afterwards by `tracing_disabled()`, on the outcome rather than on the
    mechanism.

    The context variable (1) is deliberately *not* used. It is per-context, and
    the web surface is a `ThreadingHTTPServer` -- a value set on the importing
    thread would not reach the worker threads that answer requests, which is
    exactly the kind of protection that looks present and is absent.

    Credentials are left alone. Deleting `LANGSMITH_API_KEY` from the process
    would be tidier and is not this module's decision to take; the test proves
    tracing stays off *with* a key present, which is the property that matters.
    """
    for var in _TRACING_VARS:
        os.environ[var] = "false"
    try:
        from langsmith._internal import _context as _ls_context
        _ls_context._GLOBAL_TRACING_ENABLED = False
    except Exception:                                      # noqa: BLE001
        # A version that moved the global. The environment is still closed and
        # `tracing_disabled()` is what the tests assert, so a miss here is
        # visible rather than silent.
        pass


_disable_hosted_tracing()

from dataclasses import replace                                     # noqa: E402
from typing import Annotated, Any, Literal, TypedDict               # noqa: E402

from langgraph.checkpoint.memory import InMemorySaver               # noqa: E402
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.graph import END, START, StateGraph                  # noqa: E402
from langgraph.types import interrupt                               # noqa: E402

from . import candidates as cand                                    # noqa: E402
from . import observability as obs                                  # noqa: E402
from . import understanding as und                                  # noqa: E402
from .answer import Provenance                                      # noqa: E402
from .conversation import (                                         # noqa: E402
    ConversationState, Denial, FactHistory, NewCase, SessionFact, merge_facts,
    merge_observations, opens_a_new_case,
)
from .router import Path_, split_by_topic                           # noqa: E402

# The domain types a checkpoint may rehydrate. Named rather than left to a
# permissive default: a checkpoint is data, and a deserialiser that will
# construct any class it is told to is a deserialisation bug waiting for a
# writable checkpoint store.
ALLOWED_TYPES = [
    ("assistant.conversation", "SessionFact"),
    ("assistant.conversation", "FactHistory"),
    ("assistant.conversation", "FactStatus"),
    ("assistant.conversation", "ConversationState"),
    ("assistant.understanding", "TurnUnderstanding"),
    ("assistant.understanding", "ResolvedRequest"),
    ("assistant.understanding", "Intent"),
    ("assistant.candidates", "CandidateAssessment"),
    ("assistant.candidates", "RecommendationDecision"),
    ("assistant.candidates", "Sufficiency"),
    ("assistant.candidates", "Outcome"),
    ("assistant.answer", "Provenance"),
    # The finished answer travels in the `answer` channel and is checkpointed
    # with the rest of the turn. These two were missing, and the checkpointer
    # said so on every turn -- "Blocked deserialization of assistant.answer.
    # Answer" -- while nothing failed, because until now nothing read a
    # checkpoint back. The moment the checkpointer became the source of
    # continuity that silence would have become an answer coming back as None.
    ("assistant.answer", "Answer"),
    ("assistant.answer", "SlotFact"),
    ("assistant.conversation", "NewCase"),
    # The other reducer instruction, and listed for the same reason: a node's
    # pending writes are checkpointed alongside the channel values, so a turn
    # interrupted between `resolve_state` and the reducer has one in the store.
    ("assistant.conversation", "Denial"),
    ("assistant.conversation", "Case"),
    ("assistant.router", "Path_"),
]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)


class PostgresCheckpointerUnavailable(RuntimeError):
    """Named so a caller can catch it without importing a driver."""


def checkpointer_for(dsn: str = ""):
    """Where conversation state lives. One boundary, two intended adapters.

    The same shape as `KnowledgeRepository` and for the same reason: the
    orchestration must not know whether it is talking to memory or to
    PostgreSQL, so the assessment path and the deployment path stay one system
    rather than two that resemble each other.

    **Status, stated precisely because the distinction matters.** The in-memory
    adapter is built and used. The PostgreSQL adapter is a *seam*, not a working
    adapter, and the reason is a version conflict rather than a decision:
    `langgraph-checkpoint-postgres` publishes 3.0.1, which requires
    `langgraph-checkpoint<4`, while `langgraph==1.2.11` requires `>=4.1.0`.
    Installing it downgrades the core package and breaks the graph. So this
    raises rather than pretending, and the honest summary is
    *documented/roadmap only* until the two are compatible.

    In-memory means conversation state does not survive a restart and is not
    shared between processes. For the assessment path that is correct -- one
    process, one machine, sessions that expire in thirty minutes anyway. For a
    deployed service behind more than one worker it is not, and this is the
    single place that has to change.
    """
    if not dsn:
        return InMemorySaver(serde=serializer())
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as error:
        raise PostgresCheckpointerUnavailable(
            "A PostgreSQL checkpointer was requested but "
            "`langgraph-checkpoint-postgres` is not installed, and at "
            "`langgraph==1.2.11` it cannot be: the published version requires "
            "`langgraph-checkpoint<4` and langgraph requires `>=4.1.0`. "
            "Conversation state therefore lives in memory and does not survive "
            "a restart."
        ) from error
    saver = PostgresSaver.from_conn_string(dsn)
    saver.setup()
    return saver


class ResetTrace(list):
    """A trace value that replaces rather than extends.

    The `trace` channel is checkpointed with everything else, so once the
    checkpointer started outliving a turn the append reducer began accumulating
    across the whole conversation: turn three's diagnostics listed turn one's
    nodes as well as its own. That is wrong in the place it is most likely to be
    believed -- "which path did this answer take" is exactly what someone reads
    a trace to find out, and the honest answer for one answer is one turn.

    The conversation-level view is not lost; it is the sequence of per-turn
    traces in the span tree, where `assistant/observability.py` already keeps it
    with timings attached.
    """


def _append(a: list, b) -> list:
    if isinstance(b, ResetTrace):
        return list(b)
    return list(a or []) + list(b or [])


class TurnState(TypedDict, total=False):
    """One turn's working set.

    `raw_question` is never written by any node after `understand_turn` reads
    it, and no node concatenates it with anything. That is the invariant the
    whole separation rests on, and it is cheap to hold here because the
    transcript lives in a different key entirely and only the composing node
    reads that one.
    """

    raw_question: str
    history: str                 # the transcript, for composition only
    images: list
    audiences: tuple
    turn_index: int
    session_id: str

    facts: Annotated[dict, merge_facts]
    observations: Annotated[list, merge_observations]
    # What *this turn's* upload showed, as plain data. Reset by
    # `understand_turn` like every other per-turn field: a checkpointed channel
    # keeps its value until something overwrites it, and a turn with no
    # photograph reporting the previous turn's reading would tell somebody the
    # assistant had just looked at an image it was not sent.
    perception: dict
    understanding: Any
    resolved: Any
    missing: list
    case_opened: str
    resumed_question: str
    hits: list
    decision: Any                # cand.RecommendationDecision
    answer: Any                  # assistant.answer.Answer
    boundary_answer: Any         # deterministic conversation-only response
    active_product: str         # lookup topic, never installation testimony
    answers: list                # (part, Answer) pairs, for a split message
    outcome: str
    trace: Annotated[list, _append]


class Services:
    """Everything the nodes call. Injected, so the graph stays testable.

    A `VisionProvider` is one of these rather than a module-level call, which is
    what lets a structural test supply known observations without a model and
    without a network -- the deterministic half of the evaluation the spec asks
    for.
    """

    def __init__(self, *, retriever, router, engine, repo, registry,
                 vision=None, matrix=None, understanding_enabled: bool = True,
                 interruptible: bool = True, checkpointer=None):
        self.retriever = retriever
        self.router = router
        self.engine = engine
        self.repo = repo
        # Refreshed per turn from the pinned snapshot, and read by the nodes at
        # call time rather than captured when the graph was compiled. Check 5
        # trusts this list, so a graph built once at start-up must not hold a
        # registry from start-up -- a product withdrawn since would still be
        # recommendable.
        self.registry = registry
        self.vision = vision
        self.matrix = matrix
        self.checkpointer = checkpointer or checkpointer_for()
        self.understanding_enabled = understanding_enabled
        # Whether a missing fact pauses the graph (`interrupt()`) or is rendered
        # as an answer. See `after_missing`.
        self.interruptible = interruptible


# ------------------------------------------------------------------- nodes

def build(services: Services):
    """Compile the graph. Nodes close over `services`; none imports a store."""

    def understand_turn(state: TurnState) -> dict:
        question = state["raw_question"]
        boundary = und.state_only_answer(
            question, services.router.slots, services.registry,
            _state_of(state), state.get("turn_index", 0),
            gate=services.router.gate,
            active_product=state.get("active_product", ""))
        reading = (und.deterministic(question, services.router.slots,
                                    services.router.gate, services.registry)
                   if boundary is not None else
                   und.understand(question, services.router.slots,
                                 enabled=services.understanding_enabled,
                                 gate=services.router.gate,
                                 registry=services.registry))
        # `ResetTrace`, because this node is always first and a trace describes
        # one turn. See `ResetTrace`.
        # Per-turn fields are cleared here, by the node that always runs first.
        # A checkpointed channel keeps its value until something overwrites it,
        # so anything describing *this* turn has to be reset rather than left:
        # `resumed_question` set on turn two was still being read on turn three,
        # and the page reported that it had answered a question from two
        # messages ago.
        return {"understanding": reading,
                "boundary_answer": boundary,
                "decision": None,
                "hits": [],
                "missing": [],
                "outcome": "",
                "resumed_question": "",
                "case_opened": "",
                # Cleared, or a turn that produces no answer of its own -- one
                # that pauses, for instance -- hands back the previous turn's.
                # That is how a paused selection came out of the checkpoint
                # wearing the last answer's `compose` path.
                "answers": [],
                "answer": None,
                # See `TurnState.perception`. A turn that sends no photograph
                # must not inherit the last one's reading.
                "perception": {},
                "trace": ResetTrace(["understand_turn"])}

    def case_boundary(state: TurnState) -> dict:
        """Is this turn about a different wall? If so, start a case.

        Runs before `resolve_state`, which is the only order that works: the
        inheritance in `understanding.resolve` fills every empty slot from the
        conversation, so by the time facts have been resolved the old wall's
        substrate is already on the new wall and there is nothing left to
        separate.

        This is the fix for measured contamination, not a precaution. Traced
        through the real path: "My internal wall is brick" then "Now I have
        another wall outside" produced `{substrate: brick, location: external}`
        -- a wall that exists nowhere, and the substrate printed back as
        "brick, as you told me earlier".
        """
        question = state["raw_question"]
        reading = state["understanding"]
        opens, why = opens_a_new_case(
            und.asserted_text(question),
            model_says=getattr(reading, "new_subject", False))
        if not opens:
            return {"trace": ["case_boundary:same"]}

        # What this turn itself stated stays with the new case; everything the
        # old wall established does not. `NewCase` is a reducer instruction
        # rather than an assignment, so the reset goes through `merge_facts`
        # like every other write and cannot bypass supersession.
        detected = services.router.slots.detect(und.asserted_text(question))
        turn = state.get("turn_index", 0)
        keep = {slot: SessionFact(slot, value, Provenance.STATED, turn)
                for slot, value in detected.items()}
        obs.event("case_opened", reason=why, kept=sorted(keep),
                  turn=turn)
        reset = NewCase(keep=keep, reason=why)
        return {"facts": reset,
                "active_product": "",
                "observations": reset,
                "case_opened": why,
                "trace": [f"case_boundary:new ({why})"]}

    def resolve_state(state: TurnState) -> dict:
        """This turn's stated facts, into the reducer that owns supersession."""
        reading = state["understanding"]
        resolved = und.resolve(reading, state["raw_question"],
                               services.router.slots, services.registry,
                               state=_state_of(state),
                               turn_index=state.get("turn_index", 0),
                               active_product=state.get("active_product", ""))
        new = und.facts_from(resolved, state.get("turn_index", 0))
        for slot, fact in new.items():
            if isinstance(fact, Denial):
                # A retraction is not a fact and has no provenance to log. It is
                # still the most audit-worthy thing a turn can do to state, so
                # it gets its own event rather than being folded into one that
                # would have to report an origin it does not have.
                obs.event("fact_denied", slot=slot, value=fact.value,
                          turn=state.get("turn_index", 0))
                continue
            obs.event("inherited_fact", slot=slot,
                      provenance=fact.provenance.value, source_turn=fact.source_turn)
        topic = und.topic_product(state["raw_question"], services.registry)
        return {"facts": new, "resolved": resolved,
                "active_product": topic or state.get("active_product", ""),
                "trace": ["resolve_state"]}

    def answer_state(state: TurnState) -> dict:
        # Re-read after a possible case boundary; never answer from a retired wall.
        answer = und.state_only_answer(
            state["raw_question"], services.router.slots, services.registry,
            _state_of(state), state.get("turn_index", 0),
            gate=services.router.gate,
            active_product=state.get("active_product", ""))
        perception = state.get("perception") or {}
        if perception:
            answer.diagnostics["perception"] = perception
            if perception.get("enabled") is False:
                answer.text += "\n\n" + " ".join(perception.get("summary", []))
        with obs.span("part", question=obs.fingerprint(state["raw_question"])) as span:
            span["path"] = answer.path
            span["cached"] = bool(answer.diagnostics.get("cached", False))
        return {"answer": answer, "answers": [(state["raw_question"], answer)],
                "outcome": answer.path, "trace": [f"state_only:{answer.path}"]}

    def analyse_images(state: TurnState) -> dict:
        """Observations, as observations. Never promoted into stated facts."""
        images = state.get("images") or []
        if not images:
            return {"trace": ["analyse_images:skipped"]}
        # No injected provider means the real one. The alternative -- skipping
        # perception when nothing was injected -- is how wiring the UI onto this
        # path silently turned image upload off: every surface passes no
        # provider, so every photograph was ignored and the page still said it
        # had read one. A test double is the exception, not the default.
        provider = services.vision
        if provider is None:
            from . import vision as vision_module

            # `ASSISTANT_VISION_DEMO` decides whether the default provider is
            # the real one or none at all. An **explicitly injected** provider
            # is not gated: passing one is a deliberate act by a test, a
            # channel adapter or a demo driver, and a flag that overrode it
            # would make the injection point untestable.
            #
            # Switched off, the turn takes decision 16's published path -- the
            # photograph is acknowledged, not looked at, and the enquiry can go
            # to a person. The *acknowledgement* is the part that matters:
            # returning nothing here is how the page came to say it had read a
            # photograph that nothing looked at.
            if not vision_module.enabled():
                obs.event("vision_disabled", images=len(images))
                return {"perception": vision_module.disabled_report(),
                        "trace": ["analyse_images:disabled"]}

            from .vision import slots_from_images

            class _Default:
                @staticmethod
                def observe(imgs):
                    return slots_from_images(imgs)

            provider = _Default()
        turn = state.get("turn_index", 0)
        try:
            with obs.span("vlm_perception", images=len(images)) as span:
                resolution = provider.observe(images)
                span["slots"] = sorted(resolution.slots)
                span["discarded"] = len(getattr(resolution, "discarded", ()))
                span["cannot_determine"] = len(
                    getattr(resolution, "cannot_determine_from_image", ()))
        except Exception as e:
            obs.event("vision_error", reason=str(e), images=len(images))
            return {"perception": {"enabled": True, "error": str(e),
                                   "summary": ["Image reading failed. "
                                               "Answering from text only."]},
                    "facts": {}, "observations": [],
                    "trace": ["analyse_images:error"]}
        observed = {
            slot: SessionFact(slot, value, Provenance.OBSERVED, turn,
                              confidence=_confidence_for(resolution, slot),
                              image_ref=_image_for(resolution, slot))
            for slot, value in resolution.slots.items()
        }
        # What the photograph showed, as plain data, for the surface to print.
        # Separate from `facts` on purpose: `facts` is what the system will
        # *act* on and holds only the readings that routed, while this holds
        # everything that was seen, each with its certainty and -- where it did
        # not route -- the reason. A customer asking "what can you reliably
        # identify from the photo" is asking for the second one, and answering
        # it out of the first would report a rendered wall as showing nothing.
        from .vision import perception_report

        report = perception_report(resolution)
        if report.get("refused_attributes"):
            obs.event("vision_refused_attribute",
                      attributes=report["refused_attributes"])
        if report.get("truncated"):
            obs.event("vision_truncated", images=len(images))

        # Merged through the same reducer as everything else, which is what
        # makes "a photograph never overwrites a person" structural rather
        # than a rule this node has to remember.
        return {"facts": observed,
                "observations": list(observed.values()),
                "perception": report,
                "trace": [f"analyse_images:{len(observed)}"]}

    def determine_missing_information(state: TurnState) -> dict:
        """Re-resolve after the images, then ask for the minimum."""
        resolved = und.resolve(state["understanding"], state["raw_question"],
                               services.router.slots, services.registry,
                               state=_state_of(state),
                               turn_index=state.get("turn_index", 0),
                               active_product=state.get("active_product", ""))
        missing = (cand.missing_facts(resolved)
                   if resolved.intent is und.Intent.SELECT else [])
        return {"resolved": resolved, "missing": missing,
                "trace": [f"missing:{missing}"]}

    def ask_back(state: TurnState) -> dict:
        """Pause and wait, keeping the question that caused the pause.

        `interrupt()` is the reason the ask-back cycle is expressible at all
        rather than reconstructed from a `pending` string on the next request.
        The original question is in the state, so resuming answers what was
        asked instead of answering the word that was typed.
        """
        missing = state["missing"]
        # Everything after this line runs on the *resuming* request, not this
        # one. `interrupt()` raises out of the node, the checkpoint keeps the
        # position and the whole state, and the graph re-enters here when a
        # caller invokes with `Command(resume=...)`.
        obs.event("ask_back_interrupt", missing=missing,
                  turn=state.get("turn_index", 0))
        reply = interrupt({"ask": missing[0], "resuming": state["raw_question"],
                           "missing": missing})
        # Kept so a surface can say which question it just answered. The page
        # shows "answered: <the original question>", and after a resume the
        # message the person typed was "brick" -- reporting that as the question
        # would be technically true and useless.
        resumed_question = state["raw_question"]

        turn = state.get("turn_index", 0)
        detected = services.router.slots.detect(und.asserted_text(str(reply)))
        obs.event("ask_back_resumed", supplied=sorted(detected),
                  answered=bool(detected))
        return {"facts": {slot: SessionFact(slot, value, Provenance.STATED, turn)
                          for slot, value in detected.items()},
                # Re-computed after the answer arrives, so a resume that
                # supplied the missing fact does not still look like it is
                # missing to the node downstream.
                "missing": [m for m in missing if m not in detected],
                "resumed_question": resumed_question,
                "trace": ["ask_back:resumed"]}

    def retrieve_candidates(state: TurnState) -> dict:
        """Embed the resolved request. Never the transcript.

        The request is re-resolved here rather than read from the state, and
        that is a correctness fix rather than tidiness. `determine_missing_
        information` resolved it *before* the ask-back paused, so on a resumed
        turn the stored request predates the answer the person just gave: it
        still has no substrate, `assess_evidence` recomputes the same missing
        fact, and the conversation asks the identical question again. Measured
        exactly that way -- "brick" was accepted into the facts and the reply
        was still "what is the wall built of?".

        Resolving from the current facts makes the node's contract "retrieve for
        what is known now", which is true whether this turn paused or not.
        """
        resolved = und.resolve(state["understanding"], state["raw_question"],
                               services.router.slots, services.registry,
                               state=_state_of(state),
                               turn_index=state.get("turn_index", 0),
                               active_product=state.get("active_product", ""))
        hits = services.retriever.search(
            resolved.retrieval_query(),
            audiences=tuple(state.get("audiences", ("public",))),
            product=resolved.product)
        return {"hits": hits, "resolved": resolved,
                "trace": [f"retrieve:{len(hits)}"]}

    def assess_evidence(state: TurnState) -> dict:
        decision = cand.decide(state["resolved"], state.get("hits", []),
                               services.registry, services.repo,
                               services.router.slots,
                               tuple(state.get("audiences", ("public",))),
                               services.matrix)
        return {"decision": decision, "outcome": decision.outcome.value,
                "trace": [f"assess:{decision.outcome.value}"]}

    def delegate(state: TurnState) -> dict:
        """Everything that is not a selection, answered exactly as before.

        LOOKUP, VERIFY, CALCULATE, UNDERSTAND, TROUBLESHOOT, FIND and ESCALATE
        run through the existing engine untouched. The graph orders the turn; it
        does not re-implement the seven routes that already work, and a
        regression in any of them would be a cost with no matching benefit.
        """
        resolved = state["resolved"]
        audiences = tuple(state.get("audiences", ("public",)))
        # Split here, exactly as `Assistant.ask` does. Two *jobs* in one message
        # are two questions -- "how much does Solo cost, and what coverage does
        # it give" needs the policy gate on one half and retrieval on the other.
        # Answering it as one part would have been a regression introduced by
        # the orchestration rather than by any change in the answering, which is
        # the kind of divergence having two paths invites.
        parts = ([state["raw_question"]]
                 if resolved.policy_topic in und.MANDATORY_TOPICS
                 else split_by_topic(state["raw_question"]))
        answers = []
        for part in parts:
            # One `part` span per topic, as `Assistant._ask` has always emitted.
            # It went missing when the UI moved onto this path and took the
            # route attribute with it -- the row an operator reads to answer
            # "which way did this go" with the prose long gone. Restoring it
            # here rather than inside `answer_part` avoids nesting it twice on
            # the `ask()` path, which still wraps its own.
            with obs.span("part", question=obs.fingerprint(part)) as span:
                answer = services.engine.answer_part(
                    part, audiences, carried=resolved.slots(),
                    origins=resolved.provenance, history=state.get("history", ""))
                span["path"] = answer.path
                span["cached"] = bool(answer.diagnostics.get("cached", False))
            answers.append(answer)
        return {"answers": list(zip(parts, answers)),
                "answer": answers[0] if answers else None,
                "outcome": answers[0].path if answers else "",
                "trace": [f"delegate:{'+'.join(a.path for a in answers)}"]}

    def recommend(state: TurnState) -> dict:
        decision = state["decision"]
        answer = services.engine.recommend(
            state["resolved"], decision, state.get("hits", []),
            history=state.get("history", ""))
        with obs.span("part", question=obs.fingerprint(state["raw_question"])) as span:
            span["path"] = answer.path
            span["outcome"] = answer.diagnostics.get("outcome", "")
        return {"answer": answer,
                "answers": [(state["raw_question"], answer)],
                "trace": ["recommend"]}

    def need_more_information(state: TurnState) -> dict:
        """Ask for the minimum, from either side of the evidence gate.

        Reached two ways: before retrieval, when a required fact is missing, and
        after it, when the gate itself reports one. The second passes a
        decision; the first has none yet, so one is made here rather than
        letting the node require a stage that has not run.
        """
        decision = state.get("decision") or cand.RecommendationDecision(
            outcome=cand.Outcome.NEED_MORE_INFORMATION,
            missing=tuple(state.get("missing") or ()),
            reason="a product cannot be chosen without: "
                   + ", ".join(state.get("missing") or ()))
        answer = services.engine.need_more_information(state["resolved"],
                                                       decision)
        with obs.span("part", question=obs.fingerprint(state["raw_question"])) as span:
            span["path"] = answer.path
            span["outcome"] = decision.outcome.value
        return {"answer": answer, "decision": decision,
                "answers": [(state["raw_question"], answer)],
                "outcome": decision.outcome.value,
                "trace": ["need_more_information"]}

    def no_supported_recommendation(state: TurnState) -> dict:
        answer = services.engine.no_supported_recommendation(
            state["resolved"], state["decision"])
        with obs.span("part", question=obs.fingerprint(state["raw_question"])) as span:
            span["path"] = answer.path
            span["outcome"] = answer.diagnostics.get("outcome", "")
        return {"answer": answer,
                "answers": [(state["raw_question"], answer)],
                "trace": ["no_supported_recommendation"]}

    def verify(state: TurnState) -> dict:
        """Two containment checks, and the second one does not trust the first.

        The inner check asks whether a *selection* stayed inside its approved
        set. It only engages when there is an approved set, which means it only
        engages when the request was classified ``SELECT`` -- and that makes
        intent classification the single point of failure for the whole
        recommendation safety story. Intent classification is also the one stage
        here that involves a model reading a sentence.

        The outer check asks a question that does not depend on the label at
        all: does this answer, whatever route produced it, tell somebody to use
        a product? If it does, every product it names must have been approved by
        an evidence assessment in this turn. A selection misread as a lookup
        goes to `delegate` and can compose, and without this it would compose a
        recommendation that never met the evidence gate. With it, the worst a
        misclassification can produce is a refusal.
        """
        answer = state.get("answer")
        if answer is None:
            return {"trace": ["verify:n/a"]}

        # A refusal has nothing to contain, and running the guard over one does
        # real damage. This is not a shortcut; it is a category error corrected.
        #
        # What a refusal prints is the hand-off template plus the closest
        # published passage, quoted and cited, which is the value decision 9
        # says a refusal must carry. That passage is published prose, so it
        # routinely names products and routinely reads as advice — "Suitable
        # finishing coats are Lime Green Natural Finish, Lime Green Solo or
        # Fine Stuff" is a sentence out of a datasheet, not a recommendation
        # this system made. `recommends_a_product` cannot tell those apart,
        # because by design it reads text rather than provenance.
        #
        # Measured, on the near-miss the relevance gate exists for. "What is
        # the U-value of Solo Onecoat plaster?" refuses correctly at step 4
        # with one cited passage. The guard then matched the product names
        # inside that refusal, found no approved set to excuse them — a LOOKUP
        # has no evidence assessment — and no exemption either, because the
        # question's own phrasing does not match the registry's spelling
        # "Solo Onecoat Lime Plaster" exactly. So it discarded the informative
        # refusal and substituted a bare one citing nothing at all.
        #
        # The trade was strictly negative: a refusal replaced by a refusal, one
        # of them useful. And it was invisible to the canonical evaluation,
        # because the situations half calls `Assistant.ask` while every real
        # surface -- the page, the CLI conversation -- calls `ask_turn` and
        # comes through here.
        if answer.refused:
            return {"trace": ["verify:refused"]}

        decision = state.get("decision")
        approved = decision.approved_names if decision else frozenset()

        with obs.span("verification",
                      products=len(approved),
                      route=answer.path) as span:
            # Outer: intent-independent, and scoped to what it can honestly
            # claim. The property enforced is **the system must not introduce a
            # product the person did not ask about and tell them to use it
            # without an evidence assessment.**
            #
            # A product the caller named themselves is exempt, and the exemption
            # is not a loophole -- it is the difference between answering a
            # question and making a recommendation. "Would Ultra work on my
            # brick wall?" is answered about Ultra by definition, and that
            # answer is already bound by the relevance gate and the six checks.
            # Without the exemption this guard refuses every VERIFY question
            # whose honest answer contains the word "suitable", which is most of
            # them: an over-refusal on the commonest shape of question, traded
            # for nothing.
            #
            # Every product the caller named, not only the one topic. "Compare
            # Ultra and Solo" is about both, and a request with two subjects
            # has no single `resolved.product` -- so the exemption used to
            # cover neither, and the guard refused a comparison for naming the
            # products it was asked to compare. Acceptance case T14 states the
            # rule: both are allowed because the caller named them.
            asked_about = ({und.normalise_product(p) for p in
                            ((state["resolved"].product,)
                             + tuple(state["resolved"].candidate_products)) if p}
                           | und.named_products(state["raw_question"],
                                                services.registry))
            allowed = {und.normalise_product(a) for a in approved} | asked_about
            recommended = cand.recommends_a_product(answer.text,
                                                    services.registry)
            unapproved = [p for p in recommended if p not in allowed]
            span["asked_about"] = sorted(asked_about)
            span["recommends"] = recommended
            span["unapproved"] = unapproved

            # Inner: a selection may not widen its own approved set.
            contained, intruders = (cand.contained(answer.text, decision,
                                                   services.registry)
                                    if decision and decision.approved
                                    else (True, []))
            span["contained"] = contained
            span["intruders"] = intruders

        if unapproved:
            obs.event("check_failed", check="unapproved_recommendation",
                      products=unapproved, route=answer.path,
                      intent=state["resolved"].intent.value)
            return _refuse(state, decision,
                           # Title case for the reader. The canonical spelling
                           # is an internal comparison key, not a product name
                           # anybody would recognise in a sentence.
                           "the answer recommended "
                           + ", ".join(p.title() for p in unapproved)
                           + " without an evidence assessment supporting it",
                           f"verify:unapproved {unapproved}")
        if not contained:
            obs.event("check_failed", check="containment", intruders=intruders)
            return _refuse(state, decision,
                           "the composed answer named a product the evidence "
                           "had not approved: " + ", ".join(intruders),
                           f"verify:rejected {intruders}")
        return {"trace": ["verify:ok"]}

    def _refuse(state: TurnState, decision, why: str, note: str) -> dict:
        """Fail closed, into the refusal the rest of the system already uses."""
        decision = decision or cand.RecommendationDecision(
            outcome=cand.Outcome.NO_SUPPORTED_RECOMMENDATION, reason=why)
        refused = services.engine.no_supported_recommendation(
            state["resolved"], decision, why=why)
        return {"answer": refused,
                "answers": [(state["raw_question"], refused)],
                "outcome": cand.Outcome.NO_SUPPORTED_RECOMMENDATION.value,
                "trace": [note]}

    # -------------------------------------------------------- the ordering

    def after_missing(state: TurnState) -> Literal[
            "ask_back", "need_more_information", "retrieve_candidates",
            "delegate", "answer_state"]:
        """SELECT goes through the evidence gate; everything else does not.

        Note what this cannot express: there is no edge from an insufficient
        selection to `delegate`. Compose is not a fallback for a recommendation
        whose evidence did not hold up, and the way to guarantee that is to
        build a graph in which no such edge exists.

        **A missing fact pauses the graph.** That was not true when this was
        first written, and the reason it was not is worth keeping: a paused
        graph has to be held *somewhere* between two HTTP requests, and the
        checkpointer was being rebuilt every turn, so there was nowhere to hold
        it. Rendering the question as an ordinary answer was the only thing that
        worked.

        With an application-lifetime checkpointer there is somewhere. The pause
        is a real `interrupt()`, the paused state lives in the checkpoint under
        the conversation's `thread_id`, and the next request resumes it with
        `Command(resume=...)` -- so the question the person originally asked is
        answered from the point it stopped, rather than reconstructed from a
        `pending` string on the way back in.

        `interruptible=False` still renders instead, for a caller with no way to
        resume -- a one-shot evaluation of a single turn, for instance.
        """
        if state.get("boundary_answer") is not None:
            return "answer_state"
        if state["resolved"].intent is not und.Intent.SELECT:
            return "delegate"
        if not state.get("missing"):
            return "retrieve_candidates"
        return "ask_back" if services.interruptible else "need_more_information"

    def after_evidence(state: TurnState) -> Literal[
            "recommend", "need_more_information", "no_supported_recommendation"]:
        outcome = state["decision"].outcome
        if outcome is cand.Outcome.NEED_MORE_INFORMATION:
            return "need_more_information"
        if outcome in (cand.Outcome.SUPPORTED_RECOMMENDATION,
                       cand.Outcome.CONDITIONAL_RECOMMENDATION):
            return "recommend"
        return "no_supported_recommendation"

    def after_resolve(state: TurnState) -> Literal["answer_state", "analyse_images"]:
        boundary = state.get("boundary_answer")
        if boundary is not None:
            if state.get("images"):
                return "analyse_images"
            return "answer_state"
        return "analyse_images"

    graph = StateGraph(TurnState)
    for fn in (understand_turn, case_boundary, resolve_state, analyse_images,
               determine_missing_information, ask_back, retrieve_candidates,
               assess_evidence, delegate, recommend, need_more_information,
               no_supported_recommendation, verify, answer_state):
        graph.add_node(fn.__name__, fn)

    graph.add_edge(START, "understand_turn")
    graph.add_edge("understand_turn", "case_boundary")
    graph.add_edge("case_boundary", "resolve_state")
    graph.add_conditional_edges("resolve_state", after_resolve)
    graph.add_edge("answer_state", END)
    graph.add_edge("analyse_images", "determine_missing_information")
    graph.add_conditional_edges("determine_missing_information", after_missing)
    graph.add_edge("ask_back", "retrieve_candidates")
    # `need_more_information` is reachable from both sides of the gate.
    graph.add_edge("retrieve_candidates", "assess_evidence")
    graph.add_conditional_edges("assess_evidence", after_evidence)
    graph.add_edge("recommend", "verify")
    graph.add_edge("need_more_information", "verify")
    graph.add_edge("no_supported_recommendation", "verify")
    graph.add_edge("delegate", "verify")
    graph.add_edge("verify", END)

    # The checkpointer is supplied, not constructed. Building one here meant a
    # new one per call, and `Assistant.ask_turn` called `build()` every turn --
    # so `thread_id` addressed a store that was empty by construction and
    # carried nothing between turns. Continuity came entirely from the caller
    # handing `ConversationState` back in, which is the manual discipline
    # adopting a graph was supposed to replace.
    return graph.compile(checkpointer=services.checkpointer)


# ----------------------------------------------------------------- helpers

def _state_of(state: TurnState) -> ConversationState:
    """The conversation, as `assistant/understanding.py` wants to read it."""
    return ConversationState(facts=dict(state.get("facts") or {}),
                             turn_index=state.get("turn_index", 0))


def _confidence_for(resolution, slot: str) -> float:
    for attribute in getattr(resolution, "attributes", ()):
        if attribute.slot == slot:
            return attribute.confidence
    return 0.0


def _image_for(resolution, slot: str) -> str:
    for attribute in getattr(resolution, "attributes", ()):
        if attribute.slot == slot and attribute.sources:
            return attribute.sources[0].image
    return ""


def tracing_disabled() -> bool:
    """Is the hosted tracing client actually inert, right now?

    Asks `langsmith` itself rather than inspecting the environment, because the
    environment is only one of four inputs it consults and the previous version
    of this module was caught believing otherwise. The dependency is acceptable
    only while this returns True, so it is checked by a test under a hostile
    environment rather than documented as an intention.
    """
    try:
        from langsmith.utils import tracing_is_enabled
    except ImportError:
        return True
    return not tracing_is_enabled()


def reassert_tracing_disabled() -> bool:
    """Close it again, for a caller that suspects something reopened it.

    Exists because the failure mode is silent: nothing raises when tracing turns
    on, a conversation simply leaves the building. A surface that wants to be
    certain at request time can call this instead of trusting import order.
    """
    _disable_hosted_tracing()
    return tracing_disabled()
