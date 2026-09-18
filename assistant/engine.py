"""The assembled assistant: one question in, one composite reply out.

The order here is the order in the diagram, and the reason it is a separate
module from the router is that the router decides and this executes. Keeping
those apart is what makes the router testable without Ollama running.

A question with two topics in it produces two parts, each gated and routed on
its own, and the reply says which part was answered from what. A message that
asks a price and a coverage should not have one path chosen for both.
"""

from __future__ import annotations

import os

import re
from dataclasses import dataclass, field, replace

from . import observability as obs
from . import ollama
from . import phrasing
from . import vision
from .answer import (Answer, AnswerEngine, Provenance, _named_aliases,
                     _product_aliases, _requested_fields)
from .cache import AnswerCache
from .model import AnswerLogEntry
from .retrieve import Retriever
from .router import Decision, Path_, Router, split_by_topic


# The architecture caps input at about 500 words. Capping characters instead
# cut mid-word, and because the cap is applied before the message is split by
# topic, the fragment became its own part: it matched no policy pattern, so it
# reached retrieval as gibberish. Words are the unit the limit was written in
# and the unit that cannot produce a fragment.
MAX_WORDS = 500

# What a coverage figure is printed under, for the targeted second retrieval on
# the calculation edge. Lexical rather than semantic on purpose: this runs when
# similarity has already failed to surface the passage.
COVERAGE_TERMS = ("coverage", "covers", "cover", "m2", "m²", "per bag", "per sack")

PROPERTY_TERMS = {
    "water": ("water",),
    "coverage": COVERAGE_TERMS,
    "thickness": ("thickness", "thick", "mm"),
    "backgrounds": ("background", "substrate", "masonry", "lath", "suitable"),
    "preparation": ("preparation", "prepare", "dampen", "dust", "suction", "primer"),
    "curing": ("curing", "cure", "drying", "protect", "dampen"),
    "conditions": ("temperature", "weather", "frost", "degrees", "°c"),
}

# The manufacturer, as the website writes it in front of its own product names.
# Stripped before matching, because chunks are tagged with the catalogue name.
_BRAND = "lime green"

# How many passages a word may appear in and still count as naming something
# specific. Measured rather than guessed: a word returning more than this is
# ordinary vocabulary, not the thing the question was about.
DISTINCTIVE_CAP = 6

# How many passages the relevance gate's second chance may add. Small on
# purpose: this fires when the gate is about to refuse, so it is recovering a
# specific published fact, not widening the evidence base. Three is the same
# bound the targeted coverage lookup uses.
GATE_SECOND_CHANCE = 3

# How many recovered passages may join the ranked ones. Small on purpose: this
# is a repair for a specific miss, not a second retrieval, and the five ranked
# passages are still what the answer is mostly built from.
MISSED_EVIDENCE_LIMIT = 3

# How much else a recovered passage must have in common with the question,
# beyond the rare word that found it. One shared word is a coincidence.
RELATED_WORDS = 3


def _content_words(text: str) -> set[str]:
    """The words worth matching on: long enough to mean something."""
    return {w for w in re.findall(r"[a-z]{5,}", text.lower())
            if w not in _QUESTION_NOISE}

# Words a question uses to ask rather than to specify. They pass the length
# filter and name nothing, so chasing them would add noise to every answer.
_QUESTION_NOISE = frozenset("""
about after again against along already always another anything around
because before being below between could directly during either enough every
first found further getting having instead might other please rather really
should since something still their there these thing think those through
using usually whether which while would
""".split())


def cap(question: str) -> str:
    """Trim an over-long message on a word boundary."""
    words = question.strip().split()
    return " ".join(words[:MAX_WORDS])


class _AssertionSlots:
    """Keep routing cues raw, but accept building facts only as assertions."""

    def __init__(self, detector):
        self.detector = detector

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect(self, question):
        from .understanding import asserted_text

        slots = self.detector.detect(question)
        asserted = asserted_text(question)
        if asserted != question:
            facts = self.detector.detect(asserted)
            for slot in ("substrate", "location", "exposure"):
                slots.pop(slot, None)
                if slot in facts:
                    slots[slot] = facts[slot]
        return slots


@dataclass
class Reply:
    """The composite answer to a message, part by part."""

    question: str
    parts: list[tuple[str, Answer]] = field(default_factory=list)
    audiences: tuple[str, ...] = ("public",)
    correlation_id: str = ""

    @property
    def refused(self) -> bool:
        return all(a.refused for _q, a in self.parts) if self.parts else True

    @property
    def paths(self) -> list[str]:
        return [a.path for _q, a in self.parts]


class Assistant:
    """Retrieval, routing and answering behind one call."""

    def __init__(self, repo, threshold: float | None = None, *,
                 cache: bool = True, log: bool = True,
                 source: str = "unknown", checkpointer=None) -> None:
        self.log = log
        # Which surface this instance answers for, written onto every audit row.
        # It defaults to `unknown` rather than to `cli` because a caller that
        # did not say is a caller whose traffic cannot honestly be counted as
        # anything — and the whole reason this parameter exists is that
        # evaluation traffic was being counted as real. See AnswerLogEntry.
        self.source = source
        self.repo = repo
        self.retriever = Retriever(repo, **({"threshold": threshold}
                                            if threshold is not None else {}))
        self.router = Router()
        self.router.slots = _AssertionSlots(self.router.slots)
        self.engine = AnswerEngine(repo, self.router)
        # Exact-key, audience-scoped, snapshot-scoped. See assistant/cache.py
        # for why it is not the template-keyed cache decision 14 designs.
        self.cache = AnswerCache() if cache else None
        # The engine needs the slot vocabulary for the relevance check, and the
        # router owns it. Handing over the router rather than a copy keeps one
        # definition of what a term means.
        self.engine.retriever = self.router
        # One checkpointer for this assistant's lifetime. Cheap: it constructs
        # no graph and imports no orchestration library until `ask_turn` is
        # actually called.
        self._init_orchestration(checkpointer)

    def _init_orchestration(self, checkpointer=None) -> None:
        """One checkpointer for this assistant's lifetime.

        Constructed here rather than per turn, which is the correction P0-3
        exists for: `build()` used to make a fresh `InMemorySaver` on every
        call, so `thread_id` addressed a store that was empty by construction.
        """
        from .graph import checkpointer_for

        # `ASSISTANT_CHECKPOINT_DSN`, not `ASSISTANT_POSTGRES_DSN`: where the
        # knowledge lives and where a half-finished conversation lives are
        # separate choices, and joining them would give a PostgreSQL knowledge
        # store an unusable checkpointer for free. Unset -- the local and
        # interview default -- is in-memory state: it does not survive a
        # restart and is not shared between workers, which for one process
        # answering one assessor is the right trade. A DSN here does not
        # silently fall back; `checkpointer_for` raises, because a deployment
        # that believed its conversations were durable and lost them would be
        # worse than one that refuses to start.
        self.checkpointer = checkpointer or checkpointer_for(
            os.environ.get("ASSISTANT_CHECKPOINT_DSN", ""))
        self._services = None
        self._graph = None
        # The release this turn is pinned to, set inside `ask_turn`.
        self._turn_snapshot = None

    def detect_answer_to_askback_slots(self, question: str) -> dict[str, str]:
        """Extract slots if question looks like an answer to an ask-back.

        Returns a dict of newly-detected slots (substrate, location, exposure)
        if the question is short and contains load-bearing slot terms.
        Otherwise returns empty dict.

        Used by the UI to detect when a user is answering a missing-fact
        ask-back so we can re-ask the pending question with merged slots.
        """
        if not self.router.slots.is_answer_to_askback(question):
            return {}
        return self.router.slots.detect(question)

    def ask(self, question: str, audiences: tuple[str, ...] = ("public",),
            correlation_id: str = "", carried: dict | None = None,
            images=None, session_id: str = "", turn_id: str = "",
            context: str = "", *, state=None, turn_index: int = 0) -> Reply:
        """One message in, one composite reply out.

        `carried` is what the caller already knows that this question does not
        say — slots held from an earlier turn in the same conversation. The
        engine does not decide what is worth carrying and does not store it; a
        session owns that, and hands it in. Keeping the memory outside the
        engine is what lets the CLI stay single-turn and stateless while the web
        page is neither.

        `context` is prior conversation history (optional) for multi-turn
        reasoning. When provided, it is prepended to the question before
        generation, so the model can see earlier turns and make coherent
        follow-ups. Format: "Earlier in this conversation, you answered: [Q1].
        You answered: [A1]. Then the user asked: [Q2]..."

        `images` are photographs attached to *this* turn — file paths or raw
        bytes, whatever `assistant/vision.py` reads. They are perceived here
        rather than by the caller because perception is part of answering one
        question, and because two surfaces each calling `slots_from_images`
        would be two places for the merge order to drift apart.

        **The merge order is session, then vision, then question**, and each
        step of it is a safety property rather than a preference:

        * A photograph attached now is newer evidence than a slot carried from
          a turn two questions ago, so it wins over the session.
        * The question always wins over both. Someone who uploads a photograph
          of a stone wall and types "it is brick" is correcting the image, and
          the router already merges everything handed in *under* what the
          sentence says, so that half needs no code here at all.

        What the vision step is *not* allowed to do is also structural.
        `Resolution.slots` carries only ``CONFIRMED`` attributes, and this
        method reads nothing else — not `attributes`, not a confidence, not a
        free-text observation — so an uncertain or conflicting reading leaves
        the slot uncued and decision 10's ask-back handles it exactly as it does
        when no photograph was sent. Nothing here writes a vision-derived slot
        back into a session either: a model's reading of an image must not
        become indistinguishable from something the caller said, on every later
        turn, after the image has scrolled away.

        There is no `try` around the perception call, deliberately. `observe()`
        never raises: an unreachable Ollama, an unreadable file or a malformed
        response all resolve to an empty slot dict, which is what the caller
        would have passed had there been no photograph. Coverage drops, safety
        does not, and catching here would only hide the `error` the
        `Perception` carries for a hand-off to report.

        `session_id` and `turn_id` are observability only, and both are
        optional. They name the conversation this message belongs to and the
        message itself, so the persisted trace can be replayed as a conversation
        rather than as a pile of unrelated answers. A CLI caller passes neither
        and its spans carry an empty session id, which is the truth about it:
        the CLI constructs no session and carries nothing between questions, so
        consecutive CLI turns are unrelated by construction and a synthetic id
        would group them into a conversation that shares nothing but a terminal.

        `turn_id` is separate from the trace id even though they are one-to-one
        for this single-turn engine, because a re-asked pending question is the
        same turn and a different trace, and collapsing them would make the
        ask-back cycle unreadable in the trace.
        """
        # `context` is NOT merged into the question, and the deleted line is
        # the whole point of this block.
        #
        # `question = context + "\n\n" + question` put an earlier answer's
        # prose in front of every deterministic stage: the policy gate, the
        # slot detector, the product detector and the embedder all read it.
        # Three measured consequences, none of them subtle. A previous answer
        # containing the word "cost" routed an unrelated follow-up to the price
        # referral. A previous answer mentioning plaster put a substrate on a
        # question that named none -- `detect("What plaster should I use on my
        # wall")` returns `{}`, while `detect(context + question)` returns
        # `{'substrate': 'existing_plaster', 'property_asked': 'compatibility'}`.
        # And the embedded query became the transcript, so retrieval answered
        # the conversation rather than the question.
        #
        # One cause behind all three: assistant-generated prose became an input
        # to deterministic routing, which makes the model the controller by the
        # back door. The transcript is still useful -- to the model, on Compose,
        # for resolving "it" and "that wall" -- so it travels as its own
        # argument and arrives after the route and the passages are decided.
        history = context
        question = cap(question)
        reply = Reply(question=question, audiences=audiences)

        observed: dict[str, str] = {}
        carried = dict(carried or {})
        origins: dict[str, Provenance] = {}
        if state is not None:
            carried = state.active()
            origins = state.provenance_of()

        # One id for the whole message, including every part it splits into, so
        # the lines for a two-topic question can be read as one event. A caller
        # that already has a request id — a web request, a channel adapter —
        # passes it in rather than starting a second trace for the same work.
        with obs.correlation(correlation_id) as cid:
            reply.correlation_id = cid
            # The turn owns the span tree and collects it. Perception happens
            # inside it rather than before it, so the photograph's cost is a
            # stage of this answer rather than an untimed prelude to it.
            with obs.turn(turn=turn_id, session=session_id,
                          source=self.source) as spans:
                with obs.span("answer", audiences=list(audiences),
                              question_words=len(question.split()),
                              question=obs.fingerprint(question)) as summary:
                    boundary = self._state_answer(
                        question, carried, origins, state, turn_index)
                    if images and boundary is None:
                        # Counts and slot names only. No question text, no
                        # passage text, and the photograph is named nowhere at
                        # all — which is why the uploaded filename is not
                        # carried this far.
                        #
                        # `vision_model_call` and `resolve`, the two children
                        # the review's tree hangs under this span, are not
                        # emitted: they live inside `assistant/vision.py`, which
                        # this slice does not own and which the review found
                        # already correct. The span is honest about the stage it
                        # can see and silent about the two it cannot.
                        with obs.span("perception", images=len(images)) as p:
                            resolution = vision.slots_from_images(images)
                            observed = dict(resolution.slots)
                            p["slots"] = sorted(observed)
                            p["observations"] = len(resolution.attributes)
                            p["confirmed"] = len(observed)
                            p["discarded"] = len(resolution.discarded)
                            p["cannot_determine"] = len(
                                resolution.cannot_determine_from_image)
                        carried.update(observed)
                        # Sparse: only the slots whose origin is not the
                        # default. Everything else reads as ``CARRIED``, which
                        # is what every caller without a photograph has always
                        # produced and must keep producing.
                        origins.update({slot: Provenance.OBSERVED
                                        for slot in observed})
                    self._ask(question, audiences, reply, cid, carried,
                              origins, history, state, turn_index)
                    if images and boundary is not None:
                        note = ("The uploaded photograph was not analysed for this "
                                "conversation-state response and has supplied no facts. "
                                "Ask the technical team to review it.")
                        for _, answer in reply.parts:
                            answer.text += "\n\n" + note
                            answer.diagnostics["perception"] = {
                                **vision.disabled_report(), "summary": [note]}
                    summary["parts"] = len(reply.parts)
                    summary["paths"] = reply.paths
                    summary["refused"] = reply.refused
                # Written after the root span has closed, so the tree persisted
                # is the whole tree — and after `_ask` has left its read
                # snapshot, for the reason `log_answer` writes from outside one.
                self._record_spans(spans)
        return reply

    # ------------------------------------------------- the state machine

    def answer_part(self, part: str, audiences: tuple = ("public",),
                    carried: dict | None = None, origins: dict | None = None,
                    history: str = "", *, state=None,
                    turn_index: int = 0) -> Answer:
        """The per-topic answering path, under a name the graph may call.

        `_answer_part` stays private and unchanged; this is the seam. Naming it
        rather than letting `assistant/graph.py` reach for the underscore keeps
        the boundary visible: the graph orders the turn and calls this, and
        everything this does -- the policy gate, retrieval, the router, the six
        checks -- is the code that was already there.
        """
        if state is not None:
            carried = state.active()
            origins = state.provenance_of()
        boundary = self._state_answer(part, carried, origins, state, turn_index)
        if boundary is not None:
            return boundary
        # Cached, like the path it replaces. `Assistant._ask` looks the answer
        # up before calling `_answer_part` and stores it after, and moving the
        # UI onto the graph quietly left that behind -- decision 14's measured
        # 40.78s to 0.017s on a repeated question, lost, along with the
        # `cache_lookups` counter every dashboard reads.
        #
        # The key is the same function with the same inputs, so the two paths
        # cannot disagree about what counts as the same question: the audience
        # set, the snapshot, the model, the chunking version, the carried slots,
        # their origins and the transcript digest are all in it.
        snapshot = getattr(self, "_turn_snapshot", None)
        key = (self._cache_key(part, tuple(audiences), snapshot, carried,
                               origins, history)
               if (self.cache is not None and snapshot is not None) else None)
        if key is not None:
            with obs.span("cache_lookup") as lookup:
                cached = self.cache.get(key)
                lookup["hit"] = cached is not None
            if cached is not None:
                return replace(cached, diagnostics={**cached.diagnostics,
                                                    "cached": True})

        answer = self._answer_part(part, tuple(audiences), carried, origins,
                                   history)
        if key is not None:
            answer.diagnostics["snapshot_id"] = snapshot.snapshot_id
            answer.diagnostics["embedding_model"] = snapshot.embedding_model
            answer.diagnostics["chunking_version"] = snapshot.chunking_version
            self.cache.put(key, answer)
        return answer

    def _state_answer(self, question, carried=None, origins=None, state=None,
                      turn_index=0):
        """Use trusted state, never the transcript, before cache or model work.

        Legacy dictionaries represent user-stated facts unless an explicit
        origin says otherwise. A supplied ConversationState is authoritative,
        retaining conflicts and observation provenance rather than flattening it.
        """
        from .conversation import ConversationState, FactHistory, SessionFact
        from .understanding import state_only_answer

        if state is None:
            state = ConversationState(facts={
                slot: FactHistory(SessionFact(
                    slot, value, Provenance((origins or {}).get(
                        slot, Provenance.CARRIED))))
                for slot, value in (carried or {}).items()
            })
        return state_only_answer(
            question, self.router.slots, self.engine.names.get("products", []),
            state, turn_index, gate=self.router.gate)

    def recommend(self, resolved, decision, hits, history: str = "") -> Answer:
        return self.engine.recommend(resolved, decision, hits, history=history)

    def need_more_information(self, resolved, decision) -> Answer:
        return self.engine.need_more_information(resolved, decision)

    def no_supported_recommendation(self, resolved, decision,
                                    why: str = "") -> Answer:
        return self.engine.no_supported_recommendation(resolved, decision,
                                                       why=why)

    def _graph_for(self, vision, matrix, understanding: bool):
        """The application's one compiled graph, built once and kept.

        Built lazily rather than in `__init__` because compiling it imports
        `langgraph`, and a CLI answering a text question should not pay a 1.3
        second import for a path it does not take.

        `vision`, `matrix` and the registry are set on the services object each
        turn rather than baked in at compile time. The nodes close over the
        object and read the attributes when they run, so a graph compiled at
        start-up still sees this turn's snapshot -- which matters for the
        registry in particular, because check 5 trusts it and a product
        withdrawn since start-up must stop being recommendable.
        """
        from .graph import Services, build

        if self._services is None:
            self._services = Services(
                retriever=self.retriever, router=self.router, engine=self,
                repo=self.repo, registry=[], checkpointer=self.checkpointer)
            self._graph = build(self._services)
        self._services.vision = vision
        self._services.matrix = matrix
        self._services.understanding_enabled = understanding
        self._services.registry = self.engine.names.get("products", [])
        return self._graph

    def _polish(self, answer, question: str, final: dict):
        """A more direct wording of a finished answer, or the answer itself.

        Placed here, after the graph has returned and before the reply is
        assembled, because that is the one point at which an answer is both
        *final* -- every route has run, `verify` has passed, caveats are
        attached -- and not yet anybody's. Putting it inside a node would have
        made it a step the router could reach, and the whole claim of
        `assistant/phrasing.py` is that it is not one.

        The verifier handed over is the project's own `run_checks`, closed over
        the inputs this turn was judged on, so the rewrite is measured against
        the same checks as the original rather than against a second opinion.
        `asked_terms` is not reconstructed: `phrasing` compares the rewrite's
        failures with the original's, so an input this closure gets slightly
        wrong costs a rejected rewrite and never a wrong acceptance.
        """
        if not phrasing.enabled():
            return answer
        from .answer import products_named, run_checks

        hits = final.get("hits") or []
        resolved = final.get("resolved")
        registry = self.engine.names.get("products", [])
        scope = getattr(resolved, "product", "") or ""
        asked_products = tuple(products_named(question, registry))
        terms = list(getattr(resolved, "requested_properties", ()) or ())

        def verify(text: str) -> list[str]:
            return run_checks(text, hits, self.engine.names, terms,
                              product=scope, asked_products=asked_products,
                              question=question)

        return phrasing.polish_verified_answer(
            question, answer, registry=registry,
            # No evidence to verify against means no verifier to reuse, so the
            # deterministic preservation checks stand alone rather than a
            # vacuous `run_checks` being handed a passage list of nothing --
            # which would fail every sentence on check 1 and reject every
            # rewrite for the wrong reason.
            verify=verify if hits else None)

    @staticmethod
    def _paused_on(app, thread: str) -> bool:
        """Is this conversation stopped inside a node, waiting for a reply?

        Read from the checkpoint rather than tracked separately, so there is one
        answer to the question and it is the graph's. A second flag on the
        session would be a thing to keep in step, and the two would drift the
        first time a process restarted.
        """
        try:
            snapshot = app.get_state({"configurable": {"thread_id": thread}})
        except Exception:                                  # noqa: BLE001
            return False
        return bool(getattr(snapshot, "next", ()))

    def _render_interrupt(self, final: dict, thread: str) -> dict:
        """Turn a paused graph into something a surface can display.

        The graph stops mid-node and produces no answer, which is correct for
        the graph and useless to an HTTP response. This renders the question it
        stopped to ask, using the same `need_more_information` path a
        non-interruptible run would have taken -- so the wording, the provenance
        line and the contact details are identical either way, and only the
        machinery behind them differs.

        The pause itself is untouched. The conversation stays parked in the
        checkpoint, and the next message resumes it.
        """
        from .candidates import Outcome, RecommendationDecision

        payload = final["__interrupt__"][0]
        value = getattr(payload, "value", payload) or {}
        missing = tuple(value.get("missing") or ())
        decision = RecommendationDecision(
            outcome=Outcome.NEED_MORE_INFORMATION, missing=missing,
            reason="a product cannot be chosen without: " + ", ".join(missing))
        answer = self.engine.need_more_information(final["resolved"], decision)
        with obs.span("part",
                      question=obs.fingerprint(final["resolved"].raw_question)) as span:
            span["path"] = answer.path
            span["outcome"] = decision.outcome.value
            span["interrupted"] = True
        answer.diagnostics["interrupted"] = True
        answer.diagnostics["resuming"] = bool(value.get("resuming"))
        return {**final, "answer": answer, "decision": decision,
                # `answers` as well as `answer`: `ask_turn` prefers the list, so
                # setting only the singular left the stale one in place.
                "answers": [(final["resolved"].raw_question, answer)],
                "outcome": decision.outcome.value,
                "trace": list(final.get("trace") or []) + ["interrupted"]}

    def _thread_has_state(self, thread: str) -> bool:
        """Has this conversation been seen before?

        Asked of the checkpointer, which is the source of continuity now. The
        answer decides whether the caller's `ConversationState` is used to seed
        the thread or ignored -- and ignoring it on a known thread is the point:
        continuity must not depend on the caller remembering to hand state back.
        """
        try:
            return self.checkpointer.get(
                {"configurable": {"thread_id": thread}}) is not None
        except Exception:                                  # noqa: BLE001
            return False

    def conversation_state(self, thread: str):
        """The conversation this thread holds, as a `ConversationState`.

        A read-only view for a surface that wants to show what is remembered.
        The checkpoint is authoritative; this is a projection of it.
        """
        from .conversation import ConversationState

        try:
            saved = self.checkpointer.get({"configurable": {"thread_id": thread}})
        except Exception:                                  # noqa: BLE001
            saved = None
        if not saved:
            return ConversationState()
        values = saved.get("channel_values", {}) or {}
        return ConversationState(facts=self._conversation_facts(values),
                                 observations=tuple(values.get("observations") or ()),
                                 turn_index=values.get("turn_index", 0))

    @staticmethod
    def _conversation_facts(values):
        from .conversation import FactHistory, SessionFact

        facts = dict(values.get("facts") or {})
        topic = values.get("active_product")
        if topic and "product" not in facts:
            # Preserve the legacy active() view without claiming the caller
            # installed the product that their datasheet question named.
            facts["product"] = FactHistory(SessionFact(
                "product", topic, Provenance.ASSUMED, values.get("turn_index", 0)))
        return facts

    def forget(self, thread: str) -> None:
        """Drop a conversation entirely. What "New chat" means underneath."""
        try:
            self.checkpointer.delete_thread(thread)
        except Exception:                                  # noqa: BLE001
            pass

    def ask_turn(self, turn, state=None, *, vision=None, matrix=None,
                 understanding: bool = True):
        """One turn through the state machine. Returns (Reply, ConversationState).

        The opt-in entry point. `ask()` is untouched and every existing caller
        keeps its behaviour, because a new orchestration that silently replaced
        the working one would put seven routes at risk in order to add an
        eighth.

        Three things happen here that cannot happen inside a graph node. The
        **read snapshot is opened once for the whole turn**, so every node sees
        one release -- decision 19's pinned reader, which is what stops a
        publication mid-turn mixing two versions into one answer. The **name
        lists are refreshed from that snapshot**, because check 5 trusts them
        and a stale list would let a withdrawn product through. And the **span
        tree is collected and persisted** once the root span closes.
        """
        from .conversation import ConversationState

        state = state or ConversationState()
        reply = Reply(question=turn.raw_question, audiences=turn.audiences)
        final: dict = {}
        thread = turn.session_id or "cli"

        # The caller's id when it has one. `ask()` has always taken this and
        # `ask_turn` minted a fresh one instead, so the id the browser was shown
        # in `X-Correlation-Id` addressed a trace that did not exist.
        with obs.correlation(getattr(turn, "correlation_id", "")) as cid:
            reply.correlation_id = cid
            with obs.turn(turn=turn.turn_id or str(turn.turn_index),
                          session=turn.session_id,
                          source=self.source) as spans:
                with obs.span("answer", audiences=list(turn.audiences),
                              question_words=len(turn.raw_question.split()),
                              question=obs.fingerprint(turn.raw_question),
                              orchestration="graph") as summary:
                    with self.repo.read_snapshot() as snapshot:
                        self.retriever._verify()
                        # Held for the turn so `answer_part` can key the cache on
                        # the release it is actually reading, without the graph
                        # having to carry a snapshot through its state.
                        self._turn_snapshot = snapshot
                        self.engine.names = {
                            key: snapshot.notes.get(key, default)
                            for key, default in (("products", []),
                                                 ("colours", []),
                                                 ("merchants", []),
                                                 ("contact", {}))}
                        app = self._graph_for(vision, matrix, understanding)
                        paused = self._paused_on(app, thread)
                        if paused and (self.router.gate.match(turn.raw_question)
                                       or not self.router.slots.is_answer_to_askback(
                                           turn.raw_question)):
                            # Parked waiting for a substrate, and this message is
                            # a new question rather than an answer to that. The
                            # person moved on, and resuming would feed their
                            # question in as the reply -- discarding it and
                            # re-asking the same thing, which is precisely the
                            # bug the `pending` string used to have and the
                            # reason `interrupt()` is not automatically safer.
                            #
                            # The paused turn is abandoned and the conversation
                            # kept: the facts are read out, the thread dropped,
                            # and a fresh turn seeded with them.
                            carried = self.conversation_state(thread)
                            self.forget(thread)
                            obs.event("ask_back_abandoned",
                                      kept=sorted(carried.active()))
                            state = carried
                            paused = False
                        # The facts are seeded from the checkpoint, not from the
                        # caller. `inherit()` is applied only when this thread
                        # has no checkpoint yet -- a first turn, or a caller
                        # deliberately restoring a conversation from elsewhere.
                        # Sending it every turn would re-apply the whole prior
                        # state on top of itself and make `source_turn` reset.
                        seed = {"raw_question": turn.raw_question,
                                "history": getattr(turn, "history", ""),
                                "images": list(turn.images),
                                "audiences": tuple(turn.audiences),
                                "turn_index": turn.turn_index,
                                "session_id": turn.session_id}
                        if not self._thread_has_state(thread):
                            seed["facts"] = state.inherit()
                        config = {"configurable": {"thread_id": thread}}
                        if paused:
                            # This conversation is stopped inside `ask_back`
                            # waiting for an answer, so this message *is* the
                            # answer. Resuming re-enters the node it stopped in
                            # and carries on to retrieval, evidence and a
                            # recommendation -- answering the question they
                            # originally asked, which is the whole point of
                            # pausing rather than starting again.
                            from langgraph.types import Command
                            final = app.invoke(
                                Command(resume=turn.raw_question), config)
                        else:
                            final = app.invoke(seed, config)
                        if final.get("__interrupt__"):
                            final = self._render_interrupt(final, thread)

                    # `answers` rather than `answer`: a message carrying two
                    # jobs is two parts, the same way `Assistant.ask` has always
                    # treated it. A selection produces one.
                    produced = final.get("answers") or (
                        [(turn.raw_question, final["answer"])]
                        if final.get("answer") is not None else [])
                    resolved = final.get("resolved")
                    resumed = final.get("resumed_question") or ""
                    if resumed:
                        reply.question = resumed
                    # What this turn's photographs showed, attached to every
                    # part so a surface can print it beside the answer without
                    # reaching into the graph. Set only when there was an
                    # upload, so its presence *is* the signal that a photograph
                    # was read -- a page can distinguish "looked and saw
                    # nothing" from "was sent nothing".
                    perception = final.get("perception") or {}
                    produced = [(part, self._polish(answer, part, final))
                                for part, answer in produced]
                    for part, answer in produced:
                        answer.diagnostics["correlation_id"] = cid
                        answer.diagnostics["trace_id"] = obs.trace_id()
                        if perception:
                            answer.diagnostics["perception"] = perception
                        if resolved is not None:
                            answer.diagnostics.setdefault(
                                "intent", resolved.intent.value)
                        answer.diagnostics["graph"] = final.get("trace", [])
                        if resumed:
                            answer.diagnostics["resumed_question"] = resumed
                        reply.parts.append((resumed or part, answer))
                    summary["parts"] = len(reply.parts)
                    summary["paths"] = reply.paths
                    summary["refused"] = reply.refused
                self._record_spans(spans)

        if self.log:
            self._log(reply)

        # A case boundary is retired here rather than inside the graph. The
        # graph's `NewCase` instruction resets the *facts channel*, which is
        # what stops the old wall informing the new one; filing the old case in
        # `history` is a property of the conversation object the caller holds,
        # and the graph does not own that. Both halves are needed: without the
        # reducer the new wall inherits brick, and without this the retired wall
        # is simply gone and the earlier answers stop being explicable.
        opened = final.get("case_opened")
        if opened:
            state.open_case(turn.turn_index, because=opened)
        state.facts = self._conversation_facts(final)
        state.observations = tuple(final.get("observations") or ())
        state.turn_index = turn.turn_index
        return reply, state

    def _record_spans(self, spans: list) -> None:
        """Persist the turn's trace, and never let a trace cost an answer.

        Three layers of not-failing, and each is there for a different reason.
        The adapters swallow their own write errors, because a store that cannot
        record a timing has cost the operator a debugging aid and the caller
        nothing. This also tolerates a repository that has no `record_spans` at
        all, because the engine is built against a Protocol and a test double
        that predates this method must keep answering. And the call itself is
        wrapped, because the one remaining way a trace could break an answer is
        an adapter nobody has written yet.

        It follows `self.log` rather than a switch of its own. A caller that has
        turned off recording has turned off recording, and two flags would let
        a surface leak spans from a run it thought it had silenced.
        """
        if not self.log or not spans:
            return
        recorder = getattr(self.repo, "record_spans", None)
        if recorder is None:
            return
        try:
            recorder(spans)
        except Exception as error:                     # noqa: BLE001
            obs.event("store_error", operation="record_spans",
                      error=type(error).__name__, detail=str(error))

    def _ask(self, question: str, audiences: tuple[str, ...], reply: Reply,
             cid: str, carried: dict | None = None,
             origins: dict | None = None, history: str = "", state=None,
             turn_index: int = 0) -> None:
        from .understanding import MANDATORY_TOPICS

        with self.repo.read_snapshot() as snapshot:
            self.retriever._verify()
            # Names/contact are release metadata too; refresh them together
            # with the passages rather than retaining the startup snapshot.
            self.engine.names = {key: snapshot.notes.get(key, default) for key, default in
                                 (("products", []), ("colours", []), ("merchants", []), ("contact", {}))}
            with obs.span("split_by_topic") as split:
                matched = self.router.gate.match(question)
                # Do not answer a meta/state fragment before the safety request.
                parts = ([question] if matched and matched[0] in MANDATORY_TOPICS
                         else split_by_topic(question))
                split["parts"] = len(parts)
            for part in parts:
                # One span per topic, and every stage below it hangs off this
                # one, so a two-topic question reads as two trees rather than
                # as one interleaved list.
                with obs.span("part", question=obs.fingerprint(part),
                              snapshot_id=snapshot.snapshot_id) as part_span:
                    answer = self._state_answer(
                        part, carried, origins, state, turn_index)
                    key = (self._cache_key(part, audiences, snapshot, carried,
                                           origins, history)
                           if answer is None else None)
                    # `is not None`, not truthiness. AnswerCache defines __len__,
                    # so an empty cache is falsy and `if self.cache` was False on
                    # every call — the cache could never fill, because it was empty.
                    cached = False
                    if answer is None:
                        with obs.span("cache_lookup") as lookup:
                            answer = (self.cache.get(key)
                                      if self.cache is not None else None)
                            cached = answer is not None
                            lookup["hit"] = cached
                    part_span["cached"] = cached
                    if answer is None:
                        answer = self._answer_part(part, audiences, carried,
                                                   origins, history)
                        answer.diagnostics["snapshot_id"] = snapshot.snapshot_id
                        answer.diagnostics["embedding_model"] = snapshot.embedding_model
                        answer.diagnostics["chunking_version"] = snapshot.chunking_version
                        if self.cache is not None:
                            self.cache.put(key, answer)
                    elif cached:
                        # Copy before annotating. The cache holds one Answer and
                        # hands the same object to every caller, so writing this
                        # request's correlation id onto it overwrites the last
                        # reader's — two concurrent callers on the threading server
                        # would each find the other's trace in their diagnostics.
                        # The answer text was never at risk; the ability to trace it
                        # was, which is exactly what the id is for.
                        answer = replace(answer, diagnostics={**answer.diagnostics,
                                                              "cached": True})
                    # Carried on the answer as well as in the log, so a diagnostics
                    # dump and a log line can be joined without the store.
                    answer.diagnostics["correlation_id"] = cid
                    answer.diagnostics["trace_id"] = obs.trace_id()
                    answer.diagnostics["turn_id"] = obs.turn_id()
                    part_span["path"] = answer.path
                    reply.parts.append((part, answer))

        # Logged outside the read snapshot, deliberately. SQLite's snapshot ends
        # in a rollback and the Postgres one is REPEATABLE READ READ ONLY, so an
        # insert inside either is lost or raises.
        if self.log:
            self._log(reply)

    def _missed_evidence(self, part: str, hits: list, named: str,
                         audiences: tuple[str, ...]) -> list:
        """Passages about something the question named and the evidence lacks.

        Semantic retrieval answers the question as a whole, which means a
        specific word in it can end up represented by nothing. Two real cases,
        both of which had the answer sitting in the corpus unretrieved:

          "can I use Solo over old **gypsum** plaster"  — the Solo Primer page
          lists "historic or new plasters (lime, cement or gypsum)" as suitable
          backgrounds, and not one retrieved passage contained the word.

          "my render is showing **patchy** colour"  — the Colour and Colour
          Consistency article is indexed and discusses exactly this, and the
          five passages retrieved were product marketing.

        So: any word the question uses that appears in none of the retrieved
        passages is looked up lexically. Distinctiveness is measured rather than
        guessed — a word is worth chasing only if few passages contain it. Ask
        for one more than the cap and a common word comes back full, which is
        the signal to drop it; "gypsum" returns a handful and "wall" does not.
        That keeps the rule self-tuning and needs no vocabulary of its own.

        These carry no similarity score, so they sort last and cannot become the
        top hit the abstention threshold is measured against.
        """
        blob = " ".join(f"{h.chunk.section} {h.chunk.content}" for h in hits).lower()
        known = {(h.chunk.canonical_url, h.chunk.chunk_index) for h in hits}

        # Score every candidate word by how few passages carry it, then take
        # only the rarest. Rarity is the whole signal: "gypsum" names a specific
        # background and "patchy" names a specific defect, while "plaster" names
        # the catalogue. Adding everything that matched turned a five-passage
        # answer into a fourteen-passage one and buried the evidence it found.
        scored: list[tuple[int, str, list]] = []
        for word in dict.fromkeys(re.findall(r"[a-z]{5,}", part.lower())):
            if word in blob or word in _QUESTION_NOISE:
                continue
            # Scoped to the named product and across the corpus: each recovers
            # what the other misses. Scoping finds the Solo Primer page for
            # "gypsum", which authority ordering otherwise buries; not scoping
            # finds the Colour and Colour Consistency article for "patchy",
            # which carries no product tag for a scope to match.
            found = self.retriever.find_property(
                named, (word,), audiences=audiences, limit=DISTINCTIVE_CAP + 1)
            if not found:
                # Only when the product scope found nothing. Searching both and
                # merging looked more thorough and was worse: "gypsum" returns
                # three Solo passages scoped and seven others unscoped, so the
                # union broke the rarity cap and the Solo Primer page — the one
                # that answers the question — was discarded with the rest.
                found = self.retriever.find_property(
                    "", (word,), audiences=audiences, limit=DISTINCTIVE_CAP + 1)
            unique = {(h.chunk.canonical_url, h.chunk.chunk_index): h
                      for h in found}
            if unique and len(unique) <= DISTINCTIVE_CAP:
                scored.append((len(unique), word, list(unique.values())))

        # A rare word in common is necessary and not sufficient. The passage has
        # to be about the question too, or a word that happens to appear in an
        # unrelated document drags it in — "my render is showing patchy colour"
        # pulled in a Warmshell wind barrier and a page on breathability, which
        # the diagnosis path would then have quoted at the reader as published
        # guidance on their problem.
        asked = _content_words(part)
        added: list = []
        for _count, word, passages in sorted(scored, key=lambda t: t[0]):
            for h in passages:
                key = (h.chunk.canonical_url, h.chunk.chunk_index)
                if key in known or len(added) >= MISSED_EVIDENCE_LIMIT:
                    continue
                shared = _content_words(f"{h.chunk.section} {h.chunk.content}") & asked
                if len(shared - {word}) < RELATED_WORDS:
                    continue
                known.add(key)
                added.append(h)
        return added

    def _named_product(self, part: str, carried: dict | None = None) -> str:
        """The product this question names, or the one from prior turns.

        Precedence is explicit name first, carried name second, nothing third.
        A question that names a product means that product even when an earlier
        turn established another, so "what about Forte instead?" switches; a
        question that names none inherits, which is what makes "how much for 30
        square metres" a question about the product under discussion rather than
        about the corpus at large.

        Matched against the harvested product list rather than guessed, so it
        cannot invent a product — the same list check 5 uses to refuse invented
        names. The longest match wins, because "Lime Green Solo Onecoat" and
        "Solo" are both in the list and the specific one is the one meant.

        Retrieval treats this as a bounded boost rather than a filter, so a
        wrong detection reorders and never refuses: the worst case is the same
        answer in a different order, not a silent loss.
        """
        named = self._named_products(part)
        if len(named) > 1:
            return ""

        # If question names a product, use it. Otherwise fall back to carried.
        if not named:
            if carried and "product" in carried:
                return carried["product"]
            return ""

        # Longest wins, then the maker's name comes off. Both halves matter.
        # "Lime Green Ultra" and "Ultra" are both harvested names, so the
        # longest match is the brand-prefixed one — and chunks are tagged with
        # the catalogue name, "Ultra Insulated Lime Render Base Coat", which
        # does not contain "Lime Green" and is not contained by it. Matching is
        # containment either way, so the prefixed form matched nothing at all:
        # the product boost and the targeted coverage lookup were both silently
        # no-ops for every question that named a product the way the website
        # writes it. "Ultra" finds three coverage passages; "Lime Green Ultra"
        # finds none, which is how a quantity question came to say the coverage
        # figure was unpublished while it sat in the datasheet.
        best = max(named, key=len).lower()
        return best.removeprefix(_BRAND).strip() or best

    def _named_products(self, part: str) -> list[str]:
        from .understanding import reference_text

        aliases = _product_aliases(self.engine.names.get("products", []))
        return sorted(_named_aliases(reference_text(part), aliases))

    def _property_evidence(self, part, hits, products, detected, audiences):
        """Bounded lexical recovery per requested product and field.

        Keep the repository's scores (lexical hits are zero), audience and
        active-version filtering. These passages never rescue a weak semantic
        score; they only supply evidence the unchanged router can inspect.
        """
        fields = _requested_fields(
            part, Decision(Path_.EXTRACT, "retrieval fields", slots=detected))
        known = {(h.chunk.canonical_url, h.chunk.chunk_index) for h in hits}
        added = []
        for product in products:
            for prop in fields:
                terms = PROPERTY_TERMS.get(prop)
                if not terms:
                    continue
                for hit in self.retriever.find_property(
                        product, terms, audiences=audiences,
                        limit=GATE_SECOND_CHANCE):
                    key = (hit.chunk.canonical_url, hit.chunk.chunk_index)
                    if key not in known:
                        known.add(key)
                        added.append(hit)
        if added:
            obs.event("targeted_retrieval", products=products, added=len(added),
                      reason="requested product fields")
        return hits + added

    def _cache_key(self, part: str, audiences: tuple[str, ...], snapshot,
                   carried: dict | None = None, origins: dict | None = None,
                   history: str = ""):
        """Everything that can change the right answer to the same words.

        The origins belong in the key as much as the values do, and the reason
        is not performance. "brick" as a substrate produces one printed sentence
        when the caller said it and a different one when a photograph was read
        for it — "as you told me earlier in this conversation" against "from
        the photograph you sent". Those are two answers, and serving one to the
        other's caller would tell somebody they had said something they never
        said: the exact defect the provenance distinction exists to prevent,
        reintroduced through a cache.
        """
        return AnswerCache.key(part, audiences, snapshot.snapshot_id,
                               ollama.GENERATION_MODEL, snapshot.chunking_version,
                               carried, origins, history)

    def _log(self, reply: Reply) -> None:
        """Record each part, and never let recording break the answer.

        A store that cannot write the log still has a perfectly good answer in
        hand. Losing it to an audit failure would invert the priority the rest
        of the design is built on.
        """
        for part, answer in reply.parts:
            entry = AnswerLogEntry(
                question=part,
                path_taken=answer.path,
                audiences=reply.audiences,
                snapshot_id=answer.diagnostics.get("snapshot_id", ""),
                chunk_ids=list(answer.diagnostics.get("chunk_ids", [])),
                generation_model=(ollama.GENERATION_MODEL
                                  if answer.diagnostics.get("generation_seconds")
                                  else ""),
                check_failed="; ".join(answer.failed_checks),
                source=self.source,
            )
            try:
                self.repo.log_answer(entry)
            except Exception as error:
                # Never let an audit failure cost a good answer — but never let
                # it pass unnoticed either. `CLAUDE.md`: do not silently swallow
                # failures. The answer survives; the operator finds out.
                obs.event("store_error", operation="log_answer",
                          error=type(error).__name__, detail=str(error))

    def _answer_part(self, part: str, audiences: tuple[str, ...],
                     carried: dict | None = None,
                     origins: dict | None = None, history: str = "") -> Answer:
        matched = self.router.gate.match(part)
        if matched:
            topic, spec = matched
            # The policy gate is a routing decision like any other, and it is
            # the one that keeps the model away from prices and stock. It is
            # worth being able to count how often it fires.
            with obs.span("route", path=Path_.ROUTE.value, topic=topic,
                          step="gate", reason="policy gate"):
                pass
            if spec.get("from_manifest"):
                return self.engine.documents_for(part, audiences)
            return self.engine.route(topic, spec)

        explicit = self._named_products(part)
        named = self._named_product(part, carried=carried)
        carried = dict(carried or {})
        if explicit:
            carried.pop("product", None)
        if named:
            carried["product"] = named
            from .understanding import stated_product

            origins = dict(origins or {})
            if explicit:
                origins["product"] = (
                    Provenance.STATED
                    if stated_product(part, self.engine.names.get("products", [])) == named
                    else Provenance.ASSUMED)
        products = explicit or ([named] if named else [])
        query = f"{named}: {part}" if named and not explicit else part
        hits = self.retriever.search(query, audiences=audiences, product=named)
        # A quantity question embeds as a question about quantity, so the
        # coverage figure it needs may not be in the top five at all — and a
        # path that only re-sorts what it was given cannot recover from that.
        # Ask a second time by metadata instead: the named product, and the
        # words coverage is printed under.
        #
        # Before routing, not after. The relevance gate is step 4 and the
        # calculation branch is step 6, so a coverage passage added afterwards
        # arrives too late to stop "does not state bags for this product" —
        # which is what the first version of this did, refusing a question it
        # had the evidence to answer.
        #
        # These hits carry no similarity score, so they sort last and cannot
        # become the top hit that step 1 measures against the threshold.
        with obs.span("slot_detection") as detection:
            # Detected once and reused. `detect` is pure, so calling it twice
            # would cost a second pass over the vocabulary to learn the same
            # thing — and a span around a call whose result is thrown away
            # would be a timing of something the answer does not depend on.
            # Names only: a slot *value* is a phrase out of the question.
            detected = self.router.slots.detect(part)
            detection["slots"] = sorted(detected)
        fields = _requested_fields(
            part, Decision(Path_.EXTRACT, "retrieval fields", slots=detected))
        if products and fields and not self.retriever.above_threshold(hits):
            # A short pronoun question can embed far from its datasheet even
            # with a metadata boost. Retry with explicit field context, without
            # changing the routed question or inventing a similarity score.
            for product in products:
                terms = dict.fromkeys(
                    term for prop in fields for term in PROPERTY_TERMS.get(prop, ()))
                if not terms:
                    continue
                contextual = self.retriever.search(
                    f"{product} {' '.join(terms)}. {part}",
                    audiences=audiences, product=product)
                if contextual and (not hits or contextual[0].score > hits[0].score):
                    known = {(h.chunk.canonical_url, h.chunk.chunk_index)
                             for h in contextual}
                    hits = contextual + [
                        h for h in hits
                        if (h.chunk.canonical_url, h.chunk.chunk_index) not in known]
        hits = self._property_evidence(part, hits, products, detected, audiences)
        if named and "calculation" in detected:
            known = {(h.chunk.canonical_url, h.chunk.chunk_index) for h in hits}
            found = [h for h in self.retriever.find_property(
                        named, COVERAGE_TERMS, audiences=audiences)
                     if (h.chunk.canonical_url, h.chunk.chunk_index) not in known]
            if found:
                hits = hits + found
                obs.event("targeted_retrieval", product=named, added=len(found),
                          reason="coverage for a quantity question")

        # One more pass for anything the question named that the evidence does
        # not contain. Runs before routing, because the relevance gate and the
        # diagnosis branch both read `hits` and both were deciding on evidence
        # that was missing the word the question turned on.
        missed = self._missed_evidence(part, hits, named, audiences)
        if missed:
            hits = hits + missed
            obs.event("missed_evidence", added=len(missed), product=named,
                      reason="the question named something no passage contained")

        # The relevance gate's own second chance, which `Router.unsupported_
        # terms` has always existed to offer and which nothing has ever called.
        # Its docstring says the case exactly: "semantic retrieval having failed
        # to surface the term is exactly the moment a lexical lookup is worth
        # doing: the corpus may well publish the answer in a passage the
        # embedding did not rank."
        #
        # Measured, on a question a customer asks constantly. "What is the
        # coverage of Solo Onecoat?" refused at step 4, and the Solo datasheet
        # publishes "Each bag will cover approximately 1.5m2 at 10mm thick, or
        # 3m2 at 5mm thick" -- in a section headed "Storage / Coverage" whose
        # text is mostly about storage, so a bare coverage question embeds away
        # from it and it ranked tenth. Neither existing second pass recovers it:
        # the coverage lookup above fires only on calculation words, which this
        # question has none of, and `_missed_evidence` drops "coverage" as
        # insufficiently distinctive, which it is -- every datasheet has a
        # coverage section. Both are right about their own rule and the answer
        # still went missing between them.
        #
        # Scoped to the product the question named, and skipped entirely when it
        # named none: an unscoped lexical sweep for "coverage" returns every
        # datasheet's coverage section, which is the cross-product contamination
        # this system spends a check catching. A question with no product and no
        # retrieved evidence for its term is a question the gate should refuse.
        #
        # This buys coverage and spends no safety. The gate still refuses when
        # the lookup finds nothing, which is what keeps the near-miss refusals
        # refusing: nothing in the corpus states a U-value for Solo, so nothing
        # comes back and step 4 fires exactly as before.
        unsupported = self.router.unsupported_terms(part, hits, carried)
        if unsupported and products:
            known = {(h.chunk.canonical_url, h.chunk.chunk_index) for h in hits}
            recovered = [
                h for product in products for h in self.retriever.find_property(
                    product, tuple(unsupported), audiences=audiences,
                    limit=GATE_SECOND_CHANCE)
                if (h.chunk.canonical_url, h.chunk.chunk_index) not in known]
            if recovered:
                hits = hits + recovered
                obs.event("gate_second_chance", added=len(recovered),
                          product=named, terms=len(unsupported),
                          reason="the asked-for term was in no retrieved passage")

        with obs.span("route") as routing:
            decision = self.router.route(
                part, hits, self.retriever.above_threshold(hits), audiences,
                carried)
            routing["path"] = decision.path.value
            routing["step"] = decision.step
            routing["reason"] = decision.reason
            routing["top_score"] = hits[0].score if hits else 0.0
            routing["slots"] = sorted(decision.slots)
        # The product the question named, on the decision rather than only in
        # the diagnostics written at the end. The evidence binding needs it:
        # without it, a claim about Ultra can be bound to whichever passage
        # states the property most directly, and a Warmshell system guide
        # states thicknesses and suitability as clearly as anything Ultra
        # publishes. The graph path already supplied it through `carried`; this
        # makes the two paths agree rather than leaving the binding weaker on
        # whichever one a caller happens to be on.
        #
        # `setdefault`, so the router's own merge still wins: `carried` is
        # merged *under* what the question says, and this must not overturn it.
        if named:
            decision.slots.setdefault("product", named)
        # Annotated after the routing decision rather than passed into it,
        # because nothing in the router reads an origin and nothing in it
        # should: the path a question takes must not depend on whether a
        # substrate was typed or photographed, and a router that could see the
        # difference would be a second answer route wearing a data field.
        #
        # The comparison against `carried` is what keeps the annotation honest.
        # The router merges the question over everything handed in, so a slot
        # the question also stated holds the question's value, and marking that
        # one as observed would credit the photograph with something the person
        # typed. Only a slot still carrying the value vision supplied is
        # annotated; the rest fall through to `_facts`'s own detection.
        decision.origins = {
            slot: origin for slot, origin in (origins or {}).items()
            if decision.slots.get(slot) == (carried or {}).get(slot)
        }
        produce = {
            # Every path gets the question, not just Compose. Without it the
            # other five cannot tell a slot the caller stated in this sentence
            # from one carried out of an earlier turn, so they fell back to
            # reporting both as "as you said" — which is false of a carried
            # value and reads as the system putting words in someone's mouth.
            # It showed up on a quantity question that reported "external
            # (location), as you said" when external had come from a question
            # two turns earlier about render.
            Path_.EXTRACT: lambda: self.engine.extract(decision, part),
            Path_.COMPOSE: lambda: (
                self.engine.factual(decision, part)
                or self.engine.compose(decision, part, history=history)),
            Path_.DEFER: lambda: self.engine.defer(decision, part),
            Path_.DIAGNOSIS: lambda: self.engine.diagnosis(decision, part),
            Path_.ASK_BACK: lambda: self.engine.ask_back(decision, part),
            Path_.REFUSE: lambda: self.engine.refuse(decision, decision.reason, part),
        }[decision.path]
        # `render` covers turning the decision into the reply a person reads:
        # the passage or the composed text, its sources, the caveats code
        # appends and the disclosure a surface may fold away. Counts only — the
        # sources are documents, not their text.
        with obs.span("render", path=decision.path.value) as rendering:
            answer = produce()
            rendering["sources"] = len(answer.sources)
            rendering["caveats"] = len(answer.caveats)
            rendering["disclosure"] = bool(answer.disclosure)
            rendering["refused"] = answer.refused

        # Carry detected slots into answer diagnostics so session can persist them.
        # Include product separately since it's detected by _named_product, not router.slots.
        slots = dict(decision.slots)
        if named:
            slots["product"] = named
        for fact in answer.facts:
            if fact.provenance is Provenance.ASSUMED:
                slots.pop(fact.slot, None)
        if decision.origins.get("product") is Provenance.ASSUMED:
            slots.pop("product", None)
        answer.diagnostics["slots"] = slots
        return answer


# ------------------------------------------------------------------ rendering


def render(reply: Reply, show_diagnostics: bool = False) -> str:
    """The composite reply as a person reads it."""
    blocks: list[str] = []
    multi = len(reply.parts) > 1

    for i, (part, answer) in enumerate(reply.parts, 1):
        head = f"── Part {i}: {part}" if multi else ""
        body = [head, ""] if head else []
        body.append(answer.text)

        if answer.assumptions:
            body += ["", "Assumed: " + "; ".join(answer.assumptions)]

        if answer.caveats:
            body += ["", "Also published about these products:"]
            body += [f"  • {c}" for c in answer.caveats]

        if answer.sources:
            body += ["", "Sources:"]
            for s in answer.sources:
                date = f", {s['date']}" if s["date"] else ""
                body.append(f"  [{s['marker']}] {s['name']}"
                            f"{' — ' + s['section'] if s['section'] else ''}{date}")
                body.append(f"      {s['url']}")

        if show_diagnostics:
            d = answer.diagnostics
            body += ["", f"  [path: {answer.path} · step {d.get('step', '-')} · "
                         f"top score {d.get('top_score', 0)}]",
                     f"  [why: {d.get('reason', d.get('refusal_reason', '-'))}]"]
            if d.get("slots"):
                body.append(f"  [slots: {d['slots']}]")
            if answer.failed_checks:
                body.append("  [checks that failed:]")
                body += [f"    - {c}" for c in answer.failed_checks]
            if d.get("generation_seconds"):
                body.append(f"  [generation: {d['generation_seconds']}s]")

        blocks.append("\n".join(body))

    return "\n\n".join(blocks)
