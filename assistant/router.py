"""The router: policy gate, slot detection, and eight ordered decisions.

This is the part of the system that is deliberately not a model. Every branch
below is a rule someone can read, disagree with, and change, and the ordering
is itself a decision — decision 8 puts the retrieval-shape steps first, because
whether anything was found at all has to be settled before what was found can
be interpreted.

The single most important line is step 2. If the top passage itself says "ask
us", that published deferral beats any quantity the system could assemble. The
company has already decided that question needs a person, and an assistant that
answers around its own company's referral is worse than one that does not
answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .model import Retrieved

CONFIG = Path(__file__).resolve().parents[1] / "config"


class Path_(str, Enum):
    """The five outcomes, the two composites the diagram names, and SELECT.

    ``SELECT`` is a printing path like the others and not a second pipeline. It
    exists because a recommendation has a shape the other paths do not: the
    model is allowed to choose between products and explain the runner-up, but
    only from a set `assistant/candidates.py` has already approved on published
    evidence, and the answer is checked afterwards for having stayed inside it.

    Being a member here rather than a flag on Compose is what keeps the
    containment check attached. A recommendation printed as `compose` would be
    indistinguishable, in the log and in the trace, from an ordinary composed
    answer that never had an approved set to stay inside.
    """

    ROUTE = "route"                 # fixed referral, no retrieval
    EXTRACT = "extract"             # a passage printed verbatim by code
    COMPOSE = "compose"             # the model, over retrieved passages
    DEFER = "cited hand-off"        # a published deferral, quoted
    DIAGNOSIS = "diagnosis"         # published causes plus a hand-off
    ASK_BACK = "ask back"           # a load-bearing slot is uncued
    SELECT = "select"               # a product chosen from an approved set
    REFUSE = "refuse"


@dataclass
class Decision:
    """Why this question went the way it did. Printed with every answer."""

    path: Path_
    reason: str
    step: str = ""
    topic: str = ""
    slots: dict = field(default_factory=dict)
    # Where a slot value came from, for the slots whose origin is not the
    # default. Sparse and written by `assistant/engine.py` after this decision
    # is made, never by the routing code below: nothing in the router reads it,
    # nothing in the router should, and a routing decision that depended on
    # whether a substrate was typed or photographed would be a second answer
    # path wearing a data field. It exists because `AnswerEngine._facts` cannot
    # re-derive an origin from the question — the defining property of a
    # carried or observed slot is that it is absent from the question — so the
    # origin has to travel beside the value. An empty mapping means every slot
    # is read exactly as it was before this field existed.
    origins: dict = field(default_factory=dict)
    hits: list = field(default_factory=list)
    missing_term: str = ""
    # Every word the relevance gate will accept as "the thing that was asked
    # about", computed once at step 4 and carried so that check 6 asks the same
    # question step 4 asked.
    #
    # It exists because the two halves of one gate had drifted apart. Step 4
    # ORs across every property `detect_properties` finds, on the stated
    # reasoning that a two-property question should be answered for the half
    # the corpus publishes rather than refused whole. Check 6 re-derived its
    # terms from `slots["property_asked"]`, which is the single best-scoring
    # property, so the other halves were invisible to it. "Which finish coats
    # are compatible with Forte render, including the hardening time required
    # first?" therefore passed step 4 on *coat* and *finish* and was then
    # refused by check 6 for want of *compatible* — a fully published,
    # correctly retrieved, two-document answer thrown away by the gate meant to
    # catch the near-miss. Evaluation situation S8, reproducibly.
    #
    # Carrying the terms rather than recomputing them is the fix that cannot
    # drift again: there is one definition, in `_asked_terms`, and both ends of
    # the gate read the same list. Empty means the gate has no opinion, which is
    # what steps 1 to 3 hand on, and check 6 skips itself on an empty list
    # exactly as it did before.
    asked_terms: list = field(default_factory=list)
    per_option: bool = False
    sum_refused: bool = False
    photograph: bool = False


def _load(name: str) -> dict:
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ splitting

# Split after a question mark or full stop, and before a conjunction that
# introduces a second question. The lookahead keeps the question word with the
# part it belongs to — splitting on "and what" swallows the "what" and leaves a
# fragment that retrieves badly.
_SPLIT = re.compile(
    r"(?<=[?.;])\s+"
    r"|\s+(?:and|also|plus)[,]?\s+(?=(?:what|how|can|could|do|does|is|are|which|"
    r"when|where|why|should|will|would)\b)",
    re.I,
)


def split_by_topic(question: str) -> list[str]:
    """Two *jobs* in one message are two questions. Two sentences are not.

    This used to cut after every full stop, which is the single most damaging
    thing the pipeline did to a real enquiry. Someone writing

        "My external lime render is showing patchy colour after drying.
         What could be causing it?"

    had the second half separated from every fact that made it answerable —
    external, lime render, patchy colour, drying — and "What could be causing
    it?" retrieved against nothing. The same cut destroyed a question about a
    brick wall and Ultra's thickness, and one about exposure and render
    preparation. The message was never ambiguous; the splitter made it so.

    So the rule is now the opposite one: **keep the message together unless
    there is positive evidence of two independent jobs**. Under-splitting is
    much the safer error here. A single part carrying both halves still gets
    routed, retrieved for and checked; a part stripped of its context gets
    neither good retrieval nor an honest refusal, because the system cannot
    even tell what was asked.

    Positive evidence means the two halves want different *paths*, not merely
    different words — a price and a coverage, where one must reach the policy
    gate and the other must reach retrieval. Answering that pair as a unit
    sends one of them down the wrong path, which is the case the splitter was
    written for and the only one it now fires on.

    The cost is real and worth stating: a message genuinely containing two
    retrieval questions is now answered as one, which may favour whichever half
    retrieves more strongly. That is a worse answer. The alternative was a
    confidently irrelevant one, or a refusal that could not say what it had
    been asked.
    """
    candidates = [p.strip(" ,;") for p in _SPLIT.split(question)
                  if p and p.strip(" ,;")]
    candidates = [p for p in candidates if len(p.split()) >= 3]
    if len(candidates) < 2:
        return candidates or [question.strip()]

    # Split only where the halves belong to different paths. A policy topic
    # beside a technical question is the case that matters: the policy half
    # must never reach retrieval, and the technical half must never be answered
    # from a referral. Everything else stays whole.
    gate = PolicyGate()
    gated = [bool(gate.match(p)) for p in candidates]
    if len(set(gated)) > 1:
        return candidates

    # Two policy topics are also two jobs — a price and a delivery question get
    # different referrals, and merging them would print one and drop the other.
    if all(gated):
        topics = {gate.match(p)[0] for p in candidates}
        if len(topics) > 1:
            return candidates

    return [question.strip()]


# ---------------------------------------------------------------- policy gate


class PolicyGate:
    """Topics the published corpus cannot answer, matched before retrieval."""

    def __init__(self) -> None:
        cfg = _load("routing.json")["topics"]
        self.topics = {k: v for k, v in cfg.items() if not k.startswith("_")}
        self.compiled = {
            name: [re.compile(p, re.I) for p in spec["patterns"]]
            for name, spec in self.topics.items()
        }

    def match(self, question: str) -> tuple[str, dict] | None:
        for name, patterns in self.compiled.items():
            if any(p.search(question) for p in patterns):
                return name, self.topics[name]
        return None


# ------------------------------------------------------------- slot detection


class SlotDetector:
    """Vocabulary matching, with synonyms, over the eight slots."""

    def __init__(self) -> None:
        self.spec = _load("vocabularies.json")["slots"]

    def detect(self, question: str) -> dict:
        """The best-matching value per slot, scored by how much of it matched.

        First-match-wins reads "how much water does Solo need per bag" as a
        coverage question, because "per bag" is a coverage term and coverage
        happens to be declared first. Scoring by matched-term length picks the
        longer, more specific evidence, so "how much water" beats "per bag".
        """
        lowered = question.lower()
        found: dict[str, str] = {}
        for slot, spec in self.spec.items():
            if slot.startswith("_"):
                continue
            scored = []
            for value, terms in spec["values"].items():
                score = sum(
                    len(t) for t in terms
                    if re.search(rf"\b{re.escape(t)}\b", lowered)
                )
                if score:
                    scored.append((score, value))
            if not scored:
                continue
            scored.sort(reverse=True)
            found[slot] = scored[0][1]
        return found

    def detect_properties(self, question: str) -> list[str]:
        """Every property the question asks about, best-scoring first.

        Separate from `detect` rather than a list inside it, because `slots` is
        a slot-to-single-value mapping that the whole system reads: the session
        carries it between turns, the renderer prints it, vision writes into it
        and the caveat scorer joins its values into a string. Putting a list in
        there broke the last of those immediately and would have had the session
        carrying a list of properties forward into an unrelated later question.

        The multiple values matter because one question routinely asks for two
        things — "how thick should the render be, and what preparation does the
        background need" — and keeping only the winner threw half the enquiry
        away, in that case the half with a whole knowledge-base article behind
        it.
        """
        lowered = question.lower()
        spec = self.spec.get("property_asked", {}).get("values", {})
        scored = []
        for value, terms in spec.items():
            score = sum(len(t) for t in terms
                        if re.search(rf"\b{re.escape(t)}\b", lowered))
            if score:
                scored.append((score, value))
        scored.sort(reverse=True)
        return [value for _score, value in scored]

    def load_bearing(self) -> list[str]:
        return [s for s, spec in self.spec.items()
                if not s.startswith("_") and spec.get("load_bearing")]

    def terms_for(self, slot: str, value: str) -> list[str]:
        return self.spec.get(slot, {}).get("values", {}).get(value, [])

    # A message that asks something is a new question, however short and
    # however many slots it happens to mention. Both halves are needed: "what
    # about Forte" carries no question mark, and "brick?" carries no question
    # word.
    _ASKS_SOMETHING = re.compile(
        r"^\s*(?:what|which|how|can|could|do|does|is|are|should|shall|will|"
        r"would|why|when|where|who|tell|give|any)\b|\?", re.I)

    def is_answer_to_askback(self, question: str) -> bool:
        """True if this message answers "what is the wall built of?".

        Getting this wrong is expensive in one direction. A false positive
        discards the question the person actually asked and re-answers the
        earlier one in its place, so they are answered about something they
        have moved on from and their real question disappears without trace.
        A false negative merely costs them the ask-back again.

        Two conditions, and the first used to be missing. The rule was "short
        and any slot was detected", which classed **"Can I use Ultra on the same
        wall?"** as an answer to an ask-back: it is nine words, and `detect`
        finds `property_asked=compatibility` in it. That is a new question about
        a different product, and it was being thrown away.

        So the slot has to be **load-bearing** -- one of the building facts an
        ask-back is ever raised for, which is what the docstring always claimed
        and the code did not do -- and the message must not be asking
        something. `property_asked`, `calculation` and `symptom` are shapes of a
        question rather than answers to one, and none of them can trigger this.
        """
        if len(question.split()) > 10:
            return False
        if self._ASKS_SOMETHING.search(question):
            return False
        detected = self.detect(question)
        return any(slot in detected
                   for slot in ("substrate", "location", "exposure"))


# --------------------------------------------------------------------- router


class Router:
    """The ordered decision. Nothing here is the model's call."""

    def __init__(self) -> None:
        self.gate = PolicyGate()
        self.slots = SlotDetector()
        vocab = _load("vocabularies.json")
        self.deferral = [re.compile(p, re.I)
                         for p in vocab["deferral_markers"]["patterns"]]
        self.synonyms = vocab["retrieval_synonyms"]

    # -- helpers ----------------------------------------------------------

    def _defers(self, text: str) -> bool:
        return any(p.search(text) for p in self.deferral)

    # "What is the pot life of Duro?" — the thing asked for sits between the
    # question opener and the preposition that introduces the product. This
    # exists because the vocabulary gate has a blind spot by construction: a
    # property nobody listed produces no slot, so no relevance check runs, and
    # the model composes an answer to a question the corpus never addresses.
    # The evaluation harness found exactly that with "pot life", which appears
    # nowhere in the 94 documents.
    _ASKED_PHRASE = (
        re.compile(r"\bwhat(?:'s| is| are)\s+(?:the\s+)?([a-z][a-z\s-]{2,28}?)\s+"
                   r"(?:of|for|on|in|with)\b", re.I),
        re.compile(r"\bhow\s+(?:much|many|long|thick|hot|cold)\s+"
                   r"([a-z][a-z\s-]{2,28}?)\s+(?:of|for|do|does|is|can|should)\b",
                   re.I),
        re.compile(r"\b(?:tell me|give me|what)\s+(?:the\s+)?([a-z][a-z\s-]{2,28}?)"
                   r"\s+(?:rating|value|figure|number)\b", re.I),
    )

    _PHRASE_NOISE = {"the", "a", "an", "your", "their", "its", "this", "that"}

    def asked_phrase(self, question: str) -> str:
        """The noun phrase a question asks for, when it has an obvious one."""
        for pattern in self._ASKED_PHRASE:
            m = pattern.search(question)
            if m:
                words = [w for w in m.group(1).lower().split()
                         if w not in self._PHRASE_NOISE]
                if words:
                    return " ".join(words)
        return ""

    def _asked_terms(self, question: str, slots: dict) -> tuple[str, list[str]]:
        """The property being asked about, and the words that count as it.

        Returns the slot value and every synonym for it, so the relevance gate
        can check "the asked-for term appears in a cited passage" against the
        vocabulary rather than against the customer's exact wording.

        When no slot matched, the question's own noun phrase is used instead.
        That is weaker — there are no synonyms for it — so it only refuses when
        the phrase and its head noun are both absent from every passage, which
        means the corpus genuinely does not discuss what was asked.
        """
        # A quantity question is a question about coverage, whatever words it
        # uses to ask. Without this the gate took the caller's own noun — "how
        # many bags" gives "bags" — and refused the Solo sheet for saying
        # "sack", which is the same thing and has a synonym entry to prove it.
        # The gate is meant to catch the near-miss, not the vocabulary gap.
        if "calculation" in slots and not slots.get("property_asked"):
            return "coverage", self.slots.terms_for("property_asked", "coverage")

        value = slots.get("property_asked", "")
        if value:
            # Every property the question asked for, not only the best-scoring
            # one. The gate is satisfied by any of them appearing, which is the
            # right reading of a two-property question: answer the half the
            # corpus publishes and let check 6 and the citation checks police
            # what actually prints, rather than refusing the whole enquiry
            # because the other half is not stated anywhere.
            asked = self.slots.detect_properties(question) or [value]
            terms: list[str] = []
            for prop in asked:
                terms += self.slots.terms_for("property_asked", prop)
            return value, list(dict.fromkeys(terms))

        phrase = self.asked_phrase(question)
        if not phrase:
            return "", []
        # The whole phrase, not its head noun. "pot life" reduced to "life"
        # matches "shelf life" and the gate stops working — the two are
        # different properties and one is published while the other is not.
        # Hyphenation is the only variation allowed, because sheets write both.
        variants = [phrase]
        if " " in phrase:
            variants.append(phrase.replace(" ", "-"))
            variants.append(phrase.replace(" ", ""))
        return phrase, variants

    @staticmethod
    def _present(terms: list[str], hits: list[Retrieved]) -> bool:
        blob = " ".join(f"{h.chunk.section} {h.chunk.content}" for h in hits).lower()
        return any(t.lower() in blob for t in terms)

    # -- the ordered decision ---------------------------------------------

    def unsupported_terms(self, question: str, hits: list[Retrieved],
                          carried: dict | None = None) -> list[str]:
        """The asked-for terms, when none of them appear in what was retrieved.

        This is the condition step 4 refuses on, exposed before routing so the
        caller can try once more by other means. Semantic retrieval having
        failed to surface the term is exactly the moment a lexical lookup is
        worth doing: the corpus may well publish the answer in a passage the
        embedding did not rank. Empty means the gate is satisfied and nothing
        further is needed.
        """
        slots = {**(carried or {}), **self.slots.detect(question)}
        _asked, terms = self._asked_terms(question, slots)
        if not terms or self._present(terms, hits):
            return []
        return terms

    def route(
        self,
        question: str,
        hits: list[Retrieved],
        above_threshold: bool,
        audiences: tuple[str, ...] = ("public",),
        carried: dict | None = None,
    ) -> Decision:
        # `carried` holds slots this question did not state — from an earlier
        # turn, or read off a photograph. They are merged *under* what this
        # question says, never over it: a caller who corrects themselves
        # ("actually it's stone") must not be answered from the old value, and
        # a slot detected here is evidence from the person rather than from
        # memory or from a model. That precedence is the whole safety property
        # of carrying anything at all.
        slots = {**(carried or {}), **self.slots.detect(question)}
        photo = "photograph" in slots

        # Step 1 — nothing close enough was found.
        if not above_threshold or not hits:
            return Decision(Path_.REFUSE, "nothing retrieved above the threshold",
                            "1", slots=slots, hits=hits, photograph=photo)

        # Step 2 — a published deferral beats a computed quantity.
        if self._defers(hits[0].chunk.content):
            return Decision(Path_.DEFER,
                            "the top passage refers the reader to the technical team",
                            "2", slots=slots, hits=hits, photograph=photo)

        # Step 3 — a cause or defect is asked. A photograph alone is not this.
        if "cause_asked" in slots or ("symptom" in slots and "cause_asked" in slots):
            return Decision(Path_.DIAGNOSIS,
                            "a cause or defect was asked, which is a human judgement",
                            "3", slots=slots, hits=hits, photograph=photo)

        # Step 4 — the relevance gate: right product, wrong property.
        asked, terms = self._asked_terms(question, slots)
        if terms and not self._present(terms, hits):
            return Decision(Path_.REFUSE,
                            f"the published material does not state {asked}"
                            " for this product",
                            "4", slots=slots, hits=hits, missing_term=asked,
                            photograph=photo)

        # Everything past step 4 carries the gate's own word list, so check 6
        # can enforce the rule this step just applied instead of re-deriving a
        # narrower one. See `Decision.asked_terms`.
        gate = list(terms)

        # Step 5 — a load-bearing slot is uncued.
        if "substrate" not in slots and self._needs_substrate(question, slots):
            return Decision(Path_.ASK_BACK,
                            "the substrate decides the product and was not stated",
                            "5", slots=slots, hits=hits, asked_terms=gate,
                            photograph=photo)
        per_option = "location" not in slots and self._location_matters(question, slots)

        # Step 6 — calculation words: print the published figures, refuse the sum.
        if "calculation" in slots:
            return Decision(Path_.EXTRACT,
                            "a quantity was asked; the published coverage and pack "
                            "size are printed and the multiplication is refused",
                            "6", slots=slots, hits=hits, sum_refused=True,
                            asked_terms=gate, per_option=per_option,
                            photograph=photo)

        # Step 7 — one document and a factual ask: print it, do not paraphrase.
        if self._single_document(hits) and asked:
            return Decision(Path_.EXTRACT,
                            "one document answers a factual question, so the passage "
                            "is printed rather than paraphrased",
                            "7", slots=slots, hits=hits, asked_terms=gate,
                            per_option=per_option, photograph=photo)

        # Staff see passages, not prose. Composing for staff is roadmap.
        if "staff" in audiences and "public" not in audiences:
            return Decision(Path_.EXTRACT, "staff audience: passages, not prose",
                            "7s", slots=slots, hits=hits, asked_terms=gate,
                            photograph=photo)

        # Step 8 — otherwise the model composes over what was retrieved.
        return Decision(Path_.COMPOSE, "several passages bear on the question",
                        "8", slots=slots, hits=hits, asked_terms=gate,
                        per_option=per_option, photograph=photo)

    # -- step 5 predicates -------------------------------------------------

    _PRODUCT_CHOICE = re.compile(
        r"\b(which|what) (product|plaster|render|mortar|system)\b"
        r"|\brecommend\b|\bsuitable\b|\bshould i use\b|\bbest for\b|\bwhat do i (use|need)\b",
        re.I,
    )

    # Whether the question is about the asker's own building. A first person, a
    # possessive, a symptom they can see, or a photograph they have taken all
    # say "this is my wall"; their absence says the question is about the range.
    _ABOUT_MY_BUILDING = re.compile(
        r"\b(?:my|our|mine|ours|i|i'm|i've|we|we're|we've|us|me)\b", re.I)

    def _needs_substrate(self, question: str, slots: dict) -> bool:
        """Substrate is load-bearing when a product is being chosen *for a wall*.

        Two conditions, not one. A product choice alone is not enough, because
        "what products are suitable for lime-based external finishes" is a
        question about the catalogue, and asking which wall it is for answers a
        question nobody asked. That question is the brief's own worked example,
        and it used to reach an ask-back — the single most likely thing an
        assessor types, met with a clarifying question instead of an answer.

        "Which plaster should I use" is the other shape: the same choice, but
        about a specific job, where a recommendation without a substrate is a
        guess in a liability-sensitive domain. The word that separates them is
        the first person.

        A symptom or a photograph counts as the same signal. Nobody describes
        crazing or attaches a picture about a product range in the abstract.
        """
        if not self._PRODUCT_CHOICE.search(question):
            return False
        return bool(
            self._ABOUT_MY_BUILDING.search(question)
            or "symptom" in slots
            or "photograph" in slots
        )

    _LOCATION_SENSITIVE = ("coverage", "thickness", "coats", "drying", "finish",
                           "painting", "temperature")

    def _location_matters(self, question: str, slots: dict) -> bool:
        asked = slots.get("property_asked", "")
        return bool(self._PRODUCT_CHOICE.search(question)) or asked in self._LOCATION_SENSITIVE

    @staticmethod
    def _single_document(hits: list[Retrieved]) -> bool:
        return len({h.chunk.canonical_url for h in hits}) == 1
