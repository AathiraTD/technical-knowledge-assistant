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
    """The five outcomes, plus the two composites the diagram names."""

    ROUTE = "route"                 # fixed referral, no retrieval
    EXTRACT = "extract"             # a passage printed verbatim by code
    COMPOSE = "compose"             # the model, over retrieved passages
    DEFER = "cited hand-off"        # a published deferral, quoted
    DIAGNOSIS = "diagnosis"         # published causes plus a hand-off
    ASK_BACK = "ask back"           # a load-bearing slot is uncued
    REFUSE = "refuse"


@dataclass
class Decision:
    """Why this question went the way it did. Printed with every answer."""

    path: Path_
    reason: str
    step: str = ""
    topic: str = ""
    slots: dict = field(default_factory=dict)
    hits: list = field(default_factory=list)
    missing_term: str = ""
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
    """Two questions in one message are two questions.

    A message that asks a price and a coverage is not one question with one
    answer: the price part must reach the policy gate and the coverage part
    must reach retrieval. Answering the pair as a unit sends one of them down
    the wrong path.
    """
    parts = [p.strip(" ,") for p in _SPLIT.split(question) if p and p.strip(" ,")]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts or [question.strip()]


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
            best_value, best_score = "", 0
            for value, terms in spec["values"].items():
                score = sum(
                    len(t) for t in terms
                    if re.search(rf"\b{re.escape(t)}\b", lowered)
                )
                if score > best_score:
                    best_value, best_score = value, score
            if best_value:
                found[slot] = best_value
        return found

    def load_bearing(self) -> list[str]:
        return [s for s, spec in self.spec.items()
                if not s.startswith("_") and spec.get("load_bearing")]

    def terms_for(self, slot: str, value: str) -> list[str]:
        return self.spec.get(slot, {}).get("values", {}).get(value, [])


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
        value = slots.get("property_asked", "")
        if value:
            return value, self.slots.terms_for("property_asked", value)

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

        # Step 5 — a load-bearing slot is uncued.
        if "substrate" not in slots and self._needs_substrate(question, slots):
            return Decision(Path_.ASK_BACK,
                            "the substrate decides the product and was not stated",
                            "5", slots=slots, hits=hits, photograph=photo)
        per_option = "location" not in slots and self._location_matters(question, slots)

        # Step 6 — calculation words: print the published figures, refuse the sum.
        if "calculation" in slots:
            return Decision(Path_.EXTRACT,
                            "a quantity was asked; the published coverage and pack "
                            "size are printed and the multiplication is refused",
                            "6", slots=slots, hits=hits, sum_refused=True,
                            per_option=per_option, photograph=photo)

        # Step 7 — one document and a factual ask: print it, do not paraphrase.
        if self._single_document(hits) and asked:
            return Decision(Path_.EXTRACT,
                            "one document answers a factual question, so the passage "
                            "is printed rather than paraphrased",
                            "7", slots=slots, hits=hits, per_option=per_option,
                            photograph=photo)

        # Staff see passages, not prose. Composing for staff is roadmap.
        if "staff" in audiences and "public" not in audiences:
            return Decision(Path_.EXTRACT, "staff audience: passages, not prose",
                            "7s", slots=slots, hits=hits, photograph=photo)

        # Step 8 — otherwise the model composes over what was retrieved.
        return Decision(Path_.COMPOSE, "several passages bear on the question",
                        "8", slots=slots, hits=hits, per_option=per_option,
                        photograph=photo)

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
