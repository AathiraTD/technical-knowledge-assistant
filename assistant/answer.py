"""Answering: extract, compose, six checks, hand-off, rendering.

The checks are the point of this file. The model runs on one path out of seven,
and everything it produces is treated as a proposal that has to survive six
mechanical tests before a person sees it. A failure is not repaired and not
retried — it becomes a refusal that still carries whatever the site does
publish, because a refusal is a designed outcome here rather than a fault.

Check 2 is the one that does most of the work. Every number in a generated
answer must appear, character for character, in a passage that answer cites.
Units are normalised for the comparison and never for the display, so "5-6
litres" is matched against "5 – 6 litres" but printed exactly as published.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from . import observability as obs
from . import ollama
from .logging.diagnosis_capture import DiagnosisCapture
from .model import Retrieved
from .repository import product_matches
from .router import Decision, Path_

# ---------------------------------------------------------------- the prompt

SYSTEM = (
    "You answer questions about Lime Green building products using only the "
    "numbered passages given to you.\n"
    "\n"
    "FORMAT — every sentence you write must end with the marker of the passage "
    "it came from, like [2]. A sentence without a marker is discarded. Put the "
    "marker of the passage that actually contains the fact, not the first "
    "passage in the list.\n"
    "\n"
    "STYLE — answer directly. Do not explain your reasoning, do not describe "
    "what you are doing, and do not write an introduction or a conclusion. "
    "Two or three sentences is usually the whole answer.\n"
    "\n"
    "RULES\n"
    # Deliberately no "say so if the passages do not answer it". That invited
    # exactly one sentence — "The passages do not mention gypsum plaster
    # specifically" — which check 1 then refused, because a statement about what
    # the passages *lack* cannot cite a passage and cannot overlap one. The
    # prompt was asking for the one thing the checks are built to kill. Writing
    # nothing is the right behaviour: an unsupported answer fails check 1 anyway
    # and falls into the refusal renderer, which says it better — it names what
    # was looked for, prints what is published with its source, and gives the
    # contact line.
    "- Use only the passages. Write nothing you cannot cite.\n"
    "- Copy figures exactly as written, with their units. Never convert, round "
    "or add up.\n"
    "- Keep every figure with the product it was published for.\n"
    "- Never say two products are interchangeable unless a passage says so.\n"
    "- Never promise an outcome, judge a building, or mention another brand.\n"
)

# One worked example. A 4B model follows a demonstrated format far more reliably
# than a described one, and the failure this fixes was real: the model found the
# right figure, wrote it as narration across five uncited sentences, and the
# citation check correctly threw the whole answer away.
PROMPT = """Here is an example of the required format.

Example passages:
[1] Duro Lime Plaster — Description
Duro is a general purpose lime undercoat plaster for most masonry backgrounds.
[2] Duro Lime Plaster — Mixing
Add 4.5 to 5 litres of clean water per 25 kg bag and mix for three minutes.

Example question: How much water does Duro need?

Example answer:
Add 4.5 to 5 litres of clean water per 25 kg bag [2]. Mix for three minutes [2].

Now answer this one the same way.
{history}
Passages:
{passages}

Question: {question}
{assumptions}{guidance}
Answer:"""


# The transcript, when there is one. Deliberately not called "context" and
# deliberately not shaped like a passage: it carries no citation marker, so a
# sentence drawn from it can cite nothing, and check 1 discards any sentence
# whose clauses do not overlap a passage that sentence cites. That is the
# enforcement. The wording below is only the request -- a prompt is not a
# security boundary, and the two are kept separate on purpose.
HISTORY_BLOCK = """
Earlier turns of this conversation, for working out what the question refers
to -- "it", "that wall", "the one you mentioned". This is NOT evidence. Never
take a fact from it and never cite it; every fact must still come from a
numbered passage below.
{history}
"""


# ------------------------------------------------------------- what is known

# The three facts about the caller's building that an answer may be shaped by,
# and the same three `assistant/session.py` carries between turns. The order is
# the order they are printed in.
STATABLE_SLOTS = ("substrate", "location", "exposure")


class Provenance(Enum):
    """Where a slot value came from. The whole point is that these differ.

    Printing "Assumed: location external" at somebody who wrote "my external
    lime render" makes a system that listened look like a system that guessed,
    and it invites them to correct something that was never wrong. Four states,
    and each one is produced by something this repository actually does:

    ``STATED``   the caller's own words, in the question being answered.
    ``CARRIED``  the caller's own words, in an earlier turn of the same
                 conversation — `assistant/session.py` holds exactly these three
                 slots forward. Still stated, just not in this sentence, so it
                 is printed as something they told us rather than as a guess.
    ``OBSERVED`` read off a photograph attached to this turn. Not stated, not
                 assumed, and the distinction is the whole reason this member
                 exists — see below.
    ``ASSUMED``  nothing was said and the system chose. Today the only producer
                 is the per-option answer of decision 10: inside/outside was
                 uncued, so both are answered.

    ``OBSERVED`` used to be argued *against* here, and the argument was right
    while it held: `assistant/vision.py` resolved observations to a `carried`
    dict, nothing handed one to `Assistant.ask`, and a member nothing can
    produce is a claim the renderer could never make honestly. That reasoning
    is now obsolete rather than merely inconvenient. `Assistant.ask` takes
    `images`, runs them through `vision.slots_from_images`, and merges the
    resolved slots into `carried` — so the state exists, and the moment it
    exists the *absence* of the member becomes the defect.

    The defect is worth naming, because it is worse than the one the provenance
    distinction was introduced to fix. Without ``OBSERVED``, a slot read off a
    photograph is indistinguishable from a slot the caller stated two turns ago,
    so the answer prints "brick (substrate), as you told me earlier in this
    conversation" about a wall nobody described. That attributes a model's
    uncalibrated reading of an image to the person, which is exactly the thing
    a citation-bound system must never do: it launders an inference into
    testimony. The photograph is the one source the caller can check against
    their own eyes, and saying so is the whole value of the member.

    ``INFERRED`` and ``UNKNOWN``, the other two states decision 16.1 lists, are
    still deliberately absent, for the reason ``OBSERVED`` was: nothing in this
    repository infers a slot, and an unknown fact is represented by the slot not
    being there at all.
    """

    STATED = "stated"
    CARRIED = "carried"
    OBSERVED = "observed"
    ASSUMED = "assumed"


# One sentence per provenance, which is the whole differentiation. A new
# provenance adds a row here and changes nothing else.
_PHRASE = {
    Provenance.STATED: "{value} ({slot}), as you said",
    Provenance.CARRIED: "{value} ({slot}), as you told me earlier in this conversation",
    Provenance.OBSERVED: "{value} ({slot}), from the photograph you sent",
    Provenance.ASSUMED: "{slot}: {value} — assumed, since you did not say",
}


@dataclass(frozen=True)
class SlotFact:
    """One slot value and how the system came to hold it."""

    slot: str
    value: str
    provenance: Provenance

    @property
    def sentence(self) -> str:
        return _PHRASE[self.provenance].format(
            slot=self.slot, value=self.value.replace("_", " "))

    @property
    def stated(self) -> bool:
        """Did the caller say this, whether in this turn or an earlier one?

        ``OBSERVED`` answers no, and the no is the point. A substrate read off a
        photograph was never said by anyone: the model looked at pixels and
        matched what it saw against the vocabulary. Reporting that as something
        the caller told us would put words in their mouth on the strength of an
        uncalibrated confidence score.

        The reading this property is *not* allowed to imply is "assumed, so it
        belongs under the Assumed heading". An observation is not a guess
        either, and `_finish` therefore groups on the provenance itself rather
        than on this boolean — three groups, not two. This property survives
        because "did the caller say it" is a question worth asking on its own,
        and because the prompt context and the printed "Answered for" sentence
        both want exactly the stated ones.
        """
        return self.provenance in (Provenance.STATED, Provenance.CARRIED)



# How each missing fact is asked for. A slot name is not a question -- "I need
# substrate" is not English -- and the phrasing decides whether somebody can
# answer it in one line.
_MISSING_PHRASE = {
    # Phrased to match the wording `ask_back` has always printed. The copy is
    # customer-facing, was written once and reviewed once, and "what is the wall
    # built of underneath" is the sentence the transcript and the existing tests
    # both expect. Rewording it because a different code path now produces it
    # would be changing what a customer reads for an internal reason.
    "substrate": ("what is the wall built of underneath -- brick, stone, cob, "
                  "laths, plasterboard, or an existing plaster or render?"),
    "location": "is this an inside or an outside wall?",
    "exposure": ("how exposed is the wall -- sheltered, moderate, or severe "
                 "weather?"),
    "objective": "what are you trying to achieve?",
}

# ---------------------------------------------------------------- the result


@dataclass
class Answer:
    """What the caller gets, including how it was arrived at."""

    text: str
    path: str
    sources: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    # Genuinely assumed values only, so the "Assumed:" heading the CLI renderer
    # prints is true of everything under it. What the caller actually said is in
    # `facts`, and is printed inside `text` as a sentence rather than as a list.
    assumptions: list[str] = field(default_factory=list)
    # What a photograph settled, as its own group. It is already a line inside
    # `text`, so the CLI transcript loses nothing by ignoring this; it is
    # repeated here for the same reason `disclosure` is, so a surface wanting to
    # present it separately does not have to parse prose out of the answer.
    observed: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    refused: bool = False
    # Every slot value behind this answer, with its provenance.
    facts: list[SlotFact] = field(default_factory=list)
    # The raw evidence block appended to `text` on a refusal. It stays inside
    # `text` so the CLI — canonical for the transcript — loses nothing, and it
    # is repeated here so the page can put it behind a disclosure instead of
    # opening with 600 characters of datasheet.
    disclosure: str = ""

    @property
    def body(self) -> str:
        """`text` minus the disclosure, for a surface that shows it separately."""
        # `removesuffix` of an empty string is the identity, so an ordinary
        # answer needs no branch and cannot be trimmed by accident.
        return self.text.removesuffix(self.disclosure).rstrip()


# -------------------------------------------------------------- the checks

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[(\d+)\]")
# A number with the unit attached to it, which is the unit of comparison.
_NUMBER = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:[-–—]\s*\d+(?:[.,]\d+)?)?"
    r"\s*(?:mm|cm|m2|m²|m3|m³|m|kg|g|l|litres?|ltr|%|°c|c|n/mm2|mpa|"
    r"hours?|hrs?|days?|weeks?|months?|coats?|passes?|bags?|sacks?)?",
    re.I,
)

_QUALIFIER = re.compile(
    r"\b(minimum|maximum|at least|up to|no more than|below|above)\b", re.I)

def _same_sentence(text: str, word: str, digits: str) -> bool:
    """Do a qualifier and a figure share a sentence of the cited passage?

    The unit is the sentence, not a character count. A count was tried first at
    sixty characters and refused a correct answer by eight: the rendering
    checklist says "Specify 16mm minimum thickness of lime render in moderately
    exposed locations, or 25mm in very exposed locations", where one "minimum"
    governs both figures and sits 68 characters from the second. The model read
    that correctly and the arbitrary window called it an invention.

    A sentence is what actually carries the relationship, and it stays strict
    where strictness matters: "Maximum coverage is achieved on a well prepared
    background. Apply at 10 mm per coat." keeps the qualifier in a different
    sentence from the figure, so it still fails — which is the case this check
    exists for.
    """
    for part in _SENTENCE.split(text):
        if re.search(re.escape(word), part) and re.search(re.escape(digits), part):
            return True
    return False

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "with", "as", "at", "by", "it", "this", "that", "from", "can", "will",
    "should", "must", "may", "not", "you", "your", "we", "our", "per", "if",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOP and len(w) > 2}


# Clause separators. Semicolons and colons always start a new assertion; a
# comma does so only before a word that opens one, because splitting on every
# comma would cut "5, 6 litres" and every ordinary list.
_CLAUSE = re.compile(
    r"\s*[;:]\s*|,\s+(?=(?:and|but|so|then|though|although|while|whereas|"
    r"it|they|this|these|that|which|there)\b)", re.I)


def _clauses(sentence: str) -> list[str]:
    """A sentence split where a new claim can begin."""
    return [c.strip() for c in _CLAUSE.split(sentence) if c.strip()]


def _normalise_number(token: str) -> str:
    """Comparison form only. Never used for display."""
    t = token.lower().strip()
    t = t.replace("–", "-").replace("—", "-").replace(",", ".")
    t = re.sub(r"\s+", "", t)
    t = t.replace("litres", "l").replace("litre", "l").replace("ltr", "l")
    t = t.replace("m²", "m2").replace("m³", "m3")
    t = t.replace("°c", "c")
    t = re.sub(r"(\d)\.0\b", r"\1", t)
    return t


# What a coverage or pack-size passage actually says. Deliberately about the
# words a datasheet uses rather than its headings: Duro prints "1 bag will cover
# 1 m 2" under "Storage", and Fine Stuff gives coverage its own section. A
# heading-based rule would find one and miss the other.
_COVERAGE = re.compile(
    r"\bcover(?:s|age|ing)?\b|\bspread\s*rate\b|\byield\b|"
    # An area unit, however the sheet spaces it: "1m2", "1½ m 2", "1 m²".
    r"m\s?[2²](?![0-9a-z])", re.I)


def _mentions_coverage(text: str) -> bool:
    """Does this passage carry the figure a quantity question needs?"""
    return bool(_COVERAGE.search(text))


# ------------------------------------------------------- evidence binding
#
# A compound question asks for two things and the answer has to carry two
# claims, each cited to the passage that actually states it. Nothing used to
# establish that correspondence: the model was handed five undifferentiated
# passages and one instruction to cite the right one, and on the question this
# was written for — whether Ultra suits an internal brick wall, and at what
# thickness — it bound the suitability claim to the thickness passage. Check 1
# refused the whole answer, correctly, and a fully published two-part answer
# was lost while the suitability passage sat in the evidence unread.
#
# So the correspondence is computed here, deterministically, from the same
# vocabulary the relevance gate uses, and the model is told it. This is a
# generation control and not a safety boundary: checks 1 to 6 are unchanged and
# remain the enforcement. What it removes is the *opportunity* to misbind,
# which is the same bargain `recommend` already makes by filtering passages to
# the approved products before generation.

# At most this many passages per property. A binding that named four passages
# would be property grouping rather than claim-to-evidence binding, and the
# model would be no better off than with the undifferentiated list.
EVIDENCE_BINDING_CAP = 2

# What a passage earns for stating a property beside a figure rather than
# merely mentioning its name. "in a uniform thickness of between 10 and 30mm"
# states a thickness; "builds up in thick coats" mentions one. Large enough to
# beat any difference in term length, because directness is the stronger
# signal of the two.
_STATES_IT_BONUS = 100


def _property_support(hits: list[Retrieved], terms: list[str],
                      product: str = "") -> list[int]:
    """The 1-based markers of the passages that most directly state a property.

    Ranked, then cut to the best-scoring band and capped — not every passage
    that happens to contain a related word. Asked about thickness, four of the
    five Ultra passages match something in the thickness vocabulary: one says
    "uniform thickness of between 10 and 30mm", another says "thick", a third
    "depth", a fourth "build up". Offering all four as equally valid evidence
    is how a claim ends up cited to a passage that merely alludes to it.

    Two signals, both deterministic:

    * **Specificity** — the longest vocabulary term the passage matches, the
      same matched-term-length rule `SlotDetector.detect` scores questions by.
      "thickness" beats "thick" beats "mm".
    * **Directness** — whether one of those terms shares a sentence with a
      figure. A datasheet states a property next to its number; a product page
      mentions the word in prose.

    Only the passages tied at the best score survive, so a passage that clearly
    states the property returns alone. Ties are kept, up to the cap, because two
    sections genuinely stating the same property both belong.

    **The named product wins over both signals.** Lexical ranking alone will
    hand a claim about Ultra to a Warmshell passage, because a system guide
    saying "applied at a thickness of 6mm and is suitable for solid walls"
    matches both vocabularies and states its property beside a figure. Pointing
    the model at that is worse than not binding at all — it is check 3's
    cross-product attribution, arranged by the prompt. So a named product's own
    passages are considered first, and everything else only if the product
    publishes nothing for this property: a finish coat's hardening time is
    genuinely published on the base coat's sheet, and refusing to look there
    would lose the multi-document answers the brief asks for.
    """
    scored: list[tuple[int, int]] = []
    for marker, hit in enumerate(hits, 1):
        blob = f"{hit.chunk.section} {hit.chunk.content}".lower()
        matched = [t for t in terms
                   if re.search(rf"\b{re.escape(t)}\b", blob)]
        if not matched:
            continue
        direct = any(
            re.search(r"\d", sentence) and re.search(rf"\b{re.escape(t)}\b", sentence)
            for sentence in _SENTENCE.split(blob) for t in matched)
        scored.append((max(len(t) for t in matched)
                       + (_STATES_IT_BONUS if direct else 0), marker))
    if product:
        own = [(score, marker) for score, marker in scored
               if product_matches(product, hits[marker - 1].chunk.product or "")]
        scored = own or scored
    if not scored:
        return []
    best = max(score for score, _marker in scored)
    return [marker for score, marker in scored
            if score == best][:EVIDENCE_BINDING_CAP]


def evidence_binding(decision: Decision) -> dict[str, list[int]]:
    """Each thing the question asked about, and the passage that states it.

    Empty when the decision carries no per-property terms — a `Decision` built
    directly by a test, or one from a step before the relevance gate — so every
    caller degrades to the prompt as it was.
    """
    product = decision.slots.get("product", "")
    bound: dict[str, list[int]] = {}
    for prop, terms in (decision.evidence_terms or {}).items():
        markers = _property_support(decision.hits, terms, product)
        if markers:
            bound[prop] = markers
    return bound


def _distinguishes(bound: dict[str, list[int]]) -> bool:
    """Do these bound sets actually tell the properties apart?

    They do only when no passage supports two of them. A binding whose sets
    overlap names where each half is stated and separates nothing, because the
    shared passage is an answer to both — and the model could already see that
    from the passages themselves.

    Stated as a property rather than as a special case, because the cost of
    getting it wrong is not a worse binding but a re-rendered figure. Gold
    scenario GD2 asks for a thickness and a preparation on a very exposed wall.
    Both bind to the rendering checklist's Design section, `preparation` also to
    its Application section and `thickness` also to a background-preparation
    article: two sets, overlapping on the one passage that carries the answer.
    The block told the model nothing, and the generation it perturbed printed
    the published "25mm" as "25 mm". Both forms pass check 2, which normalises
    whitespace for comparison; the evaluation compares published figures with
    whitespace collapsed and nothing else normalised, so the space was a failed
    assertion about a figure the system had in fact got right.

    So the rule is the same one `len(bound) > 1` applies for a single property,
    generalised: say nothing unless saying it separates something.
    """
    seen: set[int] = set()
    for markers in bound.values():
        if seen & set(markers):
            return False
        seen |= set(markers)
    return True


def promote_bound(hits: list[Retrieved],
                  bound: dict[str, list[int]]) -> list[Retrieved]:
    """The evidence list with the one authoritative passage moved to the front.

    An experiment in fixing a misbinding by *ordering* rather than by telling
    the model anything. The failing case is a question asking one thing —
    "would Ultra be suitable internally" — where the binding correctly
    identifies the single passage that answers it, the block is suppressed
    because one claim cannot be bound to the wrong half of itself, and the
    model then cites a different passage anyway and is refused by check 1. The
    evidence was there, ranked fourth, and nothing pointed at it.

    **Deliberately narrow.** It fires only when exactly one property is bound
    to exactly one passage, which is the shape above. Two bound properties are
    left alone: the model needs both, and promoting one of them says the other
    matters less. A property bound to two passages is a tie the ranking already
    declined to break, and breaking it here would be inventing an authority the
    binding does not claim.

    **Ordering only.** The same passages are returned, none added, none
    dropped, and everything not promoted keeps its relative order. The caller
    must then use this one list for the prompt, the checks and the printed
    sources alike, because markers are positional: a list reordered for the
    prompt and not for `run_checks` would renumber the evidence underneath the
    verification, which is the one way this change could do real harm.
    """
    if len(bound) != 1:
        return hits
    markers = next(iter(bound.values()))
    if len(markers) != 1:
        return hits
    index = markers[0] - 1
    if not 0 <= index < len(hits) or index == 0:
        return hits
    return [hits[index]] + [h for i, h in enumerate(hits) if i != index]


def _binding_guidance(decision: Decision) -> str:
    """The binding and the substrate wording, as instructions to the model.

    Two parts, each emitted only when it has something to say.

    The **binding** narrows rather than forbids. It names where each half of
    the question is stated; it does not tell the model a passage is off limits,
    because a fact can legitimately sit somewhere the vocabulary did not
    predict, and a prohibition there would buy a refusal rather than a better
    answer.

    The **substrate line** closes the gap between the caller's word and the
    corpus's. No Ultra document contains "brick"; the product page says
    "Suitable for most masonry and lath backgrounds". Left to itself the model
    bridged that inside a cited sentence — "suitable for most masonry
    backgrounds including brick" — which reads as the manufacturer having said
    "brick" when it did not. The class relation is a deterministic rule
    (`SlotDetector.class_of`), so the honest form of the sentence can be
    required rather than hoped for.
    """
    lines: list[str] = []

    bound = evidence_binding(decision)
    # Two or more, because one claim cannot be bound to the wrong half of
    # itself. A question asking a single thing is already covered by the system
    # prompt's standing rule — cite the passage that contains the fact — and
    # adding a line that tells it the same thing again buys nothing and costs
    # prompt churn on every lookup in the corpus. That cost is not theoretical:
    # the one-property form of this guidance moved "25kg sack" to "25 kg sack"
    # on the brief's first test question, and the evaluation compares published
    # figures with whitespace collapsed and nothing else normalised, so a space
    # the model inserted is a failed assertion about a figure.
    #
    # And only when the sets separate the properties — see `_distinguishes`.
    # The substrate line below is independent of this and is emitted on its own
    # terms: it closes a vocabulary gap rather than pointing at a passage, so a
    # binding that says nothing useful must not take it down with it.
    if len(bound) > 1 and _distinguishes(bound):
        lines.append("Where each thing asked about is stated:")
        lines += [f"- {prop}: "
                  + ", ".join(f"[{m}]" for m in markers)
                  for prop, markers in bound.items()]
        # States the shape without demanding the content. A property can be
        # detected incidentally — "per bag" reads as coverage in a question
        # about mixing water — and an instruction to answer every listed
        # property would turn that into a demand for a coverage sentence.
        lines.append("Write each fact in its own sentence, cited to the "
                     "passage listed for it.")

    substrate = decision.slots.get("substrate", "")
    published = decision.substrate_class
    if substrate and published:
        blob = " ".join(f"{h.chunk.section} {h.chunk.content}"
                        for h in decision.hits).lower()
        word = substrate.replace("_", " ").lower()
        if word not in blob and published.lower() in blob:
            lines.append(
                f"The passages say \"{published}\" and never say \"{word}\". "
                f"Write what they say. Do not write \"{word}\" as something "
                "the passages state.")

    return "\n".join(lines)


def _numbers(text: str) -> list[str]:
    """Figures with their units attached, which is the unit of comparison.

    Every match necessarily contains a digit, because the pattern opens with
    one; an earlier guard re-checked for a digit here and could never fire.
    """
    return [m.group(0).strip() for m in _NUMBER.finditer(text)]


def _figure_is_published(token: str, passage: str) -> bool:
    """Is this exact figure in the passage — not merely inside one of its figures?

    A plain substring test is what this used to do, and it let a fabricated
    figure through whenever it happened to be a tail of a real one. Two cases,
    both found by review rather than by use, and both are the failure the brief
    names outright — an assistant confidently wrong about a number:

        passage "Pack size is 25 kg"      answer "the pack is 5 kg"    printed
        passage "16 to 20 m2 per bag"     answer "6 to 20 m2 per bag"  printed

    A 25 kg sack quoted as 5 kg is a wrong mix on site. Dropping a leading digit
    is also the single most likely token-level error a model makes.

    The fix is to anchor the match at a digit boundary rather than to demand
    equality with a whole published figure. Equality would be stricter and
    wrong: the Solo sheet prints "5-6 litres" and an answer saying "between 5
    and 6 litres" is correct, cites correctly, and must not be refused — it is
    the brief's own first test question. So the rule is that a figure may sit
    inside a published figure, but not with another digit or a decimal point
    pressed against it. "6" may match in "5-6"; it may not match in "16".
    """
    # A dot only disqualifies when it is a decimal point — one with a digit on
    # the far side. Rejecting every dot would reject "8 mm" at the end of a
    # sentence, which is where figures most often sit.
    anchored = (rf"(?<!\d)(?<!\d\.){re.escape(token)}(?!\d)(?!\.\d)")
    return re.search(anchored, passage) is not None


def products_named(text: str, registry) -> set[str]:
    """Every published product this text names, in the one normalised spelling.

    Word boundaries, not containment. `"solo" in text` matches "solo" inside
    "isolation" and every other accident of spelling, and the thing this feeds
    refuses an answer, so a false positive is a refusal of something correct.

    Returned without the maker's name, because the site writes both forms and
    which one a passage uses is an accident of how that page was written.
    """
    lowered = text.lower()
    return {_without_brand(name.lower()) for name in registry
            if name and re.search(rf"\b{re.escape(name.lower())}\b", lowered)}


def run_checks(
    text: str,
    hits: list[Retrieved],
    names: dict,
    asked_terms: list[str],
    product: str = "",
    asked_products: tuple[str, ...] = (),
) -> list[str]:
    """The checks, in order. Returns the failures; empty means it prints.

    `product` and `asked_products` drive check 7 and nothing else. Both default
    to empty, so every existing caller runs exactly the six checks it always
    ran, and a question that resolved no product is unaffected.
    """
    failures: list[str] = []
    cited_index = {i + 1: h for i, h in enumerate(hits)}

    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]

    # 1 — every sentence cited, and the citation has to be about that sentence.
    for sentence in sentences:
        marks = [int(m) for m in _CITE.findall(sentence)]
        if not marks:
            failures.append(f"check 1: a sentence carries no citation — {sentence[:70]!r}")
            continue
        if any(m not in cited_index for m in marks):
            failures.append(f"check 1: cites a passage that was not retrieved — {marks}")
            continue
        passage_words = set()
        for m in marks:
            h = cited_index[m]
            passage_words |= _words(f"{h.chunk.section} {h.chunk.content}")
        # Clause by clause, not sentence by sentence. Measured over a whole
        # sentence, a grounded opening dilutes a fabricated tail below the
        # threshold and carries it onto the page under the opening's citation:
        #
        #   "Solo suits most solid masonry backgrounds [1]; it is fine on cob."
        #
        # printed, while the bare "Ultra is probably fine on cob [1]" refused —
        # the same invention, hidden behind a true clause. `DECISIONS.md` names
        # that inference as the design's weakest point and says check 1 kills
        # it; it only killed the standalone form.
        #
        # Each clause is still checked against the whole sentence's citations
        # rather than being made to carry its own marker, because the model is
        # told to cite per sentence and splitting that requirement would refuse
        # ordinary correct answers.
        for clause in _clauses(_CITE.sub("", sentence)):
            clause_words = _words(clause)
            if not clause_words:
                continue        # nothing but stop words; no claim to check
            # One content word is still a claim: "[1]: it will not crack"
            # reduces to {crack} and is an outcome promise the passages do not
            # support. Skipping short clauses is what let that one print.
            if len(clause_words & passage_words) / len(clause_words) < 0.4:
                failures.append(
                    f"check 1: a clause does not overlap the passage it cites — "
                    f"{clause[:70]!r}"
                )

    # 2 — every number appears verbatim in a passage the sentence cites.
    for sentence in sentences:
        marks = [int(m) for m in _CITE.findall(sentence)]
        cited_text = " ".join(
            cited_index[m].chunk.content for m in marks if m in cited_index
        )
        cited_norm = _normalise_number(cited_text)
        for token in _numbers(_CITE.sub("", sentence)):
            if not _figure_is_published(_normalise_number(token), cited_norm):
                failures.append(
                    f"check 2: {token!r} is not in the passage it is cited to"
                )

    # 3 — a number stays with the product it was published for.
    products_in_hits = {h.document.product.lower() for h in hits if h.document.product}
    for sentence in sentences:
        # Strip the citation markers before counting figures. Check 2 already
        # does; check 3 did not, so the digit inside "[1]" counted as a figure
        # and this guard never fired. Harmless in effect, because the check only
        # reports when a sentence also names another product, but it meant a
        # sentence with no figures at all was still being examined for figure
        # attribution.
        if not _numbers(_CITE.sub("", sentence)):
            continue
        marks = [int(m) for m in _CITE.findall(sentence)]
        cited_products = {
            cited_index[m].document.product.lower()
            for m in marks if m in cited_index and cited_index[m].document.product
        }
        # A passage may legitimately name another product. The Forte datasheet's
        # Finishing Coats section says which finish coats go over Forte and how
        # long to wait, naming Tradirend and Natural Finish outright — so a
        # sentence about Tradirend citing the Forte sheet is correctly
        # attributed, because the Forte sheet is what published it.
        #
        # Without this the check refused exactly the multi-document
        # compatibility answers the brief asks for, which is a false refusal and
        # the most expensive kind of mistake this system can make: the material
        # said it, the citation was right, and the answer still did not print.
        cited_text = " ".join(
            cited_index[m].chunk.content for m in marks if m in cited_index)
        named = {p for p in products_in_hits
                 if p and p not in cited_products
                 and _names_product(sentence, p)
                 and not _names_product(cited_text, p)}
        if named:
            failures.append(
                f"check 3: a figure is stated against {sorted(named)[0]!r} but cited "
                "to a different product's passage"
            )

    # 4 — qualifiers travel with their figure, inside the printed passage.
    #
    # "Inside the passage" was all this used to require, and the architecture
    # promises more than that: the qualifier and its figure must travel
    # *together*. A sheet reading "Maximum coverage is achieved on a well
    # prepared background. Apply at 10 mm per coat." accepted the answer "apply
    # at a maximum of 10 mm per coat", inventing a maximum thickness out of a
    # sentence about coverage. The word was present; the claim was not.
    for sentence in sentences:
        qualifier = _QUALIFIER.search(sentence)
        if not qualifier:
            continue
        marks = [int(m) for m in _CITE.findall(sentence)]
        cited_text = " ".join(
            cited_index[m].chunk.content for m in marks if m in cited_index
        ).lower()
        word = qualifier.group(1).lower()
        if word not in cited_text:
            failures.append(
                f"check 4: the qualifier {qualifier.group(1)!r} is not in the "
                "cited passage"
            )
            continue
        figure = _NUMBER.search(_CITE.sub("", sentence[qualifier.end():]))
        digits = re.search(r"\d+(?:[.,]\d+)?", figure.group(0)) if figure else None
        if not digits:
            continue        # a qualifier with no figure after it qualifies nothing
        if not _same_sentence(cited_text, word, digits.group(0)):
            failures.append(
                f"check 4: {qualifier.group(1)!r} and {digits.group(0)!r} are both "
                "in the cited passage but not together"
            )

    # 5 — real names only.
    known = {n.lower() for n in names.get("products", [])}
    known |= {n.lower() for n in names.get("colours", [])}
    known |= {n.lower() for n in names.get("merchants", [])}
    passage_blob = " ".join(h.chunk.content.lower() for h in hits)
    for candidate in _capitalised_runs(text):
        low = candidate.lower()
        if low in known or low in passage_blob:
            continue
        # The manufacturer's own name in front of a published product is not an
        # invented product. The site writes it both ways itself — "Lime Green
        # Solo" and "Lime Green Duro" are harvested, plain "Natural Finish" is
        # too — so which form ends up in the list is an accident of how each
        # page was written. Refusing "Lime Green Natural Finish" cost a correct,
        # fully cited, two-document answer on evaluation situation S9, which is
        # over-refusal rather than a caught invention. Only the prefix is
        # forgiven: "Lime Green Supercoat" still fails, because "Supercoat" is
        # not published.
        if _without_brand(low) in known or _without_brand(low) in passage_blob:
            continue
        if _looks_like_a_name(candidate):
            failures.append(f"check 5: {candidate!r} is not a name the site publishes")

    # 6 — the asked-for term appears in a passage the answer actually cites.
    #
    # It used to scan every retrieved passage, which is a weaker test than the
    # one the architecture describes and than decision 9 argues for: an answer
    # citing only [1] passed on a term that appeared only in uncited [2]. The
    # gate exists to catch the near-miss — a confident retrieval on the right
    # product and the wrong property — and evidence the answer did not rely on
    # cannot discharge it.
    if asked_terms:
        cited = {m for s in sentences for m in
                 (int(x) for x in _CITE.findall(s)) if m in cited_index}
        # Falling back to every hit when nothing was cited is deliberate: an
        # answer with no citations at all has already failed check 1, and
        # reporting check 6 as well would blame the wrong thing.
        considered = [cited_index[m] for m in cited] or hits
        blob = " ".join(f"{h.chunk.section} {h.chunk.content}"
                        for h in considered).lower()
        if not any(t.lower() in blob for t in asked_terms):
            failures.append("check 6: the property asked about is not in a cited passage")

    # 7 — the answer is still about the product that was asked about.
    #
    # The five checks above ask whether a claim is *supported*. None of them
    # asks whether it is about the right thing, and the gap between those two
    # is a real answer this system gave: asked "and what thickness should I
    # apply it at?" with Ultra carried from the previous turn, it replied "for
    # the general purpose Duro lime base coat, the first coat should be applied
    # between 9 to 12 mm thick" and **passed every check**.
    #
    # It passed honestly, which is what makes this worth a check of its own.
    # The sentence is supported by the passage it cites, the figure is verbatim
    # in that passage, and Duro is a real published product, so check 5 allows
    # it. Check 3 -- the one that exists to keep a figure with its product --
    # could not fire either, because it compares against the products of the
    # *retrieved passages* and no Duro passage was retrieved. Duro was named
    # inside the prose of a general FAQ answer about lime basecoats, and the
    # model copied it out. Citation correctness held; product correctness did
    # not.
    #
    # **Cited evidence only.** The allowed set grows from the passages the
    # answer actually relied on, not from everything retrieval happened to
    # return -- the same tightening check 6 already had, for the same reason.
    # Expanding it from all retrieved own-product passages was measured and is
    # too wide: on the failing turn it would have admitted Solo, Fine Stuff and
    # Natural Finish, and caught Duro only by the accident that no Ultra
    # passage mentions Duro.
    #
    # The third source is what keeps legitimate cross-product answers working.
    # Situation S8 asks which finish coats suit Forte, and the Forte
    # datasheet's own Finishing Coats section names Tradirend, Natural Finish
    # and Finish WP outright. A product the resolved product's own cited
    # evidence introduces is a product that evidence authorises, so naming it
    # back is not drift.
    #
    # Skipped entirely when nothing resolved a product, which is most of the
    # corpus's questions and all of the ones with no product to be about.
    if product:
        registry = names.get("products", [])
        cited = {m for s in sentences for m in
                 (int(x) for x in _CITE.findall(s)) if m in cited_index}
        allowed = {_without_brand(product.lower())}
        allowed |= {_without_brand(p.lower()) for p in asked_products if p}
        for marker in cited:
            hit = cited_index[marker]
            if product_matches(product, hit.chunk.product or ""):
                allowed |= products_named(hit.chunk.content, registry)
        # Compared with the shared containment rule rather than by equality,
        # so the catalogue's "Ultra Insulating Lime Render Base Coat" and a
        # caller's "Ultra" are one product and not two.
        strayed = sorted(named for named in products_named(text, registry)
                         if not any(product_matches(named, ok) for ok in allowed))
        if strayed:
            failures.append(
                f"check 7: the answer is about {strayed[0]!r}, which is not the "
                "product this question asked about")

    return list(dict.fromkeys(failures))


_CAP_RUN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-z]+)")


def _capitalised_runs(text: str) -> list[str]:
    """Candidate product, colour and merchant names in generated prose.

    A word opening a sentence is capitalised by grammar rather than by being a
    name, so those were dropped wholesale to avoid flagging ordinary prose. That
    exempted exactly the position a product name most often occupies — and the
    system prompt tells the model to "answer directly", which makes a name-first
    sentence the common shape. The result was that the same invention passed or
    failed on word order alone:

        "Supercoat is suitable for lath backgrounds [1]."       printed
        "For lath backgrounds use Supercoat [1]."               refused

    So a sentence-opening word is now dropped only when it is a single word that
    could plausibly be grammar. A multi-word run opening a sentence ("Supercoat
    Plus is…") is a name whatever its position, and a single word is still
    filtered afterwards by `_looks_like_a_name` and the `_COMMON` list.
    """
    stripped = _CITE.sub("", text)
    openings = {m.start(1) for m in _SENTENCE_START.finditer(stripped)}

    candidates = []
    for match in _CAP_RUN.finditer(stripped):
        run = match.group(0)
        if match.start() in openings:
            # Only at a sentence opening, where capitalisation is grammar
            # rather than evidence. Leading ordinary words are trimmed one at a
            # time rather than the run being discarded whole, so "Apply Solo"
            # becomes "Solo" instead of vanishing — and "Lime Green Solo" walks
            # down to "Solo" too, because both "lime" and "green" are listed.
            words = run.split()
            while words and words[0].lower() in _COMMON:
                words.pop(0)
            run = " ".join(words)
        if run:
            candidates.append(run)
    return candidates


_COMMON = {
    "lime", "lime green", "green", "the", "it", "however", "this", "these",
    "when", "where", "what", "if", "for", "use", "using", "apply", "mix",
    "water", "wall", "walls", "coat", "coats", "sources", "answer", "note",
}


# The manufacturer, as the site writes it. Not configuration: this is the one
# brand whose corpus this is, and a second brand appearing here would be a
# competitor, which the prompt forbids the model from mentioning at all.
_BRAND = "lime green"


def _without_brand(name: str) -> str:
    """A published product with the maker's name in front is still that product."""
    return name.removeprefix(_BRAND).strip() if name.startswith(_BRAND) else name


def _looks_like_a_name(candidate: str) -> bool:
    return candidate.lower() not in _COMMON and len(candidate) > 3


def _names_product(sentence: str, product: str) -> bool:
    head = product.split()[0] if product.split() else product
    return bool(re.search(rf"\b{re.escape(head)}\b", sentence, re.I))


# -------------------------------------------------------------- the renderer


def _source_rows(hits: list[Retrieved]) -> list[dict]:
    rows, seen = [], set()
    for i, h in enumerate(hits, 1):
        key = h.chunk.canonical_url
        rows.append({
            "marker": i,
            "name": h.document.citation_name,
            "section": h.chunk.section,
            "url": h.chunk.canonical_url,
            "type": h.document.document_type,
            "date": h.chunk.source_date,
            "score": round(h.score, 3),
            "first": key not in seen,
        })
        seen.add(key)
    return rows


# Documents whose job is to describe a product rather than to explain why a
# wall went wrong. A product page says a render is durable and available in
# twenty-four colours; it cannot say why this one dried patchy.
_MARKETING_TYPES = ("product_page", "commercial")


def _diagnostic_passages(hits: list[Retrieved], limit: int = 3) -> list[Retrieved]:
    """The passages a diagnosis hand-off should quote, out of what was retrieved.

    The hand-off's own sentence promises "what the site does publish on this",
    so the passages under it have to be the ones that actually bear on the
    symptom. Taking the top three by similarity does not deliver that, and the
    failure is systematic rather than occasional: on a question about patchy
    colour after drying, the three highest-scoring passages were two coloured
    render product pages and a rendering checklist, while the technical note
    that publishes the mechanism — "an even colour is the result of an even
    drying rate which in turn is the result of an even application thickness",
    together with damp patches, trowel pressure, day joints and strong sunlight
    — sat fifth and was cut. Retrieval had found it. The renderer threw it away.

    Two things push the wrong way at once. A product page repeats the words of
    the symptom because it is selling a product the symptom belongs to, so it
    scores highly on similarity alone; and `AUTHORITY` ranks `product_page`
    above `knowledge_base`, so among near-ties the marketing copy is preferred
    on purpose. Both are right for a factual lookup about a product and both
    are backwards for a question about a defect.

    So this is a stable partition rather than a re-scoring: passages keep their
    similarity order inside each group, and the only thing that changes is that
    a page selling a product cannot displace a document explaining a failure.
    Nothing is invented and nothing outside the retrieved set is reached for —
    if the corpus published nothing but product pages, product pages are what
    prints, exactly as before.

    The residual limit is worth stating: this can only choose among the
    passages retrieval returned, so a technical note that never entered the
    window is still lost. Widening the window for this path is a separate
    change with its own evidence to gather.
    """
    ordered = sorted(hits, key=lambda h: h.document.document_type in _MARKETING_TYPES)
    return ordered[:limit]


def _contact_line(names: dict) -> str:
    contact = names.get("contact", {}) or {}
    phone, hours = contact.get("phone", ""), contact.get("hours", "")
    if phone and hours:
        return f"Lime Green's technical team: {phone} ({hours})."
    if phone:
        return f"Lime Green's technical team: {phone}."
    return "Contact Lime Green's technical team via https://www.lime-green.co.uk/contact."


def _caveat_lines(decision: Decision, repo, question: str = "",
                  limit: int = 3) -> list[str]:
    """At most three document caveats, from the documents actually cited.

    Two constraints, both learned the hard way. Only documents whose passages
    were printed contribute, because a caveat from the fifth-ranked document is
    a caveat about something the reader was not told about — an early version
    appended Roman Stucco's curing limits to an answer about Solo. And the
    question itself scores them, not the slot values: matching on the word
    'water' alone ranks almost nothing.
    """
    cited = dict.fromkeys(h.chunk.canonical_url for h in decision.hits[:2])
    wanted = _words(question) | _words(" ".join(decision.slots.values()))
    scored = []
    # The same sentence, once. Several products ship two near-identical
    # datasheets — Ultra has "Insulating" and "Insulated" versions of the same
    # PDF — so a question answered from both printed "Lightly spray the Ultra in
    # hot weather" twice, which reads as a defect in the assistant rather than
    # as two documents agreeing. Deduplication is on the sentence rather than on
    # the document, because that is the thing the reader sees repeated.
    seen: set[str] = set()
    for url in cited:
        for cav in repo.caveats(url):
            key = " ".join(cav.sentence.lower().split())
            if key in seen:
                continue
            seen.add(key)
            overlap = len(_words(cav.sentence) & wanted)
            scored.append((overlap, len(cav.sentence), cav.sentence))
    # Prefer an overlapping caveat; failing that, the shortest, which is the
    # most likely to be a bare limit rather than a paragraph of context.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [s for _o, _n, s in scored[:limit]]


PHOTO_LINE = (
    "I cannot see photographs. To get this right, the two things worth telling "
    "me are what the wall is built of underneath, and whether it is inside or "
    "outside."
)


# ------------------------------------------------------------------- engine


class AnswerEngine:
    """Turns a router decision into something printable."""

    def __init__(self, repo, retriever, generation_model: str = ollama.GENERATION_MODEL):
        self.repo = repo
        self.retriever = retriever
        self.model = generation_model
        notes = repo.snapshot().notes if repo.snapshot() else {}
        self.names = {
            "products": notes.get("products", []),
            "colours": notes.get("colours", []),
            "merchants": notes.get("merchants", []),
            "contact": notes.get("contact", {}),
        }

    # -- the paths ---------------------------------------------------------

    def extract(self, decision: Decision, question: str = "") -> Answer:
        """Print the top passage whole, by code. No model, no paraphrase."""
        hits = decision.hits
        if decision.sum_refused:
            # The calculation edge asks for coverage, so print the passage that
            # carries it rather than whichever passage ranked first. Retrieval
            # ranks on the whole question, and "how many bags for 20 square
            # metres of Duro" puts Duro's *mixing water* on top — so the printed
            # evidence was the wrong half of the answer while the sentence below
            # claimed it was the right one. Selection is on content, not on the
            # heading: Duro publishes its coverage under a "Storage" heading.
            hits = sorted(hits, key=lambda h: not _mentions_coverage(h.chunk.content))

        top = hits[0]
        body = [f"From {top.document.citation_name}"
                f"{', ' + top.chunk.section if top.chunk.section else ''} [1]:",
                "", top.chunk.content]
        if decision.sum_refused:
            # Say what was actually printed. Claiming to have shown coverage
            # when the retrieved evidence does not contain any is the kind of
            # unsupported sentence the six checks exist to stop, and it arrived
            # here by a different door — written by code, so never checked.
            if _mentions_coverage(top.chunk.content):
                body += ["", "I have printed the published coverage and pack size rather "
                              "than multiplying them out. The figure that matters on site "
                              "depends on the background and the thickness, so the "
                              "arithmetic is worth doing against your own measurements."]
            else:
                body += ["", "I will not multiply this out, and the indexed material does "
                              "not state a coverage figure for it, so there is nothing "
                              "published to do the arithmetic against. "
                              + _contact_line(self.names)]
        return self._finish(decision, "\n".join(body), hits[:1], question)

    def compose(self, decision: Decision, question: str,
                history: str = "", guidance: str = "") -> Answer:
        """The one path the model runs on.

        `history` is the visible transcript, and this is the only place in the
        system it is allowed to reach. It arrives as its own argument rather
        than merged into the question because everything that decides *what
        this answer is* -- the policy gate, slot detection, product detection,
        the router and retrieval -- has already run on the question alone. What
        is left for a transcript to do is the one thing it is genuinely needed
        for: telling the model that "it" means the Ultra from two turns ago.

        It cannot become a fact. It is delimited as non-evidence, it carries no
        citation marker, and check 1 discards any sentence whose clauses do not
        overlap a passage the sentence cites. A model lifting a figure out of
        its own earlier answer would be writing an uncitable sentence, and an
        uncitable sentence does not print.
        """
        # One list, from here to the printed sources. Markers are positional,
        # so the prompt, `run_checks` and `_source_rows` must all count the
        # same order or the verification renumbers underneath the answer.
        #
        # `decision.hits` is deliberately *not* reordered. Three things read it
        # directly and all three should keep reading retrieval's own ranking:
        # the document caveats, the `top_score` in the diagnostics, and the
        # closest-guidance passage a refusal prints. Promotion is about what
        # the model reads, not about what retrieval found.
        hits = promote_bound(decision.hits, evidence_binding(decision))
        passages = "\n\n".join(
            f"[{i}] {h.document.citation_name}"
            f"{' — ' + h.chunk.section if h.chunk.section else ''}\n{h.chunk.content}"
            for i, h in enumerate(hits, 1)
        )
        context = self._prompt_context(decision)
        # Computed here rather than passed in, so every caller of Compose gets
        # it — the ordinary answering path, which passes no guidance at all, is
        # the one the binding was written for. A caller's own guidance is kept
        # and the binding is added under it: `recommend` names the approved
        # products, and which passage states which property is a different
        # instruction that does not replace it.
        #
        # Built against the promoted order, so any marker it names is the
        # marker the prompt used. Today promotion and the binding block are
        # mutually exclusive — one bound property suppresses the block, two
        # suppress promotion — but relying on that would be a trap for whoever
        # relaxes either rule next.
        binding = _binding_guidance(replace(decision, hits=hits))
        guidance = "\n".join(g for g in (guidance, binding) if g)
        prompt = PROMPT.format(
            passages=passages, question=question,
            history=(HISTORY_BLOCK.format(history=history.strip())
                     if history.strip() else ""),
            assumptions=("\nStated assumptions: " + "; ".join(context)
                         if context else ""),
            guidance=("\n" + guidance if guidance else ""),
        )
        # The one span in the system where a model runs on the answering path,
        # and the only one whose duration is ever the whole answer's duration.
        # OTel's `gen_ai.*` names, because this is the field they were written
        # for. No prompt and no generated text: a span may carry counts, and
        # `answer_log` is where the question is deliberately retained.
        with obs.span("generation",
                      **{"gen_ai.request.model": self.model,
                         "passages": len(hits),
                         "prompt_chars": len(prompt)}) as generation:
            try:
                text, seconds = ollama.generate(prompt, model=self.model,
                                                system=SYSTEM)
            except ollama.OllamaUnavailable as error:
                obs.event("ollama_error", stage="generate", model=self.model,
                          error=type(error).__name__, detail=str(error))
                raise
            # Words, not tokens. The OTel field is named for tokens and this
            # is the honest thing the code can count: Ollama's response carries
            # no usage block through the client this system uses, and inventing
            # a token count from a word count would be a number that looked
            # like a measurement. The name is kept so an exporter later needs no
            # rename; what it holds is documented here rather than implied.
            generation["gen_ai.usage.output_tokens"] = len(text.split())
            generation["seconds"] = round(seconds, 2)
        # The `generation` event this span replaced carried model, seconds,
        # passages, prompt_chars and answer_words. All five are attributes of
        # the span, which emits one line on completion, so emitting both would
        # have been the same fact twice under the same event name.

        # The words the router's relevance gate accepted at step 4, carried on
        # the decision. Check 6 is the second half of that gate and has to ask
        # the same question step 4 asked, or a two-property enquiry passes one
        # half of one gate and is refused by the other. See
        # `Decision.asked_terms` for the case that proved it.
        #
        # The old derivation is kept as a fallback rather than removed: a
        # `Decision` built directly — by a test, or by any caller that did not
        # come through `Router.route` — carries no terms, and a check 6 that
        # silently stopped running would be a safety check turning itself off.
        # Falling back narrows the gate; it never removes it.
        asked = decision.slots.get("property_asked", "")
        terms = list(getattr(decision, "asked_terms", None) or [])
        if not terms and hasattr(self.retriever, "slots"):
            terms = self.retriever.slots.terms_for("property_asked", asked)
        # A span around the checks and not inside them. The six checks are a
        # safety boundary and this slice observes them rather than touching
        # them: `run_checks` is called exactly as before and decides exactly
        # what it decided before. What the span adds is how long they cost and
        # which ones fired, which is how over-refusal becomes countable.
        # The product this answer is supposed to be about, and the products the
        # caller named themselves. Check 7 reads both; nothing else does. Taken
        # from the question rather than from retrieval, because "can I use Ultra
        # over Solo" authorises Solo whether or not a Solo passage was ranked.
        scope = decision.slots.get("product", "")
        asked_products = tuple(products_named(question,
                                              self.names.get("products", [])))
        with obs.span("checks", count=7, product=scope) as checking:
            failures = run_checks(text, hits, self.names, terms,
                                  product=scope, asked_products=asked_products)
            # The check *numbers*, not the failure messages. The review's table
            # asks for "check numbers and text" and the text cannot come: check
            # 1 quotes seventy characters of the offending sentence and check 5
            # quotes the invented name, so a span carrying the messages would
            # carry generated answer text into a table forbidden to hold any.
            # The number is what the operational question needs anyway — which
            # check fires most often — and the message is already on the
            # `check_failed` event and in `answer_log.check_failed`, where
            # retention was decided deliberately.
            checking["failed"] = sorted({f.split(":", 1)[0] for f in failures})
            checking["failures"] = len(failures)
            checking["passed"] = not failures

        if failures:
            # The single most operationally important line in the system: it is
            # how over-refusal is noticed, and which check is responsible.
            obs.event("check_failed", checks=failures, model=self.model,
                      passages=len(hits))
            answer = self.refuse(decision,
                                 "the generated answer did not pass its checks",
                                 question)
            answer.failed_checks = failures
            answer.diagnostics["generation_seconds"] = round(seconds, 2)
            return answer

        answer = self._finish(decision, text, hits, question)
        answer.diagnostics["generation_seconds"] = round(seconds, 2)
        return answer

    def defer(self, decision: Decision, question: str = "") -> Answer:
        top = decision.hits[0]
        text = (
            "Lime Green's own published material refers this question to their "
            f"technical team rather than answering it [1]:\n\n"
            f"“{top.chunk.content.strip()}”\n\n{_contact_line(self.names)}"
        )
        return self._finish(decision, text, decision.hits[:1], question)

    def diagnosis(self, decision: Decision, question: str = "") -> Answer:
        chosen = _diagnostic_passages(decision.hits)
        published = "\n\n".join(
            f"[{i}] {h.document.citation_name}"
            f"{' — ' + h.chunk.section if h.chunk.section else ''}\n{h.chunk.content}"
            for i, h in enumerate(chosen, 1)
        )
        text = (
            "Working out what has actually gone wrong with a wall is a judgement "
            "for Lime Green's technical team, not something to settle from "
            "published documents. What the site does publish on this is below, "
            "and it is worth reading before you call.\n\n"
            f"{published}\n\n{_contact_line(self.names)}"
        )

        # Capture diagnosis hand-off for the failure library (fire-and-forget)
        try:
            capture = DiagnosisCapture()
            chunk_ids = [h.chunk.chunk_id for h in chosen]
            tags = [slot for slot in ["symptom", "cause_asked"] if slot in decision.slots]
            capture.capture(
                question=question,
                images=[],
                chunk_ids=chunk_ids,
                refusal_reason="diagnosis_handed_off",
                tags=tags,
            )
        except Exception as e:
            obs.event("diagnosis_capture_error", error=str(e))

        return self._finish(decision, text, chosen, question)

    def ask_back(self, decision: Decision, question: str = "") -> Answer:
        text = (
            "I need one more detail before pointing you at a product. What is the "
            "wall built of underneath — brick, stone, cob, laths, plasterboard, "
            "or an existing plaster or render?\n\n"
            "That decides the answer more than anything else, and the published "
            "material gives different products for each.\n\n"
            f"{_contact_line(self.names)}"
        )
        answer = self._finish(decision, text, decision.hits[:3], question)
        answer.refused = False
        return answer

    def route(self, topic: str, spec: dict) -> Answer:
        contact = self.names.get("contact", {}) or {}
        substitutions = {
            "contact": contact.get("phone", "the number on the contact page"),
            "hours": contact.get("hours", "office hours"),
        }
        text = spec["referral"]
        step = spec["next_step"].format(**substitutions)
        return Answer(
            text=f"{text}\n\n{step}",
            path=Path_.ROUTE.value,
            diagnostics={"topic": topic, "step": "policy gate",
                         "grounded_in": spec.get("grounded_in", "")},
        )

    def documents_for(self, question: str, audiences: tuple[str, ...]) -> Answer:
        """A document request is answered from the manifest, not by retrieval."""
        manifest = self.repo.manifest(audiences)
        words = _words(question)
        scored = [(len(_words(d.citation_name + " " + d.product) & words), d)
                  for d in manifest]
        scored = [(s, d) for s, d in scored if s]
        scored.sort(key=lambda t: (-t[0], t[1].authority))
        picks = [d for _s, d in scored[:6]]
        if not picks:
            picks = [d for d in manifest if d.document_type == "datasheet"][:6]

        lines = ["These are the published documents that match, from the index:", ""]
        for d in picks:
            version = self.repo.active_version(d.canonical_url)
            when = f" — fetched {version.fetched_at[:10]}" if version else ""
            lines.append(f"- {d.citation_name} ({d.document_type.replace('_', ' ')})"
                         f"{when}\n  {d.canonical_url}")

        excluded = self.repo.excluded()
        if re.search(r"\b(sds|msds|safety data sheet|dop|epd)\b", question, re.I):
            lines += ["", "Safety data sheets, declarations of performance and "
                          "environmental declarations are deliberately not indexed: "
                          "they are controlled documents that have to be read whole "
                          "and current, not quoted in fragments. They are on each "
                          "product page.",
                      f"({len(excluded)} documents are excluded by rule and recorded "
                      "in the index.)"]
        lines += ["", _contact_line(self.names)]
        return Answer("\n".join(lines), Path_.ROUTE.value,
                      diagnostics={"step": "manifest", "topic": "document_request"})

    def refuse(self, decision: Decision, why: str, question: str = "") -> Answer:
        """A refusal that hands over everything it has, without reading like a dump.

        The shape changed after review; what it carries did not. It used to open
        with "I could not find this", then announce the closest passage, then
        print up to 600 characters of raw datasheet — so a public visitor asking
        one short question met a wall of text whose first useful line was the
        phone number at the bottom. The objection is a presentation one and it
        is right: the summary should say what was looked for and what the
        nearest guidance actually is, in a sentence.

        Nothing is dropped, because every element a refusal is required to carry
        is still here. What was looked for: the missing term, named. What *is*
        published, with its source: the nearest document and section named in
        the summary, its passage quoted in `disclosure`, and its row in
        `sources`. The document's own caveats: appended by `_finish`, as before.
        The contact line: from the manifest, never typed. The passage stays
        inside `text` so the CLI transcript — canonical evidence — loses
        nothing; `disclosure` repeats it so the page can fold it behind "Show
        source passage" rather than open with it.
        """
        obs.event("refusal", why=why, step=decision.step,
                  missing_term=decision.missing_term,
                  top_score=decision.hits[0].score if decision.hits else 0.0)

        top = decision.hits[0] if decision.hits else None
        # "Solo Onecoat Lime Plaster datasheet, Mixing" — enough for the reader
        # to know which document was nearest without reading it first.
        where = ""
        if top is not None:
            where = top.document.citation_name
            if top.chunk.section:
                where += f", {top.chunk.section}"

        if top is not None and decision.missing_term:
            summary = (
                "I could not find published Lime Green guidance that states "
                f"{decision.missing_term} for this product. The closest guidance is "
                f"{where}, which does not state {decision.missing_term}. I would "
                "rather say that than "
                "give you a figure from a neighbouring document."
            )
        elif top is not None:
            summary = (
                "I could not find published Lime Green guidance that specifically "
                f"answers this. The closest guidance is {where}, which is related "
                "but does not answer the question, so I am not going to answer it "
                "from anything else."
            )
        else:
            summary = (
                "I could not find this in Lime Green's published material, and "
                "nothing retrieved is close enough to be worth showing you, so I "
                "am not going to answer it from anything else."
            )

        parts = [summary, "", "Please check with Lime Green's technical team.",
                 "", _contact_line(self.names)]

        disclosure = ""
        if top is not None:
            # Still capped at 600 characters, exactly as before: the whole of a
            # 3,600-character section is a scroll, not a disclosure.
            disclosure = (f"Source passage — {where}:\n"
                          f"“{top.chunk.content.strip()[:600]}”")

        answer = self._finish(decision, "\n".join(parts),
                              decision.hits[:1], question)
        # Appended last, and only here, so it is always the suffix of `text`:
        # that is what lets `Answer.body` hand a surface the prose without the
        # evidence, with no second copy of the parsing.
        if disclosure:
            answer.text = f"{answer.text}\n\n{disclosure}"
        answer.disclosure = disclosure
        # The path recorded is the path actually taken. A compose that failed
        # its checks ends in a refusal, and reporting that as "compose" in the
        # transcript would describe the attempt rather than the outcome.
        answer.path = Path_.REFUSE.value
        answer.refused = True
        answer.diagnostics["refusal_reason"] = why
        return answer

    # -- shared ------------------------------------------------------------

    # ------------------------------------------------------ recommendation

    def _select_decision(self, resolved, hits, path, step, reason) -> Decision:
        """A router `Decision` for a selection, so `_finish` works unchanged.

        Provenance travels in `origins`, which is what lets a recommendation
        print "brick (substrate), from the photograph you sent" instead of
        attributing a model's reading of an image to the person.
        """
        return Decision(path, reason, step, slots=resolved.slots(),
                        origins=dict(resolved.provenance), hits=list(hits))

    def recommend(self, resolved, decision, hits, history: str = "") -> Answer:
        """Choose from a set the evidence has already approved.

        Two properties make putting a model here defensible, and neither of them
        is the prompt.

        The **passages are filtered to the approved products' own documents**
        before generation, so the model physically cannot cite a Warmshell
        figure in support of Ultra -- no such passage is in front of it. Check 3
        would catch that attribution afterwards; removing the opportunity is
        better than catching the attempt.

        And the answer is produced by `compose`, so the **six checks run
        unchanged**. A recommendation is not a privileged kind of answer that
        skips them. It is an ordinary composed answer whose candidate set was
        settled first. Containment -- that no product outside the approved set
        is named -- is checked after this returns, because it is a property of
        the finished text rather than of the generation.
        """
        approved = decision.approved_names
        wanted = {p.lower() for p in approved}
        supporting = [h for h in hits
                      if (getattr(h.chunk, "product", "") or "").lower() in wanted]
        supporting = supporting or list(hits)

        names = ", ".join(sorted(approved))
        conditions = [c for a in decision.approved for c in a.caveats]
        unknown = sorted({p for a in decision.approved
                          for p in a.unsupported_properties})

        guidance = (
            "Recommend only from these products: " + names + ". Do not name any "
            "other product. If more than one is listed, say which you would use "
            "and what the other is for."
        )
        if unknown:
            guidance += (" The passages do not state " + ", ".join(unknown)
                         + " for these products, so do not state it either.")

        router_decision = self._select_decision(
            resolved, supporting, Path_.SELECT, "4b",
            "a product was chosen from " + str(len(approved))
            + " evidence-supported candidate(s)")
        answer = self.compose(router_decision, resolved.raw_question,
                              history=history, guidance=guidance)
        answer.diagnostics["approved"] = sorted(approved)
        answer.diagnostics["rejected"] = [
            {"product": a.product, "status": a.status.value,
             "reason": a.rejected_because} for a in decision.rejected]
        answer.diagnostics["outcome"] = decision.outcome.value
        answer.diagnostics["evidence"] = {
            a.product: a.evidence for a in decision.approved}
        if conditions:
            answer.caveats = list(
                dict.fromkeys(list(answer.caveats) + conditions))[:3]
        return answer

    def need_more_information(self, resolved, decision) -> Answer:
        """Ask for the minimum, and say why it decides the answer.

        Named for what it is rather than folded into `ask_back`, because the two
        differ in what they can say. `ask_back` asks the single fixed question
        decision 10 privileges; this names whichever facts *this* job requires,
        which for an external render includes the exposure and for a repair
        includes neither.
        """
        missing = list(decision.missing)
        disputed = [slot for slot in missing if slot in resolved.unsettled]
        head = missing[0] if missing else ""

        asked = _MISSING_PHRASE.get(head, "one more detail about the wall.")
        text = ("I need one more detail before I can point you at a product. "
                + asked[0].upper() + asked[1:])

        if disputed:
            # The conflict is quoted back rather than settled silently. The
            # person is the only one who can settle it, and showing them both
            # readings is what makes it answerable in one reply.
            text += (chr(10) * 2 + "The photograph and what you told me do not "
                     "agree about the " + disputed[0]
                     + ", so I would rather ask than guess.")
        if len(missing) > 1:
            text += (chr(10) * 2 + "It would also help to know the "
                     + " and the ".join(missing[1:]) + ".")
        text += chr(10) * 2 + _contact_line(self.names)

        router_decision = self._select_decision(
            resolved, [], Path_.ASK_BACK, "4b",
            "a product cannot be chosen without: " + ", ".join(missing))
        answer = self._finish(router_decision, text, [], resolved.raw_question)
        answer.diagnostics["outcome"] = decision.outcome.value
        answer.diagnostics["missing"] = missing
        return answer

    def no_supported_recommendation(self, resolved, decision,
                                    why: str = "") -> Answer:
        """The corpus does not establish suitability, so nothing is recommended.

        A refusal rather than a best guess, and a refusal that still carries
        what is published, which is the shape every other refusal in this system
        takes. What it adds is the list of products actually considered and why
        each was set aside -- because "nothing is suitable" and "the sheets do
        not say" are different statements, and the second is the true one.
        """
        reason = why or decision.reason or (
            "the published material does not establish that any product suits "
            "this background")
        text = ("I cannot recommend a product for this: " + reason + "."
                + chr(10) * 2
                + "I would rather say so than put a product on a wall the "
                "published material does not cover.")

        considered = [a for a in decision.rejected if a.rejected_because]
        if considered:
            text += chr(10) * 2 + "What I looked at:" + chr(10) + chr(10).join(
                "- " + a.product + ": " + a.rejected_because
                for a in considered[:4])
        text += chr(10) * 2 + _contact_line(self.names)

        router_decision = self._select_decision(
            resolved, [], Path_.REFUSE, "4b", reason)
        answer = self._finish(router_decision, text, [], resolved.raw_question)
        answer.refused = True
        answer.diagnostics["outcome"] = decision.outcome.value
        answer.diagnostics["rejected"] = [
            {"product": a.product, "status": a.status.value,
             "reason": a.rejected_because} for a in decision.rejected]
        return answer

    def _prompt_context(self, decision: Decision) -> list[str]:
        """What the model is told about the caller's building, for the prompt only.

        Deliberately unchanged in wording from the list this used to produce,
        because it is part of a prompt whose output is meant to be reproducible.
        Provenance belongs to the reader, not to the model: the model needs to
        know the wall is brick, not who said so.
        """
        out = []
        if decision.per_option:
            out.append("answered for both internal and external use, "
                       "since you did not say which")
        for slot, value in decision.slots.items():
            if slot in STATABLE_SLOTS:
                out.append(f"{slot}: {value.replace('_', ' ')}")
        return out

    def _detected(self, question: str) -> tuple[set[str], bool]:
        """Which slots this question states in its own words, if that is knowable.

        The router owns the vocabulary and is handed in whole rather than
        copied, so this asks the same detector that produced the slots in the
        first place. The boolean says whether the answer means anything: with no
        question text there is nothing to detect from, and every slot is then
        reported as stated — which is true, because nothing in this engine ever
        invents a slot value. Every value in `decision.slots` was either read
        off this question or handed in by a caller the person told. What is lost
        without the question is only the distinction between the two, and "as
        you said" is the weaker, still-honest reading of both.

        `assistant/engine.py` passes the question on the compose path today. The
        other paths would each report a carried slot as carried the moment it
        passes the question there too; that file belongs to the integration
        owner, so the parameter is optional and the fallback is the safe one.
        """
        if not question or not hasattr(self.retriever, "slots"):
            return set(), False
        return set(self.retriever.slots.detect(question)), True

    def _facts(self, decision: Decision, question: str) -> list[SlotFact]:
        """Every slot value behind this answer, each with where it came from.

        Two sources of truth about provenance, and they are in a deliberate
        order. Detection from the question text decides ``STATED``, because a
        value present in this sentence was said in this sentence whatever else
        also supplied it — the router merges `carried` *under* the question for
        the same reason, and a caller who uploads a photograph of a stone wall
        and then types "it's brick" is correcting the image rather than being
        corrected by it.

        Everything else reads `decision.origins`, the parallel slot-to-origin
        mapping the engine threads in beside `carried`. This cannot be
        re-derived here and must not be guessed at: the whole property of an
        observed slot is that it is *not* in the question, which is precisely
        the shape a carried slot has too. The two are indistinguishable from
        the text, so the origin travels with the value or it is lost.

        The mapping is sparse and defaults to ``CARRIED``. That is what keeps
        every existing caller — the CLI, the evaluation harness, the web page
        with no upload — behaving exactly as it did before this parameter
        existed, rather than being migrated by a change nobody asked for.
        """
        said, knowable = self._detected(question)
        facts = []
        for slot in STATABLE_SLOTS:
            value = decision.slots.get(slot)
            if value:
                stated = not knowable or slot in said
                origin = decision.origins.get(slot)
                if origin is not None and not (knowable and slot in said):
                    provenance = origin
                else:
                    provenance = (Provenance.STATED if stated
                                  else Provenance.CARRIED)
                facts.append(SlotFact(slot, value, provenance))
        if decision.per_option:
            # The one genuine assumption the system currently makes: decision 10
            # answers both options because neither was cued.
            facts.append(SlotFact("location", "internal and external",
                                  Provenance.ASSUMED))
        return facts

    def _finish(self, decision: Decision, text: str, hits: list[Retrieved],
                question: str = "") -> Answer:
        caveats = _caveat_lines(decision, self.repo, question)
        facts = self._facts(decision, question)
        if decision.photograph:
            text = f"{text}\n\n{PHOTO_LINE}"
        # What the caller told us is repeated back as a sentence inside the
        # answer, where it reads as the system having listened. What the system
        # genuinely assumed stays in `assumptions`, which every surface prints
        # under a heading saying "Assumed" — now truthfully.
        #
        # Three groups, not two. An observed value is neither something the
        # caller said nor something the system assumed, and collapsing it into
        # either prints a false sentence: "as you told me" attributes a model's
        # reading to a person, and "assumed, since you did not say" calls a
        # photograph a guess. So it gets its own line, and it is kept out of
        # `assumptions` — the list every surface prints under a heading reading
        # "Assumed", which must stay true of everything under it.
        told = [f.sentence for f in facts
                if f.provenance in (Provenance.STATED, Provenance.CARRIED)]
        seen = [f.sentence for f in facts if f.provenance is Provenance.OBSERVED]
        if told:
            text = f"{text}\n\nAnswered for " + "; ".join(told) + "."
        if seen:
            # "Also" only when there is something for it to be additional to.
            # The phrase itself already ends in "from the photograph you sent",
            # so the lead-in deliberately does not repeat the photograph.
            lead = "Also answered for" if told else "Answered for"
            text = f"{text}\n\n{lead} " + "; ".join(seen) + "."
        return Answer(
            text=text,
            path=decision.path.value,
            sources=_source_rows(hits),
            caveats=caveats,
            assumptions=[f.sentence for f in facts
                         if f.provenance is Provenance.ASSUMED],
            observed=seen,
            facts=facts,
            diagnostics={
                "step": decision.step,
                "reason": decision.reason,
                "slots": decision.slots,
                "top_score": round(decision.hits[0].score, 3) if decision.hits else 0.0,
                "chunk_ids": [h.chunk.chunk_id for h in hits],
            },
        )
