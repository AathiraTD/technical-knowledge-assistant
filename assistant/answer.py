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
from dataclasses import dataclass, field

from . import observability as obs
from . import ollama
from .model import Retrieved
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
    "- Use only the passages. If they do not answer the question, say so in "
    "one sentence.\n"
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

Passages:
{passages}

Question: {question}
{assumptions}
Answer:"""


# ---------------------------------------------------------------- the result


@dataclass
class Answer:
    """What the caller gets, including how it was arrived at."""

    text: str
    path: str
    sources: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    refused: bool = False


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

# How close a qualifier has to sit to its figure in the passage to count as
# attached. A sentence is the natural unit, and a datasheet sentence runs to
# roughly this length; wider and a qualifier from the previous sentence starts
# to count, which is the failure this window exists to stop.
QUALIFIER_WINDOW = 60

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


def run_checks(
    text: str,
    hits: list[Retrieved],
    names: dict,
    asked_terms: list[str],
) -> list[str]:
    """The six checks, in order. Returns the failures; empty means it prints."""
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
        near = any(abs(w.start() - f.start()) <= QUALIFIER_WINDOW
                   for w in re.finditer(re.escape(word), cited_text)
                   for f in re.finditer(re.escape(digits.group(0)), cited_text))
        if not near:
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
    for url in cited:
        for cav in repo.caveats(url):
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

    def extract(self, decision: Decision) -> Answer:
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
        return self._finish(decision, "\n".join(body), hits[:1])

    def compose(self, decision: Decision, question: str) -> Answer:
        """The one path the model runs on."""
        hits = decision.hits
        passages = "\n\n".join(
            f"[{i}] {h.document.citation_name}"
            f"{' — ' + h.chunk.section if h.chunk.section else ''}\n{h.chunk.content}"
            for i, h in enumerate(hits, 1)
        )
        assumptions = self._assumptions(decision)
        prompt = PROMPT.format(
            passages=passages, question=question,
            assumptions=("\nStated assumptions: " + "; ".join(assumptions)
                         if assumptions else ""),
        )
        try:
            text, seconds = ollama.generate(prompt, model=self.model,
                                            system=SYSTEM)
        except ollama.OllamaUnavailable as error:
            obs.event("ollama_error", stage="generate", model=self.model,
                      error=type(error).__name__, detail=str(error))
            raise
        obs.event("generation", model=self.model, seconds=round(seconds, 2),
                  passages=len(hits), prompt_chars=len(prompt),
                  answer_words=len(text.split()))

        asked = decision.slots.get("property_asked", "")
        terms = (self.retriever.slots.terms_for("property_asked", asked)
                 if hasattr(self.retriever, "slots") else [])
        failures = run_checks(text, hits, self.names, terms)

        if failures:
            # The single most operationally important line in the system: it is
            # how over-refusal is noticed, and which check is responsible.
            obs.event("check_failed", checks=failures, model=self.model,
                      passages=len(hits))
            answer = self.refuse(decision,
                                 "the generated answer did not pass its checks")
            answer.failed_checks = failures
            answer.diagnostics["generation_seconds"] = round(seconds, 2)
            return answer

        answer = self._finish(decision, text, hits, assumptions, question)
        answer.diagnostics["generation_seconds"] = round(seconds, 2)
        return answer

    def defer(self, decision: Decision) -> Answer:
        top = decision.hits[0]
        text = (
            "Lime Green's own published material refers this question to their "
            f"technical team rather than answering it [1]:\n\n"
            f"“{top.chunk.content.strip()}”\n\n{_contact_line(self.names)}"
        )
        return self._finish(decision, text, decision.hits[:1])

    def diagnosis(self, decision: Decision) -> Answer:
        published = "\n\n".join(
            f"[{i}] {h.document.citation_name}"
            f"{' — ' + h.chunk.section if h.chunk.section else ''}\n{h.chunk.content}"
            for i, h in enumerate(decision.hits[:3], 1)
        )
        text = (
            "Working out what has actually gone wrong with a wall is a judgement "
            "for Lime Green's technical team, not something to settle from "
            "published documents. What the site does publish on this is below, "
            "and it is worth reading before you call.\n\n"
            f"{published}\n\n{_contact_line(self.names)}"
        )
        return self._finish(decision, text, decision.hits[:3])

    def ask_back(self, decision: Decision) -> Answer:
        text = (
            "I need one more detail before pointing you at a product. What is the "
            "wall built of underneath — brick, stone, cob, laths, plasterboard, "
            "or an existing plaster or render?\n\n"
            "That decides the answer more than anything else, and the published "
            "material gives different products for each.\n\n"
            f"{_contact_line(self.names)}"
        )
        answer = self._finish(decision, text, decision.hits[:3])
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

    def refuse(self, decision: Decision, why: str) -> Answer:
        obs.event("refusal", why=why, step=decision.step,
                  missing_term=decision.missing_term,
                  top_score=decision.hits[0].score if decision.hits else 0.0)
        parts = []
        if decision.missing_term:
            parts.append(
                f"The indexed material does not state {decision.missing_term} for "
                "this product. I would rather say that than give you a figure from "
                "a neighbouring document."
            )
        else:
            parts.append(
                "I could not find this in Lime Green's published material, so I am "
                "not going to answer it from anything else."
            )

        if decision.hits:
            top = decision.hits[0]
            parts += [
                "",
                "The closest published passage is this, which does not answer the "
                "question but may still be useful:",
                "",
                f"From {top.document.citation_name}"
                f"{', ' + top.chunk.section if top.chunk.section else ''}:",
                f"“{top.chunk.content.strip()[:600]}”",
            ]
        parts += ["", _contact_line(self.names)]

        answer = self._finish(decision, "\n".join(parts), decision.hits[:1])
        # The path recorded is the path actually taken. A compose that failed
        # its checks ends in a refusal, and reporting that as "compose" in the
        # transcript would describe the attempt rather than the outcome.
        answer.path = Path_.REFUSE.value
        answer.refused = True
        answer.diagnostics["refusal_reason"] = why
        return answer

    # -- shared ------------------------------------------------------------

    def _assumptions(self, decision: Decision) -> list[str]:
        out = []
        if decision.per_option:
            out.append("answered for both internal and external use, "
                       "since you did not say which")
        for slot, value in decision.slots.items():
            if slot in ("substrate", "location", "exposure"):
                out.append(f"{slot}: {value.replace('_', ' ')}")
        return out

    def _finish(self, decision: Decision, text: str, hits: list[Retrieved],
                assumptions: list[str] | None = None, question: str = "") -> Answer:
        caveats = _caveat_lines(decision, self.repo, question)
        if decision.photograph:
            text = f"{text}\n\n{PHOTO_LINE}"
        return Answer(
            text=text,
            path=decision.path.value,
            sources=_source_rows(hits),
            caveats=caveats,
            assumptions=assumptions or self._assumptions(decision),
            diagnostics={
                "step": decision.step,
                "reason": decision.reason,
                "slots": decision.slots,
                "top_score": round(decision.hits[0].score, 3) if decision.hits else 0.0,
                "chunk_ids": [h.chunk.chunk_id for h in hits],
            },
        )
