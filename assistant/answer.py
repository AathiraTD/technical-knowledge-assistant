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

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "with", "as", "at", "by", "it", "this", "that", "from", "can", "will",
    "should", "must", "may", "not", "you", "your", "we", "our", "per", "if",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in _STOP and len(w) > 2}


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


def _numbers(text: str) -> list[str]:
    """Figures with their units attached, which is the unit of comparison.

    Every match necessarily contains a digit, because the pattern opens with
    one; an earlier guard re-checked for a digit here and could never fire.
    """
    return [m.group(0).strip() for m in _NUMBER.finditer(text)]


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
        sentence_words = _words(_CITE.sub("", sentence))
        if sentence_words and len(sentence_words & passage_words) / len(sentence_words) < 0.4:
            failures.append(
                f"check 1: a sentence does not overlap the passage it cites — "
                f"{sentence[:70]!r}"
            )

    # 2 — every number appears verbatim in a passage the sentence cites.
    for sentence in sentences:
        marks = [int(m) for m in _CITE.findall(sentence)]
        cited_text = " ".join(
            cited_index[m].chunk.content for m in marks if m in cited_index
        )
        cited_norm = _normalise_number(cited_text)
        for token in _numbers(_CITE.sub("", sentence)):
            if _normalise_number(token) not in cited_norm:
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
        named = {p for p in products_in_hits
                 if p and p not in cited_products and _names_product(sentence, p)}
        if named:
            failures.append(
                f"check 3: a figure is stated against {sorted(named)[0]!r} but cited "
                "to a different product's passage"
            )

    # 4 — qualifiers travel with their figure, inside the printed passage.
    for sentence in sentences:
        if re.search(r"\b(?:minimum|maximum|at least|up to|no more than|below|above)\b",
                     sentence, re.I):
            marks = [int(m) for m in _CITE.findall(sentence)]
            cited_text = " ".join(
                cited_index[m].chunk.content for m in marks if m in cited_index
            ).lower()
            qualifier = re.search(
                r"\b(minimum|maximum|at least|up to|no more than|below|above)\b",
                sentence, re.I)
            if qualifier and qualifier.group(1).lower() not in cited_text:
                failures.append(
                    f"check 4: the qualifier {qualifier.group(1)!r} is not in the "
                    "cited passage"
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
        if _looks_like_a_name(candidate):
            failures.append(f"check 5: {candidate!r} is not a name the site publishes")

    # 6 — the asked-for term appears in a cited passage.
    if asked_terms:
        blob = " ".join(f"{h.chunk.section} {h.chunk.content}" for h in hits).lower()
        if not any(t.lower() in blob for t in asked_terms):
            failures.append("check 6: the property asked about is not in any passage")

    return list(dict.fromkeys(failures))


_CAP_RUN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-z]+)")


def _capitalised_runs(text: str) -> list[str]:
    """Candidate product, colour and merchant names in generated prose."""
    starts = set(_SENTENCE_START.findall(text))
    return [m for m in _CAP_RUN.findall(_CITE.sub("", text)) if m not in starts]


_COMMON = {
    "lime", "lime green", "green", "the", "it", "however", "this", "these",
    "when", "where", "what", "if", "for", "use", "using", "apply", "mix",
    "water", "wall", "walls", "coat", "coats", "sources", "answer", "note",
}


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
        top = decision.hits[0]
        body = [f"From {top.document.citation_name}"
                f"{', ' + top.chunk.section if top.chunk.section else ''} [1]:",
                "", top.chunk.content]
        if decision.sum_refused:
            body += ["", "I have printed the published coverage and pack size rather "
                          "than multiplying them out. The figure that matters on site "
                          "depends on the background and the thickness, so the "
                          "arithmetic is worth doing against your own measurements."]
        return self._finish(decision, "\n".join(body), decision.hits[:1])

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
        text, seconds = ollama.generate(prompt, model=self.model, system=SYSTEM)

        asked = decision.slots.get("property_asked", "")
        terms = (self.retriever.slots.terms_for("property_asked", asked)
                 if hasattr(self.retriever, "slots") else [])
        failures = run_checks(text, hits, self.names, terms)

        if failures:
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
