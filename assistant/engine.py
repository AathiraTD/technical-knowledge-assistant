"""The assembled assistant: one question in, one composite reply out.

The order here is the order in the diagram, and the reason it is a separate
module from the router is that the router decides and this executes. Keeping
those apart is what makes the router testable without Ollama running.

A question with two topics in it produces two parts, each gated and routed on
its own, and the reply says which part was answered from what. A message that
asks a price and a coverage should not have one path chosen for both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from . import observability as obs
from . import ollama
from . import vision
from .answer import Answer, AnswerEngine, Provenance
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

# The manufacturer, as the website writes it in front of its own product names.
# Stripped before matching, because chunks are tagged with the catalogue name.
_BRAND = "lime green"

# How many passages a word may appear in and still count as naming something
# specific. Measured rather than guessed: a word returning more than this is
# ordinary vocabulary, not the thing the question was about.
DISTINCTIVE_CAP = 6

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
                 source: str = "unknown") -> None:
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
        self.engine = AnswerEngine(repo, self.router)
        # Exact-key, audience-scoped, snapshot-scoped. See assistant/cache.py
        # for why it is not the template-keyed cache decision 14 designs.
        self.cache = AnswerCache() if cache else None
        # The engine needs the slot vocabulary for the relevance check, and the
        # router owns it. Handing over the router rather than a copy keeps one
        # definition of what a term means.
        self.engine.retriever = self.router

    def ask(self, question: str, audiences: tuple[str, ...] = ("public",),
            correlation_id: str = "", carried: dict | None = None,
            images=None) -> Reply:
        """One message in, one composite reply out.

        `carried` is what the caller already knows that this question does not
        say — slots held from an earlier turn in the same conversation. The
        engine does not decide what is worth carrying and does not store it; a
        session owns that, and hands it in. Keeping the memory outside the
        engine is what lets the CLI stay single-turn and stateless while the web
        page is neither.

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
        """
        question = cap(question)
        reply = Reply(question=question, audiences=audiences)

        observed: dict[str, str] = {}
        if images:
            resolution = vision.slots_from_images(images)
            observed = dict(resolution.slots)
            # Counts and slot names only. The privacy posture of this system is
            # that no question text and no passage text reaches the log; a
            # photograph is the caller's own property and is named nowhere at
            # all, which is why the uploaded filename is not carried this far.
            obs.event("perception", images=len(images),
                      slots=sorted(observed), discarded=len(resolution.discarded),
                      cannot_determine=len(resolution.cannot_determine_from_image))
        carried = {**(carried or {}), **observed}
        # Sparse: only the slots whose origin is not the default. Everything
        # else reads as ``CARRIED``, which is what every caller without a
        # photograph has always produced and must keep producing.
        origins = {slot: Provenance.OBSERVED for slot in observed}

        # One id for the whole message, including every part it splits into, so
        # the lines for a two-topic question can be read as one event. A caller
        # that already has a request id — a web request, a channel adapter —
        # passes it in rather than starting a second trace for the same work.
        with obs.correlation(correlation_id) as cid:
            reply.correlation_id = cid
            with obs.timed("answer", audiences=list(audiences),
                           question_words=len(question.split()),
                           question=obs.fingerprint(question)) as summary:
                self._ask(question, audiences, reply, cid, carried, origins)
                summary["parts"] = len(reply.parts)
                summary["paths"] = reply.paths
                summary["refused"] = reply.refused
        return reply

    def _ask(self, question: str, audiences: tuple[str, ...], reply: Reply,
             cid: str, carried: dict | None = None,
             origins: dict | None = None) -> None:
        with self.repo.read_snapshot() as snapshot:
            self.retriever._verify()
            # Names/contact are release metadata too; refresh them together
            # with the passages rather than retaining the startup snapshot.
            self.engine.names = {key: snapshot.notes.get(key, default) for key, default in
                                 (("products", []), ("colours", []), ("merchants", []), ("contact", {}))}
            for part in split_by_topic(question):
                key = self._cache_key(part, audiences, snapshot, carried,
                                      origins)
                # `is not None`, not truthiness. AnswerCache defines __len__,
                # so an empty cache is falsy and `if self.cache` was False on
                # every call — the cache could never fill, because it was empty.
                answer = self.cache.get(key) if self.cache is not None else None
                obs.event("cache", hit=answer is not None,
                          question=obs.fingerprint(part),
                          snapshot_id=snapshot.snapshot_id)
                if answer is None:
                    answer = self._answer_part(part, audiences, carried,
                                               origins)
                    answer.diagnostics["snapshot_id"] = snapshot.snapshot_id
                    answer.diagnostics["embedding_model"] = snapshot.embedding_model
                    answer.diagnostics["chunking_version"] = snapshot.chunking_version
                    if self.cache is not None:
                        self.cache.put(key, answer)
                else:
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

    def _named_product(self, part: str) -> str:
        """The product this question names, if it names one the corpus knows.

        Matched against the harvested product list rather than guessed, so it
        cannot invent a product — the same list check 5 uses to refuse invented
        names. The longest match wins, because "Lime Green Solo Onecoat" and
        "Solo" are both in the list and the specific one is the one meant.

        Retrieval treats this as a bounded boost rather than a filter, so a
        wrong detection reorders and never refuses: the worst case is the same
        answer in a different order, not a silent loss.
        """
        lowered = part.lower()
        named = [p for p in self.engine.names.get("products", [])
                 if p and p.lower() in lowered]
        if not named:
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

    def _cache_key(self, part: str, audiences: tuple[str, ...], snapshot,
                   carried: dict | None = None, origins: dict | None = None):
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
                               carried, origins)

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
                     origins: dict | None = None) -> Answer:
        matched = self.router.gate.match(part)
        if matched:
            topic, spec = matched
            # The policy gate is a routing decision like any other, and it is
            # the one that keeps the model away from prices and stock. It is
            # worth being able to count how often it fires.
            obs.event("route", path=Path_.ROUTE.value, topic=topic, step="gate")
            if spec.get("from_manifest"):
                return self.engine.documents_for(part, audiences)
            return self.engine.route(topic, spec)

        named = self._named_product(part)
        hits = self.retriever.search(part, audiences=audiences, product=named)
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
        if named and "calculation" in self.router.slots.detect(part):
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

        decision = self.router.route(
            part, hits, self.retriever.above_threshold(hits), audiences, carried)
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
        obs.event("route", path=decision.path.value, step=decision.step,
                  reason=decision.reason,
                  top_score=hits[0].score if hits else 0.0,
                  slots=sorted(decision.slots))

        return {
            # Every path gets the question, not just Compose. Without it the
            # other five cannot tell a slot the caller stated in this sentence
            # from one carried out of an earlier turn, so they fell back to
            # reporting both as "as you said" — which is false of a carried
            # value and reads as the system putting words in someone's mouth.
            # It showed up on a quantity question that reported "external
            # (location), as you said" when external had come from a question
            # two turns earlier about render.
            Path_.EXTRACT: lambda: self.engine.extract(decision, part),
            Path_.COMPOSE: lambda: self.engine.compose(decision, part),
            Path_.DEFER: lambda: self.engine.defer(decision, part),
            Path_.DIAGNOSIS: lambda: self.engine.diagnosis(decision, part),
            Path_.ASK_BACK: lambda: self.engine.ask_back(decision, part),
            Path_.REFUSE: lambda: self.engine.refuse(decision, decision.reason, part),
        }[decision.path]()


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
