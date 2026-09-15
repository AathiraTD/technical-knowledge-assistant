"""The assembled assistant: one question in, one composite reply out.

The order here is the order in the diagram, and the reason it is a separate
module from the router is that the router decides and this executes. Keeping
those apart is what makes the router testable without Ollama running.

A question with two topics in it produces two parts, each gated and routed on
its own, and the reply says which part was answered from what. A message that
asks a price and a coverage should not have one path chosen for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .answer import Answer, AnswerEngine
from .retrieve import Retriever
from .router import Decision, Path_, Router, split_by_topic


# The architecture caps input at about 500 words. Capping characters instead
# cut mid-word, and because the cap is applied before the message is split by
# topic, the fragment became its own part: it matched no policy pattern, so it
# reached retrieval as gibberish. Words are the unit the limit was written in
# and the unit that cannot produce a fragment.
MAX_WORDS = 500


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

    @property
    def refused(self) -> bool:
        return all(a.refused for _q, a in self.parts) if self.parts else True

    @property
    def paths(self) -> list[str]:
        return [a.path for _q, a in self.parts]


class Assistant:
    """Retrieval, routing and answering behind one call."""

    def __init__(self, repo, threshold: float | None = None) -> None:
        self.repo = repo
        self.retriever = Retriever(repo, **({"threshold": threshold}
                                            if threshold is not None else {}))
        self.router = Router()
        self.engine = AnswerEngine(repo, self.router)
        # The engine needs the slot vocabulary for the relevance check, and the
        # router owns it. Handing over the router rather than a copy keeps one
        # definition of what a term means.
        self.engine.retriever = self.router

    def ask(self, question: str, audiences: tuple[str, ...] = ("public",)) -> Reply:
        question = cap(question)
        reply = Reply(question=question, audiences=audiences)

        with self.repo.read_snapshot() as snapshot:
            self.retriever._verify()
            # Names/contact are release metadata too; refresh them together
            # with the passages rather than retaining the startup snapshot.
            self.engine.names = {key: snapshot.notes.get(key, default) for key, default in
                                 (("products", []), ("colours", []), ("merchants", []), ("contact", {}))}
            for part in split_by_topic(question):
                answer = self._answer_part(part, audiences)
                answer.diagnostics["snapshot_id"] = snapshot.snapshot_id
                answer.diagnostics["embedding_model"] = snapshot.embedding_model
                answer.diagnostics["chunking_version"] = snapshot.chunking_version
                reply.parts.append((part, answer))
        return reply

    def _answer_part(self, part: str, audiences: tuple[str, ...]) -> Answer:
        matched = self.router.gate.match(part)
        if matched:
            topic, spec = matched
            if spec.get("from_manifest"):
                return self.engine.documents_for(part, audiences)
            return self.engine.route(topic, spec)

        hits = self.retriever.search(part, audiences=audiences)
        decision = self.router.route(
            part, hits, self.retriever.above_threshold(hits), audiences)

        return {
            Path_.EXTRACT: lambda: self.engine.extract(decision),
            Path_.COMPOSE: lambda: self.engine.compose(decision, part),
            Path_.DEFER: lambda: self.engine.defer(decision),
            Path_.DIAGNOSIS: lambda: self.engine.diagnosis(decision),
            Path_.ASK_BACK: lambda: self.engine.ask_back(decision),
            Path_.REFUSE: lambda: self.engine.refuse(decision, decision.reason),
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
