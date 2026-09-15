"""Retrieval: a question to passages the caller is allowed to see.

Thin by design. The repository does the filtering and the ranking, because that
is where the audience filter belongs — in the query, against rows, not in a
prompt. What is left here is the part that is genuinely about the question:
expanding it with the handful of synonyms that close the vocabulary gap, and
refusing to run at all when the index was built by a different model.

That refusal is the important line in this file. An index built with one
embedding model and queried with another does not fail — it returns plausible,
confidently wrong passages, which is the exact failure mode the whole design
exists to avoid. So it is checked once, at load, and raised rather than warned.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ollama
from .model import Retrieved
from .repository import IndexMismatch

CONFIG = Path(__file__).resolve().parents[1] / "config"

# Below this, nothing retrieved is close enough to answer from. The value is
# provisional until the threshold sweep in the evaluation harness sets it;
# it is recorded with every answer so a transcript says which value produced it.
DEFAULT_THRESHOLD = 0.45


def _vocabularies() -> dict:
    return json.loads((CONFIG / "vocabularies.json").read_text(encoding="utf-8"))


# Qwen3-Embedding is trained asymmetrically: documents are embedded as plain
# text, questions with an instruction prefix. Without it the query and the
# passage sit in slightly different regions of the space, and measurably worse
# passages come back — the Solo datasheet fell below a page about aerogel
# adhesive on a question naming Solo. The prefix is applied to the question
# only, never to a chunk, and never to anything printed.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a building product, retrieve the "
    "passages from the manufacturer's documents that answer it\nQuery: "
)


def as_query(question: str) -> str:
    return QUERY_INSTRUCTION + question


class Retriever:
    """Embeds questions and returns passages, or explains why it will not."""

    def __init__(self, repo, threshold: float = DEFAULT_THRESHOLD,
                 embed_model: str = ollama.EMBED_MODEL) -> None:
        self.repo = repo
        self.threshold = threshold
        self.embed_model = embed_model
        self.synonyms = _vocabularies()["retrieval_synonyms"]
        self._verify()

    def _verify(self) -> None:
        """Refuse a model or chunking mismatch rather than degrade quietly."""
        snapshot = self.repo.snapshot()
        if snapshot is None:
            raise IndexMismatch(
                "No active index. Build one with: python -m assistant.index"
            )
        if snapshot.embedding_model != self.embed_model:
            raise IndexMismatch(
                f"This index was built with {snapshot.embedding_model!r} but the "
                f"engine is configured for {self.embed_model!r}. Querying an index "
                "with a different embedding model returns confident nonsense. "
                "Rebuild with: python -m assistant.index"
            )
        if snapshot.embedding_dimensions != ollama.EMBED_DIMENSIONS:
            raise IndexMismatch(
                f"Index vectors are {snapshot.embedding_dimensions}d, the engine "
                f"expects {ollama.EMBED_DIMENSIONS}d."
            )
        self.snapshot = snapshot

    def expand(self, question: str) -> str:
        """Add the few synonyms worth stating outright, for the embedding only.

        The expanded string is never shown and never cited. It exists so that a
        customer who says 'bag' reaches a datasheet that says 'sack'.
        """
        lowered = question.lower()
        extra = [
            term
            for phrase, terms in self.synonyms.items()
            if not phrase.startswith("_") and phrase in lowered
            for term in terms
            if term not in lowered
        ]
        return f"{question} {' '.join(dict.fromkeys(extra))}".strip() if extra else question

    def search(
        self,
        question: str,
        audiences: tuple[str, ...] = ("public",),
        top_k: int = 5,
        per_document_cap: int = 3,
    ) -> list[Retrieved]:
        vector = ollama.embed_one(as_query(self.expand(question)),
                                  model=self.embed_model)
        return self.repo.retrieve(vector, audiences=audiences, top_k=top_k,
                                  per_document_cap=per_document_cap)

    def best_score(self, hits: list[Retrieved]) -> float:
        return hits[0].score if hits else 0.0

    def above_threshold(self, hits: list[Retrieved]) -> bool:
        return bool(hits) and hits[0].score >= self.threshold
