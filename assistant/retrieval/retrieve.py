"""Retrieval: a question to passages the caller is allowed to see.

Thin by design. The repository does the filtering and the ranking, because that
is where the audience filter belongs — against rows inside the repository,
not in a
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
import re
from pathlib import Path

from .. import observability as obs
from .. import ollama
from ..model import Retrieved
from ..repository import IndexMismatch, RetrievalRequest
from ..indexing.index import CHUNKING_VERSION
from .. import paths

CONFIG = paths.CONFIG_DIR

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


# How much wider than `top_k` to ask the repository for, so that dropping a
# duplicate frees the slot for a different document instead of simply returning
# fewer passages. Two is enough for a corpus whose duplication is pairwise.
OVERFETCH = 2

_COLLAPSE = re.compile(r"\s+")


def _distinct(hits: list[Retrieved]) -> list[Retrieved]:
    """Drop a passage whose text a higher-ranked passage already carries.

    The site publishes two product pages for the same product -- `/products/duro`
    and `/products/duro-plaster`, `/products/ultra` and `/products/ultra-render`
    -- and each links the *same* datasheet under a different filename. The crawl
    is faithful, so the index holds `Duro TDS.pdf` and `Duro TDS_1.pdf` with
    identical content hashes under two product names, and the same for Ultra.
    Thirty-six of 552 passages live in a duplicated document, and they belong to
    the two products a demonstration is most likely to ask about.

    The cost is a retrieval slot. Asked "what temperature range can Duro be
    applied in", the five passages came back as Duro's Application section at
    rank 1 and *the same Application section* at rank 5, so a fifth of the
    evidence put in front of the model was a passage it already had. That is
    exactly what `per_document_cap` exists to prevent -- one document crowding
    out the others -- defeated by the document appearing twice under two names.
    So this applies the same principle where the cap cannot see it.

    Identity is the passage text, not the document, because two sections of one
    datasheet are legitimately different and must both survive. Comparison
    collapses whitespace and case and nothing else: a near-duplicate that
    differs by a word is a judgement call this should not be making, and silently
    dropping evidence on a similarity threshold is how a retrieval layer starts
    deciding what the answer may rest on.
    """
    seen: set[str] = set()
    kept: list[Retrieved] = []
    for hit in hits:
        key = _COLLAPSE.sub(" ", hit.chunk.content).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(hit)
    return kept


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
                "No active index. Build one with: python -m assistant.indexing.index"
            )
        if snapshot.embedding_model != self.embed_model:
            raise IndexMismatch(
                f"This index was built with {snapshot.embedding_model!r} but the "
                f"engine is configured for {self.embed_model!r}. Querying an index "
                "with a different embedding model returns confident nonsense. "
                "Rebuild with: python -m assistant.indexing.index"
            )
        if snapshot.embedding_dimensions != ollama.EMBED_DIMENSIONS:
            raise IndexMismatch(
                f"Index vectors are {snapshot.embedding_dimensions}d, the engine "
                f"expects {ollama.EMBED_DIMENSIONS}d."
            )
        if snapshot.chunking_version != CHUNKING_VERSION:
            raise IndexMismatch("Index chunking configuration changed; rebuild with python -m assistant.indexing.index")
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
        product: str = "",
    ) -> list[Retrieved]:
        """Passages for a question, and for the product it named if it named one.

        `product` is the name the caller detected in the question, not a guess
        made here — detection is the router's vocabulary work. Given one, the
        repository boosts that product's passages by a bounded margin so a
        semantically adjacent product cannot displace it. It is a boost and not
        a filter, because a question naming Ultra may still be answered partly
        from a shared system guide, and refusing those would be a silent loss.
        """
        expanded = self.expand(question)
        # A span rather than a plain timing, with the model call and the store
        # call as its two children: "retrieval was slow" is not an answer, and
        # the useful next question is always which of the two it was — the
        # network hop to embed, or the search. The names follow OpenTelemetry's
        # conventions (`gen_ai.request.model`, `db.system`) so an exporter later
        # is a shim rather than a rename of every call site.
        with obs.span("retrieval", audiences=list(audiences), top_k=top_k,
                      per_document_cap=per_document_cap, product=product,
                      expanded=expanded != question) as record:
            with obs.span("embed_question",
                          **{"gen_ai.request.model": self.embed_model,
                             "dimension": ollama.EMBED_DIMENSIONS}):
                try:
                    vector = ollama.embed_one(as_query(expanded),
                                              model=self.embed_model)
                except ollama.OllamaUnavailable as error:
                    # The embedding call is the first thing a question touches,
                    # so this is where a stopped model server is usually
                    # discovered.
                    obs.event("ollama_error", stage="embed",
                              model=self.embed_model,
                              error=type(error).__name__, detail=str(error))
                    raise
            with obs.span("search") as search:
                # Which adapter served this, named the way OTel names it. It is
                # read off the repository rather than configured, because the
                # engine is built against a Protocol and genuinely does not know
                # which store it has until it asks.
                search["db.system"] = ("postgresql"
                                       if "Postgres" in type(self.repo).__name__
                                       else "sqlite")
                # Asked wider than `top_k` so that dropping a duplicate frees
                # the slot for a different document rather than returning four
                # passages where five were wanted. The repository still applies
                # the audience filter and the per-document cap over everything
                # it returns, so nothing here reaches a row it should not.
                found = self.repo.retrieve_for(RetrievalRequest(
                    embedding=vector, audiences=audiences,
                    top_k=top_k * OVERFETCH,
                    per_document_cap=per_document_cap, product=product))
                hits = _distinct(found)[:top_k]
                search["returned"] = len(hits)
                search["duplicates_dropped"] = len(found) - len(_distinct(found))
            record["hits"] = len(hits)
            record["returned"] = len(hits)
            record["top_score"] = self.best_score(hits)
            record["above_threshold"] = self.above_threshold(hits)
            record["documents"] = len({h.chunk.canonical_url for h in hits})
        return hits

    def best_score(self, hits: list[Retrieved]) -> float:
        return hits[0].score if hits else 0.0

    def above_threshold(self, hits: list[Retrieved]) -> bool:
        return bool(hits) and hits[0].score >= self.threshold

    def find_property(
        self,
        product: str,
        terms: tuple[str, ...],
        audiences: tuple[str, ...] = ("public",),
        limit: int = 3,
    ) -> list[Retrieved]:
        """A second, targeted retrieval for a property the first pass may miss.

        "How many bags for twenty square metres" embeds as a question about
        quantity, and the coverage figure it needs sits in a short section that
        need not rank in the top five. A path that only re-sorts what it was
        given cannot recover from that; this can, because it asks by metadata
        rather than by similarity: the named product, and the words the property
        is printed under.

        No model call, no embedding, nothing to be unavailable — which also
        means the scores are 0.0 and must not be compared with the threshold.
        The guarantee these passages carry is lexical presence of the term,
        which is the thing the relevance gate wants to establish anyway.
        """
        # The terms are counted and not named, which is a change from the event
        # this span replaced and is not cosmetic. Two callers reach here: the
        # calculation edge, whose terms are the module constant COVERAGE_TERMS,
        # and `Assistant._missed_evidence`, whose term is a rare word lifted
        # straight out of the question. The second is question text, and the
        # event stream could hold it while a persisted span cannot — the review
        # §2.5 rule is that `answer_log.question` is the one deliberate
        # retention point and the trace does not duplicate it. A count answers
        # the operational question anyway: how often the second pass runs and
        # whether it finds anything.
        with obs.span("targeted_retrieval", product=product,
                      terms=len(terms), audiences=list(audiences),
                      limit=limit) as record:
            hits = self.repo.find_passages(product, terms, audiences=audiences,
                                           limit=limit)
            record["hits"] = len(hits)
        return hits
