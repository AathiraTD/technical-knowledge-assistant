"""The assembled assistant: splitting, dispatch and rendering.

The engine decides nothing about safety. Its job is to take the parts of a
message one at a time, send each to the path the router chose, and print the
result with its sources. So the risks it carries are wiring risks, and they are
the ones tested here: a price question that reaches retrieval has already cost
a pointless embedding call and put the model in front of a question the policy
gate exists to keep it away from; a two topic message answered as one sends
half of it down the wrong path; a reply reported as refused when one part
answered throws away a good answer.

It runs on a real SQLite store built in a temporary directory, because the
repository boundary is worth exercising rather than mocking, but never on
Ollama: the embedding call and the generation call are both replaced, and the
tests that must not reach a model replace them with something that fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.index import CHUNKING_VERSION
from assistant import ollama                                      # noqa: E402
from assistant.answer import Answer                               # noqa: E402
from assistant.engine import MAX_WORDS, cap, Assistant, Reply, render, split_by_topic  # noqa: E402
from assistant.model import (                                     # noqa: E402
    Caveat, Chunk, Document, DocumentVersion, Retrieved, Snapshot,
)
from assistant.router import Decision, Path_                      # noqa: E402
from assistant.store import SQLiteKnowledgeRepository              # noqa: E402

DIMS = 1024
SOLO = "https://example.invalid/solo"
DURO = "https://example.invalid/duro"

SOLO_TEXT = ("Mix Solo with 5-6 litres of clean water per 25 kg sack and stir "
             "for three minutes.")
DURO_TEXT = "Duro covers approximately 2.5 m2 per 25 kg bag at 11 mm thickness."

CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}


def unit(axis: int) -> list[float]:
    """A unit vector along one axis, so cosine similarity is predictable."""
    v = [0.0] * DIMS
    v[axis] = 1.0
    return v


def document(url: str, product: str) -> Document:
    return Document(canonical_url=url, title=f"{product} datasheet",
                    document_type="datasheet", authority=1, product=product,
                    link_text=f"{product} Datasheet")


def version(url: str) -> DocumentVersion:
    return DocumentVersion(canonical_url=url, version=1, content_hash="h1",
                           source_path=f"cache/{product_of(url)}.pdf",
                           first_seen_at="2026-01-01",
                           fetched_at="2026-01-01T00:00:00Z",
                           checked_at="2026-01-01T00:00:00Z")


def product_of(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def chunk(url: str, index: int, section: str, content: str, product: str,
          axis: int) -> Chunk:
    return Chunk(canonical_url=url, version=1, chunk_index=index, section=section,
                 content=content, product=product, document_type="datasheet",
                 authority=1, source_date="2024-07-01", embedding=unit(axis))


def build_repo(tmp_path, two_documents: bool = False) -> SQLiteKnowledgeRepository:
    """A store holding one or two datasheets, with an active snapshot."""
    documents = [document(SOLO, "Solo")]
    versions = [version(SOLO)]
    chunks = [chunk(SOLO, 0, "Mixing", SOLO_TEXT, "Solo", 0),
              chunk(SOLO, 1, "Application", "Apply Solo in one coat.", "Solo", 2)]
    caveats = [Caveat(SOLO, "temperature", "Do not apply below 5 degrees C.",
                      "Mixing")]

    if two_documents:
        documents.append(document(DURO, "Duro"))
        versions.append(version(DURO))
        chunks.append(chunk(DURO, 0, "Coverage", DURO_TEXT, "Duro", 1))

    snapshot = Snapshot(
        snapshot_id="snap-test", created_at="2026-01-01T00:00:00Z",
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION, document_count=len(documents),
        chunk_count=len(chunks),
        notes={"products": ["Solo", "Duro"], "colours": ["York"],
               "merchants": ["The Lime Centre"], "contact": CONTACT})

    repo = SQLiteKnowledgeRepository(tmp_path / "index" / "knowledge.db")
    repo.publish(documents, versions, chunks, snapshot, caveats)
    return repo


@pytest.fixture
def no_ollama(monkeypatch):
    """Answer the embedding call locally and forbid the generation call."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))

    def refuse_to_generate(*_a, **_k):
        raise AssertionError("the model was called on a path that must not use it")

    monkeypatch.setattr(ollama, "generate", refuse_to_generate)


@pytest.fixture
def assistant(tmp_path, no_ollama):
    repo = build_repo(tmp_path)
    try:
        yield Assistant(repo)
    finally:
        repo.close()


@pytest.fixture
def two_document_assistant(tmp_path, no_ollama):
    repo = build_repo(tmp_path, two_documents=True)
    try:
        yield Assistant(repo)
    finally:
        repo.close()


def quoting(prompt: str, **_kwargs) -> tuple[str, float]:
    """A model that does the one thing this system wants: quote and cite.

    It reads the passages out of the prompt it was handed and returns the first
    sentence of each under the marker it arrived with, so the six checks pass
    and Compose actually prints. The leading `.strip()` matters: the passage
    block opens with a newline, and without it the first passage is silently
    dropped and the answer looks multi-source while citing one document.
    """
    sentences = []
    for block in prompt.split("Passages:", 1)[-1].split("\n\n"):
        head, _, body = block.strip().partition("\n")
        head, body = head.strip(), body.strip()
        if head.startswith("[") and "]" in head and body:
            marker = head[:head.index("]") + 1]
            sentences.append(f"{body.split('. ')[0].rstrip('.')} {marker}.")
    return " ".join(sentences) or "Nothing was supplied.", 1.25


def forbid_retrieval(monkeypatch, assistant):
    """Make any retrieval call fail the test rather than quietly succeed."""
    def boom(*_a, **_k):
        raise AssertionError("retrieval ran on a question the policy gate owns")

    monkeypatch.setattr(assistant.retriever, "search", boom)


def hit(url: str, section: str, content: str, product: str,
        score: float = 0.9) -> Retrieved:
    return Retrieved(chunk=chunk(url, 0, section, content, product, 0), score=score,
                     document=document(url, product))


# ----------------------------------------------------------------- splitting


def test_two_topics_in_one_message_become_two_parts():
    """A price and a coverage in one message need two different paths."""
    parts = split_by_topic("How much does Solo cost? And what coverage does it give?")
    assert len(parts) == 2
    assert parts[0].startswith("How much does Solo cost")


def test_a_conjunction_that_introduces_a_second_question_splits_too():
    """Without punctuation the second question still has to reach its own path."""
    parts = split_by_topic("What does Solo cost and what coverage does it give")
    assert len(parts) == 2


def test_one_question_stays_one_part():
    question = "How much water does Solo need per bag"
    assert split_by_topic(question) == [question]


def test_a_trailing_fragment_is_not_treated_as_a_question():
    """'Thanks.' retrieves noise and would print a refusal nobody asked for."""
    assert split_by_topic("Is Solo suitable for cob? Thanks.") == [
        "Is Solo suitable for cob?"]


# ------------------------------------------------------------- the two gates


def test_a_price_question_never_reaches_retrieval(assistant, monkeypatch):
    """Prices are not published, so embedding the question is pure waste."""
    forbid_retrieval(monkeypatch, assistant)

    reply = assistant.ask("How much does Solo cost?")

    assert reply.paths == [Path_.ROUTE.value]
    assert reply.parts[0][1].diagnostics["topic"] == "price"
    assert CONTACT["phone"] in reply.parts[0][1].text


def test_a_document_request_is_answered_from_the_manifest(assistant, monkeypatch):
    """Which datasheet is current is a fact in the index, not something to retrieve."""
    forbid_retrieval(monkeypatch, assistant)

    reply = assistant.ask("Where can I find the Solo datasheet?")
    answer = reply.parts[0][1]

    assert answer.path == Path_.ROUTE.value
    assert answer.diagnostics["step"] == "manifest"
    assert "Solo Datasheet" in answer.text
    assert SOLO in answer.text


def test_the_gated_part_of_a_mixed_message_does_not_drag_the_rest_with_it(
        two_document_assistant, monkeypatch):
    """The price part routes and the thickness part still gets its passage."""
    monkeypatch.setattr(ollama, "generate",
                        lambda *_a, **_k: ("Apply Solo in one coat [2].", 0.8))
    reply = two_document_assistant.ask(
        "How much does Solo cost? What thickness does Solo go on at?")

    assert len(reply.parts) == 2
    assert reply.paths[0] == Path_.ROUTE.value
    assert reply.paths[1] != Path_.ROUTE.value


def test_an_oversized_message_is_capped_before_anything_else_happens(
        assistant, monkeypatch):
    """An unbounded question is a denial of service path into the embedding call."""
    monkeypatch.setattr(assistant.retriever, "search", lambda *_a, **_k: [])

    reply = assistant.ask("What thickness does Solo go on at? " * 500)

    # The literal, not the constant. `== MAX_WORDS` is the expression under
    # test compared against itself: setting MAX_WORDS to 20 left this green,
    # which mutation confirmed. `docs/architecture.md` says "capped at about
    # 500 words", so 500 is the documented contract and the number worth
    # pinning — a silent tightening would truncate real questions, and a
    # silent loosening would reopen the denial-of-service path.
    assert len(reply.question.split()) == 500 == MAX_WORDS
    assert reply.refused is True


def test_the_cap_never_leaves_a_half_word_at_the_end():
    """A character cap cut mid-word, and the fragment became its own part that
    matched no policy pattern and reached retrieval as gibberish."""
    long_question = "supercalifragilistic " * 400
    capped = cap(long_question)
    assert capped.split()[-1] == "supercalifragilistic"
    assert not capped.endswith(" ")


def test_a_short_question_is_returned_unchanged():
    assert cap("  How much water does Solo need?  ") == "How much water does Solo need?"


# ---------------------------------------------------------------- dispatch


def test_a_single_document_factual_question_prints_its_passage_without_a_model(
        assistant):
    """Extract runs by code: a paraphrase of a mixing ratio is a site error."""
    reply = assistant.ask("How much water does Solo need per bag")
    answer = reply.parts[0][1]

    assert answer.path == Path_.EXTRACT.value
    assert SOLO_TEXT in answer.text
    assert answer.sources[0]["name"] == "Solo Datasheet"
    assert answer.refused is False


def test_several_documents_reach_the_compose_path(two_document_assistant,
                                                  monkeypatch):
    """The model runs on one path only, and this is the one.

    The stub answers both halves of the question and cites both documents. It
    used to answer only the water half while citing one passage, and passed
    because check 6 then scanned every retrieved passage rather than the cited
    ones — so an answer that never addressed coverage satisfied a gate about
    coverage. With check 6 tightened to cited evidence, an incomplete answer is
    correctly refused, and this test has to supply a complete one.
    """
    # Quote whatever passages arrive, under the markers they arrive with,
    # rather than hard-coding [1] and [2]. Retrieval order is not a fact this
    # test is about, and pinning it made the test fail the moment a named
    # product started boosting its own passages — a reordering that is correct.
    monkeypatch.setattr(ollama, "generate", quoting)

    reply = two_document_assistant.ask("What water and coverage does Solo have")
    answer = reply.parts[0][1]

    assert answer.path == Path_.COMPOSE.value
    assert answer.diagnostics["generation_seconds"] == 1.25


@pytest.mark.parametrize("path", [Path_.EXTRACT, Path_.COMPOSE, Path_.DEFER,
                                  Path_.DIAGNOSIS, Path_.ASK_BACK, Path_.REFUSE])
def test_every_path_the_router_can_choose_has_somewhere_to_go(
        assistant, monkeypatch, path):
    """An unhandled path is a KeyError in front of a customer."""
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: ("Mix Solo with 5-6 litres of clean water per 25 kg "
                           "sack [1].", 0.9))
    hits = [hit(SOLO, "Mixing", SOLO_TEXT, "Solo")]
    monkeypatch.setattr(assistant.retriever, "search", lambda *_a, **_k: hits)
    monkeypatch.setattr(
        assistant.router, "route",
        lambda *_a, **_k: Decision(path, "because the test said so", "9", hits=hits))

    reply = assistant.ask("What thickness does Solo go on at")

    assert reply.paths == [path.value]
    assert reply.parts[0][1].text


def test_a_staff_question_is_retrieved_for_the_staff_audience(assistant,
                                                              monkeypatch):
    """The audience set has to reach the query, since that is where it is enforced."""
    seen: dict = {}

    def search(part, audiences=("public",), **_k):
        seen["audiences"] = audiences
        return [hit(SOLO, "Mixing", SOLO_TEXT, "Solo")]

    monkeypatch.setattr(assistant.retriever, "search", search)
    assistant.ask("What thickness does Solo go on at", audiences=("staff",))

    assert seen["audiences"] == ("staff",)


def test_an_explicit_threshold_overrides_the_default(tmp_path, no_ollama):
    """The sweep sets this value, so it has to be settable from outside."""
    repo = build_repo(tmp_path)
    try:
        assert Assistant(repo, threshold=0.8).retriever.threshold == 0.8
        assert Assistant(repo).retriever.threshold != 0.8
    finally:
        repo.close()


# ------------------------------------------------------------ the composite


def answered(text: str = "Mix with 5-6 litres [1].") -> Answer:
    return Answer(text=text, path=Path_.EXTRACT.value)


def refused(text: str = "I could not find this.") -> Answer:
    return Answer(text=text, path=Path_.REFUSE.value, refused=True)


def test_a_reply_is_refused_only_when_every_part_refused():
    """Reporting a half answered message as a refusal throws away a real answer."""
    mixed = Reply(question="q", parts=[("a", answered()), ("b", refused())])
    both = Reply(question="q", parts=[("a", refused()), ("b", refused())])

    assert mixed.refused is False
    assert both.refused is True


def test_a_reply_with_no_parts_is_refused():
    """Nothing answered is not the same as nothing asked, and prints as a refusal."""
    assert Reply(question="q").refused is True


def test_the_paths_taken_are_reported_part_by_part():
    """The diagnostics claim is that a transcript can say how each part was answered."""
    reply = Reply(question="q", parts=[("a", answered()), ("b", refused())])
    assert reply.paths == [Path_.EXTRACT.value, Path_.REFUSE.value]


# -------------------------------------------------------------- the renderer


def rendered_answer(**overrides) -> Answer:
    fields = {
        "text": "Mix Solo with 5-6 litres of clean water per 25 kg sack [1].",
        "path": Path_.EXTRACT.value,
        "sources": [{"marker": 1, "name": "Solo Datasheet", "section": "Mixing",
                     "url": SOLO, "type": "datasheet", "date": "2024-07-01",
                     "score": 0.91, "first": True}],
        "caveats": ["Do not apply below 5 degrees C."],
        "assumptions": ["substrate: brick"],
        "diagnostics": {"step": "7", "reason": "one document answers it",
                        "top_score": 0.91, "slots": {"property_asked": "water"}},
    }
    fields.update(overrides)
    return Answer(**fields)


def test_render_prints_the_answer_its_sources_its_caveats_and_its_assumptions():
    """A cited answer whose source is not printed is not traceable by the reader."""
    text = render(Reply(question="q", parts=[("q", rendered_answer())]))

    assert "5-6 litres" in text
    assert "Assumed: substrate: brick" in text
    assert "Do not apply below 5 degrees C." in text
    assert "[1] Solo Datasheet" in text
    assert "Mixing" in text
    assert "2024-07-01" in text
    assert SOLO in text


def test_a_source_with_no_date_or_section_still_prints():
    """Product pages carry neither, and must not render a stray separator."""
    answer = rendered_answer(
        sources=[{"marker": 1, "name": "Solo product page", "section": "",
                  "url": SOLO, "type": "product_page", "date": "", "score": 0.7,
                  "first": True}],
        caveats=[], assumptions=[])
    text = render(Reply(question="q", parts=[("q", answer)]))

    assert "[1] Solo product page" in text
    assert "Assumed:" not in text
    assert "Also published" not in text


def test_diagnostics_are_hidden_unless_they_are_asked_for():
    """A customer reading chunk ids and scores is reading the wrong document."""
    reply = Reply(question="q", parts=[("q", rendered_answer())])

    assert "[path:" not in render(reply)
    assert "[path:" in render(reply, show_diagnostics=True)


def test_diagnostics_name_the_route_the_slots_and_the_checks_that_failed():
    """A refusal nobody can explain is indistinguishable from a broken system."""
    answer = rendered_answer(
        path=Path_.COMPOSE.value,
        failed_checks=["check 2: '7 litres' is not in the passage it is cited to"],
        diagnostics={"step": "8", "reason": "several passages bear on it",
                     "top_score": 0.88, "slots": {"property_asked": "water"},
                     "generation_seconds": 4.2, "refusal_reason": "checks failed"})

    text = render(Reply(question="q", parts=[("q", answer)]), show_diagnostics=True)

    assert "path: compose" in text
    assert "step 8" in text
    assert "slots:" in text
    assert "checks that failed" in text
    assert "check 2" in text
    assert "generation: 4.2s" in text


def test_a_refusal_with_no_step_or_score_still_renders_its_diagnostics():
    """A policy routed part carries no retrieval numbers and must not blow up."""
    answer = Answer(text="Ask your stockist.", path=Path_.ROUTE.value,
                    diagnostics={"topic": "price", "step": "policy gate"})
    text = render(Reply(question="q", parts=[("q", answer)]), show_diagnostics=True)

    assert "path: route" in text
    assert "top score 0" in text
    assert "why: -" in text


def test_render_labels_the_parts_when_there_is_more_than_one():
    """Two answers run together read as one, and the second loses its question."""
    reply = Reply(question="q", parts=[("what does it cost", refused()),
                                       ("how much water", answered())])
    text = render(reply)

    assert "Part 1: what does it cost" in text
    assert "Part 2: how much water" in text


def test_a_single_part_is_not_labelled():
    """One question does not need a table of contents."""
    text = render(Reply(question="q", parts=[("how much water", answered())]))
    assert "Part 1" not in text


# --------------------------------------------------- the cache and the log


def test_an_empty_cache_is_still_a_cache(tmp_path, no_ollama):
    """`if self.cache` was False on every call because AnswerCache has __len__.

    The cache could never fill, because it was empty. Truthiness on a container
    means emptiness, and emptiness is not absence.
    """
    repo = build_repo(tmp_path)
    try:
        assistant = Assistant(repo)
        assert len(assistant.cache) == 0
        assert not assistant.cache, "an empty cache is falsy, which is the trap"

        assistant.ask("How much does Solo cost")
        assert assistant.cache.misses == 1, "the cache was skipped while empty"
    finally:
        repo.close()


def test_a_repeated_question_is_served_from_the_cache(tmp_path, no_ollama):
    """A composed answer costs about forty seconds; a repeat should cost nothing."""
    repo = build_repo(tmp_path)
    try:
        assistant = Assistant(repo)
        assistant.ask("How much does Solo cost")
        reply = assistant.ask("How much does Solo cost")

        assert assistant.cache.hits == 1
        assert reply.parts[0][1].diagnostics["cached"] is True
    finally:
        repo.close()


def test_a_staff_answer_is_not_replayed_to_a_public_caller(tmp_path, no_ollama):
    """Decision 14's named failure, at the level that would actually leak it."""
    repo = build_repo(tmp_path)
    try:
        assistant = Assistant(repo)
        assistant.ask("How much does Solo cost", audiences=("staff",))
        before = assistant.cache.hits
        assistant.ask("How much does Solo cost", audiences=("public",))

        assert assistant.cache.hits == before, "a staff entry was served to public"
    finally:
        repo.close()


def test_the_cache_can_be_turned_off(tmp_path, no_ollama):
    repo = build_repo(tmp_path)
    try:
        assistant = Assistant(repo, cache=False)
        assistant.ask("How much does Solo cost")
        assistant.ask("How much does Solo cost")
        assert assistant.cache is None
    finally:
        repo.close()


def test_every_answered_part_is_recorded_against_its_snapshot(tmp_path, no_ollama):
    """The audit chain: an answer names the snapshot and the passages it used."""
    repo = build_repo(tmp_path)
    try:
        Assistant(repo).ask("How much does Solo cost")
        logged = repo.answer_log(limit=5)

        assert logged, "nothing was recorded"
        assert logged[0].path_taken == "route"
        assert logged[0].snapshot_id == "snap-test"
        assert logged[0].audiences == ("public",)
    finally:
        repo.close()


def test_a_log_that_fails_does_not_cost_the_answer(tmp_path, no_ollama, monkeypatch):
    """A store that cannot write the audit row still has a good answer in hand."""
    repo = build_repo(tmp_path)
    try:
        assistant = Assistant(repo)

        def refuse(_entry):
            raise RuntimeError("the log is unavailable")

        monkeypatch.setattr(repo, "log_answer", refuse)
        reply = assistant.ask("How much does Solo cost")

        assert reply.parts, "the answer was lost to an audit failure"
        assert "does not publish prices" in reply.parts[0][1].text
    finally:
        repo.close()


def test_logging_can_be_turned_off(tmp_path, no_ollama):
    repo = build_repo(tmp_path)
    try:
        Assistant(repo, log=False).ask("How much does Solo cost")
        assert repo.answer_log(limit=5) == []
    finally:
        repo.close()


def test_a_cached_answer_is_not_shared_between_callers(assistant, monkeypatch):
    """The cache held one Answer and handed the same object to everyone.

    Each caller then wrote its own correlation id onto it, so on the threading
    server two concurrent readers of the same cached answer would each find the
    other's trace in their diagnostics. The text was never at risk; the ability
    to trace it was, which is the one thing the id exists for.
    """
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: ("Mix Solo with 5-6 litres of clean water per 25 kg "
                           "sack [1].", 0.01))

    first = assistant.ask("How much water does Solo need", correlation_id="trace-one")
    second = assistant.ask("How much water does Solo need", correlation_id="trace-two")

    one, two = first.parts[0][1], second.parts[0][1]
    assert two.diagnostics["cached"] is True, "the second ask should have hit the cache"
    assert one is not two, "both callers were handed the same Answer object"
    assert one.diagnostics["correlation_id"] == "trace-one"
    assert two.diagnostics["correlation_id"] == "trace-two"
    assert one.text == two.text


def test_the_first_caller_of_a_cached_answer_is_not_marked_cached(assistant,
                                                                  monkeypatch):
    """`cached` must describe this reply, not leak backwards onto the original."""
    monkeypatch.setattr(
        ollama, "generate",
        lambda *_a, **_k: ("Mix Solo with 5-6 litres of clean water per 25 kg "
                           "sack [1].", 0.01))

    first = assistant.ask("How much water does Solo need")
    assistant.ask("How much water does Solo need")

    assert "cached" not in first.parts[0][1].diagnostics


# ------------------------------------------- the calculation edge's second look


def test_a_quantity_question_fetches_the_coverage_passage_it_did_not_rank(
        tmp_path, monkeypatch):
    """Re-sorting the top five cannot surface a passage that was never in it.

    "How many bags for twenty square metres" embeds as a question about
    quantity, so the coverage figure it needs can rank outside the window
    entirely — and the calculation path used to be handed those five and left
    to make the best of them. It now asks a second time by metadata instead:
    the product the question named, and the words coverage is printed under.

    Those hits carry no similarity score, so they are appended after the ranked
    ones and cannot reach the abstention threshold, which step 1 has already
    decided.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))

    repo = build_repo(tmp_path)
    # Six passages near the question and the coverage passage far from it, so
    # the top five fill with the near ones and coverage genuinely never ranks.
    # That is the case being fixed: re-sorting five hits cannot surface a
    # passage that was never among them.
    near = [chunk(SOLO, i, f"Section {i}",
                  f"Solo is applied in one coat. Note {i} about application.",
                  "Solo", 0)
            for i in range(6)]
    repo.publish(
        [document(SOLO, "Solo")], [version(SOLO)],
        [*near,
         chunk(SOLO, 9, "Coverage",
               "Solo covers 1.5 m2 per 25 kg sack at 10 mm.", "Solo", 7)],
        Snapshot(snapshot_id="snap-calc", created_at="2026-01-01T00:00:00Z",
                 embedding_model=ollama.EMBED_MODEL,
                 embedding_dimensions=ollama.EMBED_DIMENSIONS,
                 chunking_version=CHUNKING_VERSION, document_count=1,
                 chunk_count=7,
                 notes={"products": ["Solo"], "colours": [], "merchants": [],
                        "contact": CONTACT}))
    try:
        assistant = Assistant(repo, cache=False, log=False)
        reply = assistant.ask("How many bags of Solo do I need for 20 square metres?")
        answer = reply.parts[0][1]

        assert answer.path == Path_.EXTRACT.value, answer.path
        assert "1.5 m2" in answer.text, answer.text
        assert "I have printed the published coverage" in answer.text
    finally:
        repo.close()


def test_a_coverage_passage_already_retrieved_is_not_added_twice(tmp_path,
                                                                 monkeypatch):
    """The second look must not duplicate what similarity already found.

    A duplicated passage would be cited twice, counted twice by the
    per-document cap, and read as two sources agreeing when it is one source
    repeated.
    """
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))

    repo = build_repo(tmp_path)
    repo.publish(
        [document(SOLO, "Solo")], [version(SOLO)],
        [chunk(SOLO, 0, "Coverage",
               "Solo covers 1.5 m2 per 25 kg sack at 10 mm.", "Solo", 0)],
        Snapshot(snapshot_id="snap-dup", created_at="2026-01-01T00:00:00Z",
                 embedding_model=ollama.EMBED_MODEL,
                 embedding_dimensions=ollama.EMBED_DIMENSIONS,
                 chunking_version=CHUNKING_VERSION, document_count=1,
                 chunk_count=1,
                 notes={"products": ["Solo"], "colours": [], "merchants": [],
                        "contact": CONTACT}))
    try:
        reply = Assistant(repo, cache=False, log=False).ask(
            "How many bags of Solo do I need for 20 square metres?")
        answer = reply.parts[0][1]

        assert answer.text.count("1.5 m2") == 1, answer.text
        assert len(answer.sources) == 1, answer.sources
    finally:
        repo.close()


def test_a_quantity_question_naming_no_product_still_answers(tmp_path, monkeypatch):
    """The second look needs a product name to ask by, and may simply not get one."""
    monkeypatch.setattr(ollama, "embed_one", lambda *_a, **_k: unit(0))

    repo = build_repo(tmp_path)
    try:
        reply = Assistant(repo, cache=False, log=False).ask(
            "How many bags do I need for 20 square metres?")
        assert reply.parts, "a quantity question with no product named produced nothing"
    finally:
        repo.close()
