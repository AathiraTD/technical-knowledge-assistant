"""The seven answer paths, and what each one is obliged to carry.

`test_checks.py` covers the six checks in isolation. This covers everything
around them: what Extract prints, what a refusal still hands over, what a
document request answers from, and the two rules a caveat has to obey.

The recurring assertion is that a failure still carries value. A refusal that
names nothing, cites nothing and gives no way forward is a worse outcome than a
wrong answer, because the customer is left with neither information nor a
person to call.

Only Compose touches the model, and that call is replaced here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import ollama                                   # noqa: E402
from assistant.answer import (                                 # noqa: E402
    PHOTO_LINE,
    AnswerEngine,
    _caveat_lines,
    _contact_line,
    _source_rows,
)
from assistant.model import (                                  # noqa: E402
    Caveat, Chunk, Document, DocumentVersion, Excluded, Retrieved, Snapshot,
)
from assistant.router import Decision, Path_, Router           # noqa: E402
from assistant.store import SQLiteKnowledgeRepository          # noqa: E402

DIMS = ollama.EMBED_DIMENSIONS
SOLO_URL = "https://example/solo"
DURO_URL = "https://example/duro"

CONTACT = {"phone": "0800 538 5746", "hours": "Mon - Fri 9:00am - 5:00pm"}
NOTES = {"products": ["Solo Onecoat Lime Plaster", "Duro Lime Plaster"],
         "colours": ["York", "Cotswold"], "merchants": ["Womersley's"],
         "contact": CONTACT}


def vec(first: float = 1.0) -> list[float]:
    return [first] + [0.0] * (DIMS - 1)


def chunk(url, i, section, content, product, dtype="datasheet", authority=1):
    return Chunk(canonical_url=url, version=1, chunk_index=i, section=section,
                 content=content, product=product, document_type=dtype,
                 authority=authority, source_date="2026-01-01", embedding=vec())


def hit(url, section, content, product, dtype="datasheet", authority=1,
        score=0.8) -> Retrieved:
    return Retrieved(
        chunk=chunk(url, 0, section, content, product, dtype, authority),
        score=score,
        document=Document(canonical_url=url, title=f"{product} | Lime Green",
                          document_type=dtype, authority=authority,
                          product=product, link_text=f"{product} datasheet"),
    )


MIXING = hit(SOLO_URL, "Mixing",
             "Add between 5 and 6 litres of clean water per 25kg sack.",
             "Solo Onecoat Lime Plaster")
COVERAGE = hit(SOLO_URL, "Coverage",
               "Coverage is approximately 2 m2 per 25 kg sack at 10 mm.",
               "Solo Onecoat Lime Plaster")
DURO = hit(DURO_URL, "Description", "Duro is a general purpose lime undercoat.",
           "Duro Lime Plaster")


def store() -> SQLiteKnowledgeRepository:
    repo = SQLiteKnowledgeRepository(Path(tempfile.mkdtemp()) / "a.db")
    repo.publish(
        [Document(canonical_url=SOLO_URL, title="Solo Onecoat | Lime Plaster | Lime Green",
                  document_type="datasheet", authority=1,
                  product="Solo Onecoat Lime Plaster", link_text="Solo Datasheet"),
         Document(canonical_url=DURO_URL, title="Duro", document_type="datasheet",
                  authority=1, product="Duro Lime Plaster",
                  link_text="Duro Datasheet")],
        [DocumentVersion(canonical_url=SOLO_URL, version=1, content_hash="h",
                         source_path="s", is_active=True, fetched_at="2026-01-01"),
         DocumentVersion(canonical_url=DURO_URL, version=1, content_hash="h",
                         source_path="d", is_active=True, fetched_at="2026-01-01")],
        [chunk(SOLO_URL, 0, "Mixing", MIXING.chunk.content, "Solo Onecoat Lime Plaster"),
         chunk(DURO_URL, 0, "Description", DURO.chunk.content, "Duro Lime Plaster")],
        Snapshot(snapshot_id="s1", created_at="2026-01-01T00:00:00Z",
                 embedding_model=ollama.EMBED_MODEL, embedding_dimensions=DIMS,
                 chunking_version="test/1.0", document_count=2, chunk_count=2,
                 notes=NOTES),
        caveats=[
            Caveat(SOLO_URL, "temperature",
                   "Ensure the room remains above 5 degrees C.", "Mixing"),
            Caveat(SOLO_URL, "incompatibility",
                   "Many MgO boards are not suitable for Solo.", "Application"),
            Caveat(SOLO_URL, "diy",
                   "A long sentence about water and mixing that mentions water "
                   "repeatedly so it scores on question overlap.", "Mixing"),
            Caveat(DURO_URL, "temperature", "Duro must not be applied in frost.",
                   "Curing"),
        ],
        excluded=[Excluded("https://example/sds.pdf",
                           "safety data sheets are read whole, not in fragments")],
    )
    return repo


def engine(repo=None) -> AnswerEngine:
    repo = repo or store()
    made = AnswerEngine(repo, Router())
    made.retriever = Router()
    return made


def decision(path: Path_, hits, **kw) -> Decision:
    return Decision(path, kw.pop("reason", "because"), kw.pop("step", "7"),
                    hits=hits, **kw)


# ------------------------------------------------------------------- extract


def test_extract_prints_the_passage_whole_and_names_its_document():
    """On a lookup the model can contribute nothing but paraphrase drift."""
    a = engine().extract(decision(Path_.EXTRACT, [MIXING]))
    assert "between 5 and 6 litres of clean water per 25kg sack" in a.text
    assert "Solo Onecoat Lime Plaster" in a.text
    assert a.path == Path_.EXTRACT.value
    assert a.refused is False
    assert len(a.sources) == 1


def test_a_quantity_question_says_the_sum_was_refused():
    """The arithmetic depends on background and thickness, and silence would look like an oversight."""
    a = engine().extract(decision(Path_.EXTRACT, [COVERAGE], sum_refused=True))
    assert "rather than multiplying" in a.text


# -------------------------------------------------------------------- refuse


def test_a_refusal_still_carries_the_nearest_passage_and_the_contact_line():
    """A refusal that gives neither information nor a person to call is worse than useless."""
    a = engine().refuse(decision(Path_.REFUSE, [MIXING]), "nothing close enough")
    assert a.refused is True
    assert "5 and 6 litres" in a.text
    assert "0800 538 5746" in a.text
    assert a.diagnostics["refusal_reason"] == "nothing close enough"


def test_a_near_miss_refusal_names_the_property_it_looked_for():
    """'Not stated' is only useful if it says what was not stated."""
    a = engine().refuse(
        decision(Path_.REFUSE, [MIXING], missing_term="thermal"), "near miss")
    assert "thermal" in a.text
    assert "does not state" in a.text


def test_a_refusal_with_nothing_retrieved_still_hands_over():
    a = engine().refuse(decision(Path_.REFUSE, []), "nothing at all")
    assert a.refused is True
    assert "0800 538 5746" in a.text
    assert a.sources == []


# ---------------------------------------------------------- deferral and diagnosis


def test_a_cited_hand_off_quotes_the_published_deferral():
    """The company's own instruction stands, and it is quoted rather than paraphrased."""
    deferring = hit(SOLO_URL, "Application",
                    "Many MgO boards are not suitable for Solo. Contact us first.",
                    "Solo Onecoat Lime Plaster")
    a = engine().defer(decision(Path_.DEFER, [deferring]))
    assert "Contact us first" in a.text
    assert "refers this question to their technical team" in a.text


def test_diagnosis_quotes_published_causes_and_still_hands_over():
    """Judging a wall is a human decision, so the reply informs rather than concludes."""
    a = engine().diagnosis(decision(Path_.DIAGNOSIS, [MIXING, DURO]))
    assert "human judgement" in a.text or "judgement" in a.text
    assert "0800 538 5746" in a.text
    assert len(a.sources) == 2


def test_ask_back_names_the_detail_it_needs():
    """An ask-back that does not say what to send wastes the customer's next turn."""
    a = engine().ask_back(decision(Path_.ASK_BACK, [MIXING]))
    assert "built of" in a.text
    assert "brick" in a.text.lower()
    assert a.refused is False


# --------------------------------------------------------------------- route


def test_a_policy_route_substitutes_the_harvested_contact_details():
    """A phone number the assistant invents is close to the worst thing it can print."""
    router = Router()
    _topic, spec = router.gate.match("How much does Solo cost?")
    a = engine().route("price", spec)
    assert "0800 538 5746" in a.text
    assert "{contact}" not in a.text and "{hours}" not in a.text
    assert a.path == Path_.ROUTE.value
    assert a.sources == []


def test_a_document_request_is_answered_from_the_manifest():
    """The answer is the document list itself, so it must not depend on retrieval."""
    a = engine().documents_for("Which Solo datasheet is current?", ("public",))
    assert "Solo Datasheet" in a.text
    assert SOLO_URL in a.text
    assert a.diagnostics["step"] == "manifest"


def test_a_safety_sheet_request_explains_why_it_is_not_indexed():
    """'Why isn't that in here?' has an answer on file rather than an improvised one."""
    a = engine().documents_for("Can you send me the safety data sheet?", ("public",))
    assert "controlled documents" in a.text
    assert "excluded by rule" in a.text


def test_a_document_request_matching_nothing_falls_back_to_the_datasheets():
    a = engine().documents_for("zzz qqq", ("public",))
    assert "Datasheet" in a.text


# ------------------------------------------------------------------- caveats


def test_caveats_come_only_from_the_documents_actually_cited():
    """An early version appended Roman Stucco's curing limits to an answer about Solo."""
    lines = _caveat_lines(decision(Path_.EXTRACT, [MIXING]), store(),
                          "how much water")
    assert lines
    assert not any("Duro" in line for line in lines)


def test_at_most_three_caveats_are_appended():
    """A document with five relevant caveats shows the three that best match."""
    assert len(_caveat_lines(decision(Path_.EXTRACT, [MIXING]), store(),
                             "how much water")) <= 3


def test_the_question_scores_the_caveats():
    """Matching on the slot value alone ranks almost nothing."""
    lines = _caveat_lines(decision(Path_.EXTRACT, [MIXING]), store(),
                          "how much water and mixing")
    assert any("water" in line.lower() for line in lines)


# ------------------------------------------------------- shared presentation


def test_the_photograph_line_is_appended_to_whatever_path_ran():
    """The line keys on the slot, not the path, so a refusal carries it too."""
    for made in (engine().extract(decision(Path_.EXTRACT, [MIXING], photograph=True)),
                 engine().refuse(decision(Path_.REFUSE, [MIXING], photograph=True), "x")):
        assert PHOTO_LINE in made.text


def test_an_uncued_location_is_stated_as_an_assumption():
    """An assumption the reader cannot see is an assumption they cannot correct."""
    a = engine().extract(decision(Path_.EXTRACT, [MIXING], per_option=True))
    assert any("internal and external" in x for x in a.assumptions)


def test_cued_slots_are_stated_back():
    a = engine().extract(decision(Path_.EXTRACT, [MIXING],
                                  slots={"substrate": "lath", "location": "internal"}))
    assert any("lath" in x for x in a.assumptions)


def test_citation_names_drop_the_search_engine_tail():
    """A citation has to be checkable by a plasterer, not a page title."""
    rows = _source_rows([MIXING])
    assert rows[0]["name"] == "Solo Onecoat Lime Plaster datasheet"
    assert "|" not in rows[0]["name"]


def test_source_rows_mark_the_first_appearance_of_each_document():
    rows = _source_rows([MIXING, COVERAGE, DURO])
    assert [r["first"] for r in rows] == [True, False, True]


def test_the_contact_line_degrades_without_falling_back_to_invention():
    """With no harvested number, it points at the contact page rather than guessing one."""
    assert "0800 538 5746" in _contact_line({"contact": CONTACT})
    assert "9:00am" in _contact_line({"contact": CONTACT})
    assert "0800 538 5746" in _contact_line({"contact": {"phone": "0800 538 5746"}})
    assert "lime-green.co.uk/contact" in _contact_line({})


# -------------------------------------------------------------------- compose


def test_compose_publishes_an_answer_that_passes_its_checks(monkeypatch):
    monkeypatch.setattr(
        ollama, "generate",
        lambda *a, **k: ("Add between 5 and 6 litres of clean water per 25kg sack [1].",
                         1.5))
    a = engine().compose(
        decision(Path_.COMPOSE, [MIXING], step="8",
                 slots={"property_asked": "water"}), "How much water?")
    assert a.refused is False
    assert a.failed_checks == []
    assert a.diagnostics["generation_seconds"] == 1.5


def test_a_failed_check_becomes_a_refusal_rather_than_a_repair(monkeypatch):
    """A generated figure that is not in a passage does not print, and is not retried."""
    monkeypatch.setattr(
        ollama, "generate",
        lambda *a, **k: ("Add 9 litres of clean water per 25kg sack [1].", 1.0))
    a = engine().compose(
        decision(Path_.COMPOSE, [MIXING], step="8",
                 slots={"property_asked": "water"}), "How much water?")
    assert a.refused is True
    assert any("check 2" in c for c in a.failed_checks)
    assert "0800 538 5746" in a.text


def test_a_stopped_model_server_is_reported_and_not_disguised(monkeypatch, caplog):
    """A missing model must fail loudly, not silently become a refusal.

    The distinction matters operationally. A refusal says the corpus does not
    answer the question and is a designed outcome; a stopped Ollama says the
    deployment is broken. Collapsing the second into the first would hide an
    outage behind an answer that looks deliberate, and the refusal rate would
    stay flat while the system served nothing.
    """
    import pytest

    def unavailable(*_a, **_k):
        raise ollama.OllamaUnavailable("connection refused on 127.0.0.1:11434")

    monkeypatch.setattr(ollama, "generate", unavailable)

    with caplog.at_level("INFO", logger="assistant"):
        with pytest.raises(ollama.OllamaUnavailable):
            engine().compose(
                decision(Path_.COMPOSE, [MIXING], step="8",
                         slots={"property_asked": "water"}), "How much water?")

    logged = [r for r in caplog.records if getattr(r, "event", "") == "ollama_error"]
    assert logged, "an outage passed through without an event"
    assert logged[0].fields["stage"] == "generate"


# ------------------------------------------------- the remaining edge branches


def test_a_sentence_with_no_figures_skips_the_attribution_check():
    """Check 3 compares figures against products; a sentence with none has nothing to compare."""
    from assistant.answer import run_checks
    failures = run_checks(
        "Solo is a one-coat lime plaster for interior use [1].",
        [hit(SOLO_URL, "Description",
             "Solo is a one-coat lime plaster for interior use on masonry.",
             "Solo Onecoat Lime Plaster")],
        NOTES, [])
    assert not any(f.startswith("check 3") for f in failures), failures


def test_an_empty_answer_produces_no_failures():
    """An empty string is refused earlier; the checks must not invent a reason of their own."""
    from assistant.answer import run_checks
    assert run_checks("", [MIXING], NOTES, []) == []


def test_a_qualifier_cited_to_no_retrieved_passage_is_caught():
    """A marker pointing nowhere must not silently satisfy the qualifier check."""
    from assistant.answer import run_checks
    failures = run_checks("Apply at a minimum thickness of 8 mm [9].",
                          [MIXING], NOTES, [])
    assert any(f.startswith("check 1") for f in failures)


def test_the_calculation_edge_prints_the_coverage_passage_not_the_top_one():
    """Extract printed whichever passage ranked first, and then said otherwise.

    "How many bags of Duro for 20 square metres" ranks Duro's *mixing water*
    above its coverage, because retrieval scores the whole question. Extract
    printed that, then appended a code-written sentence claiming it had shown
    "the published coverage and pack size". Unsupported prose reaching the page
    is exactly what the six checks exist to stop — and this arrived by a door
    they do not watch, because code wrote it rather than the model.
    """
    mixing = hit(SOLO_URL, "Mixing",
                 "Add approximately 4.5 to 5 litres of water per bag. "
                 "Mix for between 3 and 10 minutes.", "Duro")
    coverage = hit(SOLO_URL, "Storage",
                   "Store in a dry place. At 10mm thick 1 bag will cover 1 m 2.",
                   "Duro")

    answer = engine().extract(
        decision(Path_.EXTRACT, [mixing, coverage], step="6", sum_refused=True))

    assert "1 bag will cover" in answer.text, answer.text
    assert "I have printed the published coverage" in answer.text


def test_the_calculation_edge_says_so_when_no_coverage_was_published():
    """Solo Filler has no datasheet at all; some products publish no coverage.

    The sentence must describe what was printed. Claiming a coverage figure
    that is not in the evidence is the same defect in the other direction.
    """
    mixing = hit(SOLO_URL, "Mixing",
                 "Add approximately 4.5 to 5 litres of water per bag.", "Duro")

    answer = engine().extract(
        decision(Path_.EXTRACT, [mixing], step="6", sum_refused=True))

    assert "does not state a coverage figure" in answer.text, answer.text
    assert "I have printed the published coverage" not in answer.text
