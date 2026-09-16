"""Indexing, tested on the rules that decide what a passage is.

Chunking is where the corpus is most dangerous. Solo's application section is
one bullet per background at roughly 2,000 characters, and a splitter that cuts
by size takes a thickness away from the substrate it applies to. Nothing
downstream can detect that: the passage is real, the citation is real, and the
figure now belongs to the wrong wall. So the bullet rule, the merge rule and
the hard ceiling are tested against the shapes that break them.

The tidying functions look cosmetic and are not. `product_name` feeds check 3,
which compares a figure against the product it was published for, and it is
prefixed onto the chunk before embedding, so a page title dressed as a product
name costs twice. `iso_date` is what "newest wins within a type" actually
compares, and left as published a lexical sort puts 9/12/24 before 5/9/24.

The build tests run the whole ingestion path over the real shipped cache with
three seams replaced: the model server, the embedding cache location and the
output directory. No network, no Ollama and no pre-built index are required,
and the repository under test is the real SQLite adapter rather than a mock,
so the publication path is exercised for real.
"""

from __future__ import annotations

import json
from dataclasses import replace
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import index                                       # noqa: E402
from assistant.embedcache import EmbeddingCache                    # noqa: E402
from assistant.extract import Section                              # noqa: E402
from assistant.index import (                                      # noqa: E402
    HARD_MAX,
    MIN_CHARS,
    TARGET_CHARS,
    _split_on_bullets,
    chunk_sections,
    embedding_text,
    iso_date,
    product_name,
    summarise,
)
from assistant.model import Chunk                                  # noqa: E402
from assistant.ollama import OllamaUnavailable                     # noqa: E402
from assistant.store import EmbeddedRepository                     # noqa: E402

CACHE = ROOT / "data" / "cache"

SENTENCE = "Apply a coat of eight millimetres and allow it to firm up before the next. "


def bullet_list(count: int, repeats: int = 9) -> str:
    return "\n".join(f"- Background {i}: " + SENTENCE * repeats for i in range(count))


# ------------------------------------------------------------------ chunking


def test_a_bullet_is_never_split_mid_item():
    """A thickness cut away from the background it applies to is the worst thing this corpus can do."""
    text = bullet_list(3)
    assert len(text) > TARGET_CHARS

    out = chunk_sections([Section("Preparation & Application", text)])
    assert len(out) > 1
    for original in text.split("\n"):
        assert sum(original.strip() in body for _, body in out) == 1
    for _, body in out:
        assert body.startswith("- Background")


def test_an_over_long_single_bullet_stays_whole_rather_than_being_cut():
    """One bullet longer than the target is left over-long on purpose; cutting it loses the condition."""
    text = "- Background 0: " + SENTENCE * 40
    assert len(text) > TARGET_CHARS

    out = chunk_sections([Section("Application", text)])
    assert len(out) == 1
    assert out[0][1] == text.strip()


def test_a_short_section_is_merged_into_the_one_after_it():
    """A two-line 'Colours' heading retrieves as noise alone and as context when attached."""
    short = Section("Colours", "Twenty four colours.")
    assert len(short.text) < MIN_CHARS

    out = chunk_sections([short, Section("Mixing", "Add water carefully. " * 20)])
    assert len(out) == 1
    heading, body = out[0]
    assert heading == "Colours / Mixing"
    assert body.startswith("Twenty four colours.")
    assert "Add water carefully." in body


def test_a_trailing_short_section_is_carried_onto_the_previous_passage():
    """A closing note dropped on the floor is published material that becomes unretrievable."""
    out = chunk_sections([
        Section("Mixing", "Add water carefully. " * 12),
        Section("Note", "Keep it damp."),
    ])
    assert len(out) == 1
    assert out[0][0] == "Mixing"
    assert out[0][1].endswith("Keep it damp.")


def test_a_document_that_is_nothing_but_one_short_section_still_indexes():
    """Otherwise a thin page extracts to nothing and its content silently leaves the corpus."""
    assert chunk_sections([Section("", "Just a note.")]) == [("Introduction", "Just a note.")]


def test_an_empty_section_produces_no_passage():
    """An empty chunk matches nothing and prints nothing, so writing one only inflates the count."""
    assert chunk_sections([Section("Empty", "   ")]) == []


def test_a_long_bullet_less_section_under_the_ceiling_is_kept_whole():
    """An over-long passage costs context; a badly cut one costs correctness, so the ceiling is high."""
    text = "Sentence about preparing the wall before work begins. " * 40
    assert TARGET_CHARS < len(text) <= HARD_MAX

    out = chunk_sections([Section("Description", text)])
    assert len(out) == 1


def test_a_bullet_less_section_over_the_ceiling_splits_at_blank_lines_only():
    """Above the ceiling something must give, and a paragraph boundary is the only safe place to cut."""
    paragraph = "Paragraph about preparing the wall before work begins. " * 14
    text = "\n\n".join(paragraph for _ in range(6))
    assert len(text) > HARD_MAX

    out = chunk_sections([Section("Description", text)])
    assert len(out) > 1
    assert all(body.startswith("Paragraph about") for _, body in out)
    assert all(heading == "Description" for heading, _ in out)


def test_every_passage_of_the_real_solo_datasheet_keeps_its_bullets_intact():
    """The section this rule was written for, checked against the sheet rather than against a fixture."""
    from assistant.extract import extract_pdf

    sheet = CACHE / "documents" / "solo-one-coat-lime-plaster-tds.pdf"
    if not sheet.exists():
        pytest.skip("the shipped cache is missing")

    sections = extract_pdf(str(sheet)).sections
    application = [s for s in sections if "Application" in s.heading]
    assert application

    out = chunk_sections(application)
    for line in application[0].text.split("\n"):
        if line.strip().startswith(("-", "•")):
            assert any(line.strip() in body for _, body in out)


# ----------------------------------------------------------- bullet splitting


def test_a_section_with_a_single_bullet_is_not_split():
    """One marker is a list of one, and splitting on it separates the bullet from its own lead-in."""
    assert _split_on_bullets("Lead prose.\n- only one") == ["Lead prose.\n- only one"]
    assert _split_on_bullets("no bullets here at all") == ["no bullets here at all"]


def test_lead_prose_above_the_first_bullet_is_kept_as_its_own_piece():
    """The sentence introducing a list usually carries the condition the list applies under."""
    assert _split_on_bullets("Lead prose before.\n- one\n- two") == [
        "Lead prose before.", "- one", "- two"]


def test_a_section_beginning_on_a_bullet_produces_no_empty_leading_piece():
    """An empty first piece becomes an empty chunk, which retrieves on nothing."""
    assert _split_on_bullets("- one\n- two\n- three") == ["- one", "- two", "- three"]


def test_lettered_and_numbered_markers_count_as_bullets():
    """The sheets mark sub-steps 'a)' and '1.' as often as they use a dash."""
    assert _split_on_bullets("a) first step\nb) second step") == ["a) first step", "b) second step"]
    assert _split_on_bullets("1. first step\n2. second step") == ["1. first step", "2. second step"]


# -------------------------------------------------------------- product_name


def test_the_search_engine_tail_is_not_part_of_the_product_name():
    """The crawl records the page title, and 'Natural Lime Mortar | Lime Green' fails check 3 as a name."""
    assert product_name(
        {"url": "https://x/products/lime-mortar/natural-lime-mortar"},
        "Natural Lime Mortar for Old Buildings, Repointing, Pointing Repair "
        "and Stone Walls | Lime Green",
    ) == "Natural Lime Mortar"


def test_three_products_in_one_title_become_the_first_of_them():
    """A chunk prefixed with three product names retrieves for all three and belongs to none."""
    assert product_name(
        {"product": "Aerogel wall insulation, Solo Onecoat Lime Plaster / "
                    "Silic8 AeroGel Adhesive",
         "url": "https://x/a"},
        "",
    ) == "Aerogel wall insulation"


def test_a_document_with_no_product_falls_back_to_its_url_slug():
    """An empty product prefix leaves a coverage figure with nothing tying it to a product at all."""
    assert product_name(
        {"product": "", "url": "https://x/documents/Fine-Stuff-TDS.pdf"}, "",
    ) == "Fine Stuff Tds"


def test_an_escaped_url_slug_is_readable_as_a_name():
    """'Warmshell%20Aerogel_Board.pdf' prefixed onto a passage is noise to the embedding model."""
    assert product_name(
        {"product": "", "url": "https://x/documents/Warmshell%20Aerogel_Board-v2.pdf"}, "",
    ) == "Warmshell Aerogel Board V2"


def test_an_initials_length_product_name_is_treated_as_missing():
    """Two characters is not a name, and it would be prefixed onto every chunk of that document."""
    assert product_name(
        {"product": "AB", "url": "https://x/documents/solo-tds.pdf"}, "",
    ) == "Solo Tds"


def test_the_entry_product_wins_over_the_page_title():
    """The crawl's own classification is better evidence than the title the site chose for search."""
    assert product_name(
        {"product": "Solo Onecoat Lime Plaster", "url": "https://x/a"}, "ignored",
    ) == "Solo Onecoat Lime Plaster"


def test_a_runaway_name_is_truncated():
    """The name is a column and a prefix; an unbounded one distorts the embedding of every chunk."""
    assert len(product_name({"product": "X" * 200, "url": "https://x/a"}, "")) == 80


# ------------------------------------------------------------------ iso_date


@pytest.mark.parametrize("printed, expected", [
    ("9/12/24", "2024-12-09"),
    ("5/9/24", "2024-09-05"),
    ("7/6/19", "2019-06-07"),
    ("29/5/25", "2025-05-29"),
    ("1-3-24", "2024-03-01"),
    ("9.12.2024", "2024-12-09"),
])
def test_a_printed_sheet_date_is_read_as_day_first(printed, expected):
    """Lexically, 9/12/24 sorts before 5/9/24, which inverts the rule newest-wins exists to serve."""
    assert iso_date(printed) == expected


@pytest.mark.parametrize("printed, expected", [
    ("4 th April 2017", "2017-04-04"),
    ("20 July 2017", "2017-07-20"),
    ("17 Sep 2020", "2020-09-17"),
])
def test_an_article_date_with_an_ordinal_suffix_is_read(printed, expected):
    """The site prints '4 th April 2017' with the suffix split off, and every article carries one."""
    assert iso_date(printed) == expected


@pytest.mark.parametrize("printed", ["", "not a date at all", "32 Smarch 2017"])
def test_an_unparseable_date_falls_back_to_the_crawl_date(printed):
    """A guessed date would outrank a real one under newest-wins; the crawl date is at least true."""
    assert iso_date(printed, "2026-09-15T12:22:44+00:00") == "2026-09-15"


def test_a_document_with_no_date_at_all_yields_an_empty_string():
    """An invented date is worse than no date, so the field stays empty and the tiebreak simply abstains."""
    assert iso_date("") == ""


# ------------------------------------------------------------- embedding_text


def test_the_retrieval_view_names_the_product_the_passage_belongs_to():
    """'Add 5-6 litres per 25kg sack' never says Solo, so a question naming Solo could not reach it."""
    chunk = Chunk(canonical_url="u", version=1, chunk_index=0, section="Mixing",
                  content="Add water.", product="Solo Onecoat Lime Plaster")
    assert embedding_text(chunk) == "Solo Onecoat Lime Plaster — Mixing\nAdd water."


def test_a_passage_with_no_product_is_still_prefixed_by_its_section():
    """An article has no product, and its section heading is the only context the passage carries."""
    chunk = Chunk(canonical_url="u", version=1, chunk_index=0, section="Mixing",
                  content="Add water.", product="")
    assert embedding_text(chunk) == "Mixing\nAdd water."


def test_a_passage_with_neither_is_embedded_exactly_as_published():
    """A dangling separator embedded as context is noise, and it is not what the page says."""
    chunk = Chunk(canonical_url="u", version=1, chunk_index=0, section="",
                  content="Add water.", product="")
    assert embedding_text(chunk) == "Add water."


def test_the_printed_passage_is_not_what_gets_embedded():
    """The prefix is the retrieval view only; if it leaked into the chunk it would print as published text."""
    chunk = Chunk(canonical_url="u", version=1, chunk_index=0, section="Mixing",
                  content="Add water.", product="Solo")
    embedding_text(chunk)
    assert chunk.content == "Add water."


# ------------------------------------------------------------------ summarise


def base_report(**overrides) -> dict:
    report = {
        "snapshot": "snap-20260915T120000Z",
        "public_documents": 94, "evaluation_fixtures": 1,
        "chunks": 601, "caveats": 42, "excluded": 60,
        "colours": ["York"], "products": ["Solo"], "merchants": ["The Lime Centre"],
        "embedding_model": "qwen3-embedding:0.6b", "embedding_dimensions": 1024,
        "embed_seconds": 11.0, "total_seconds": 21.0,
        "rows": [{"quality": "clean", "name": "Solo.pdf", "note": ""}],
    }
    report.update(overrides)
    return report


def test_a_document_that_extracted_badly_is_named_in_the_summary():
    """An assessor should read that a twenty-page guide produced two sections, not infer it from a bad answer."""
    text = summarise(base_report(rows=[
        {"quality": "clean", "name": "Solo.pdf", "note": ""},
        {"quality": "flat", "name": "roof-guide.pdf", "note": "no headings detected"},
        {"quality": "failed", "name": "gone.pdf", "note": "cached file missing"},
    ]))
    assert "documents that extracted badly" in text
    assert "[flat] roof-guide.pdf" in text
    assert "[failed] gone.pdf" in text
    assert "clean 1, failed 1, flat 1" in text


def test_a_clean_run_says_so_rather_than_printing_nothing():
    """Silence reads as 'the report was not written'; the absence of failures has to be stated."""
    assert "no document failed extraction." in summarise(base_report())


def test_the_fixture_count_reads_correctly_in_both_singular_and_plural():
    """The staff fixture is the only non-public document indexed, and miscounting it misreports the corpus."""
    assert "1 evaluation fixture (staff-only)" in summarise(base_report())
    assert "2 evaluation fixtures (staff-only)" in summarise(
        base_report(evaluation_fixtures=2))


# --------------------------------------------------------------------- build
#
# The three seams replaced below are the model server, the embedding cache
# location and the output directory. Everything else is real: the real cached
# HTML and PDFs, the real extraction, the real chunking and the real SQLite
# adapter. Nothing here reaches the network or Ollama.


PAGES = CACHE / "pages"
DOCS = CACHE / "documents"

needs_cache = pytest.mark.skipif(
    not (PAGES / "contact.html").exists(),
    reason="the shipped cache is missing; the build path has nothing to run against",
)


def build_once(repo, verbose: bool = False) -> dict:
    return index.build(repo=repo, verbose=verbose)


def fake_vector(text: str) -> list[float]:
    """A deterministic vector, so a cache hit is provably the same value as the miss that wrote it."""
    seed = int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return [((seed + i) % 1000) / 1000.0 for i in range(index.ollama.EMBED_DIMENSIONS)]


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A trimmed crawl log over real cached files, with the model server stubbed out."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    fetched = [
        {"url": "https://www.lime-green.co.uk/products/lime-plaster/solo-onecoat-plaster",
         "path": "data/cache/pages/products-lime-plaster-solo-onecoat-plaster.html",
         "kind": "page", "doc_type": "product_page",
         "title": "Solo Onecoat Lime Plaster | Lime Green", "link_text": "", "product": ""},
        {"url": "https://www.lime-green.co.uk/find-a-supplier",
         "path": "data/cache/pages/find-a-supplier.html",
         "kind": "page", "doc_type": "commercial",
         "title": "Find a supplier", "link_text": "", "product": ""},
        {"url": "https://www.lime-green.co.uk/contact",
         "path": "data/cache/pages/contact.html",
         "kind": "page", "doc_type": "commercial",
         "title": "Contact", "link_text": "", "product": ""},
        {"url": "https://www.lime-green.co.uk/support/faq",
         "path": "data/cache/pages/support-faq.html",
         "kind": "page", "doc_type": "faq",
         "title": "FAQ", "link_text": "", "product": ""},
        # No title on the entry: the name has to come out of the PDF itself.
        {"url": "https://www.lime-green.co.uk/documents/Fine-Stuff-TDS.pdf",
         "path": "data/cache/documents/Fine-Stuff-TDS.pdf",
         "kind": "document", "doc_type": "datasheet", "link_text": "Data Sheet"},
        {"url": "https://www.lime-green.co.uk/documents/vanished.pdf",
         "path": "data/cache/documents/vanished.pdf",
         "kind": "document", "doc_type": "datasheet", "title": "Vanished"},
    ]
    skipped = [
        {"url": "https://www.lime-green.co.uk/", "reason": "outside the corpus boundary"},
        {"url": "https://www.lime-green.co.uk/news"},        # no reason recorded
    ]
    (cache_dir / "crawl-log.json").write_text(
        json.dumps({"fetched": fetched, "skipped": skipped}), encoding="utf-8")

    ledger = {
        e["url"]: {"version": 1, "content_hash": f"sha256:{i}",
                   "etag": "", "last_modified": "",
                   "first_seen_at": "2026-09-15T12:22:44+00:00",
                   "fetched_at": "2026-09-15T12:22:44+00:00",
                   "checked_at": "2026-09-15T12:23:04+00:00"}
        # The datasheet is deliberately absent, so the defaults are exercised.
        for i, e in enumerate(fetched) if not e["path"].endswith("Fine-Stuff-TDS.pdf")
    }
    (cache_dir / "versions.json").write_text(json.dumps(ledger), encoding="utf-8")

    index_dir = tmp_path / "index"
    embeddings = tmp_path / "embeddings.db"

    monkeypatch.setattr(index, "CACHE", cache_dir)
    monkeypatch.setattr(index, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index, "EmbeddingCache", lambda: EmbeddingCache(embeddings))
    monkeypatch.setattr(index.ollama, "require", lambda *models: None)

    def fake_embed(texts, progress=None):
        if progress:
            progress(len(texts), len(texts))
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(index.ollama, "embed", fake_embed)
    return index_dir


@needs_cache
def test_the_whole_ingestion_path_publishes_a_queryable_snapshot(staged, tmp_path):
    """The build is the only thing that writes the index; if it half-works the assistant answers from half a corpus."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        report = build_once(repo)

        # Five real documents: the sixth entry's cached file is gone, so it is
        # reported and not indexed.
        assert report["public_documents"] == 5
        assert report["evaluation_fixtures"] == 1
        assert report["documents"] == 6
        assert report["chunks"] > 30
        assert report["caveats"] > 0
        assert report["excluded"] == 2

        snapshot = repo.snapshot()
        assert snapshot.embedding_model == index.ollama.EMBED_MODEL
        assert snapshot.chunking_version == index.CHUNKING_VERSION
        assert snapshot.chunk_count == report["chunks"]
    finally:
        repo.close()


@needs_cache
def test_a_cached_file_that_has_gone_missing_is_reported_not_fatal(staged, tmp_path):
    """One vanished document must not end a ninety-four document build, and must not vanish silently either."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        report = build_once(repo)
        failed = [r for r in report["rows"] if r["quality"] == "failed"]
        assert [r["note"] for r in failed] == ["cached file missing"]
        assert failed[0]["chunks"] == 0
    finally:
        repo.close()


@needs_cache
def test_the_name_lists_the_real_names_check_depends_on_are_built(staged, tmp_path):
    """Check 5 refuses an invented colour, product or merchant, and it has only these lists to work from."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        report = build_once(repo)
        assert len(report["colours"]) == 24
        assert len(report["merchants"]) == 39
        assert "Solo Onecoat Lime Plaster" in report["products"]
        assert report["contact"]["phone"] == "0800 538 5746"
    finally:
        repo.close()


@needs_cache
def test_a_duplicate_colour_is_listed_once(staged, tmp_path):
    """The swatch block repeats seven colours; listed twice they read as fourteen different finishes."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        colours = build_once(repo)["colours"]
        assert len(colours) == len({c.lower() for c in colours})
    finally:
        repo.close()


@needs_cache
def test_the_staff_fixture_is_indexed_as_staff_and_not_as_public(staged, tmp_path):
    """The audience filter has nothing real to act on, so without this fixture it is an untested claim."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        build_once(repo)
        public = {d.canonical_url for d in repo.manifest(("public",))}
        staff = {d.canonical_url for d in repo.manifest(("staff",))}
        fixture = "fixture://staff/solo-margin-note"
        assert fixture in staff
        assert fixture not in public
    finally:
        repo.close()


@needs_cache
def test_the_document_name_falls_back_to_the_sheet_when_the_crawl_gave_none(staged, tmp_path):
    """A datasheet with no title on the entry would otherwise be cited as a blank name."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        build_once(repo)
        doc = repo.document("https://www.lime-green.co.uk/documents/Fine-Stuff-TDS.pdf")
        assert doc is not None
        assert doc.title
        assert doc.citation_name == "Data Sheet"
    finally:
        repo.close()


@needs_cache
def test_the_ingestion_report_is_written_whether_or_not_the_run_went_well(staged, tmp_path):
    """The report is the evidence that a document extracted badly; unwritten, the claim is unfalsifiable."""
    repo = EmbeddedRepository(tmp_path / "knowledge.db")
    try:
        build_once(repo)
        written = json.loads((staged / "ingestion-report.json").read_text(encoding="utf-8"))
        assert written["chunking_version"] == index.CHUNKING_VERSION
        assert any(r["quality"] == "failed" for r in written["rows"])
    finally:
        repo.close()


@needs_cache
def test_an_unchanged_passage_is_not_embedded_a_second_time(staged, tmp_path):
    """Embedding 601 passages takes fifteen minutes on this hardware; a rebuild that repeats it is the cost decision 17 exists for."""
    first = EmbeddedRepository(tmp_path / "first.db")
    try:
        # Verbose, because the progress counter is what an assessor watches for
        # forty minutes and it must not divide by zero on an empty batch.
        one = build_once(first, verbose=True)
    finally:
        first.close()
    assert one["embeddings_computed"] > 0
    assert one["embeddings_cached"] == 0

    second = EmbeddedRepository(tmp_path / "second.db")
    try:
        two = build_once(second)
    finally:
        second.close()
    assert two["embeddings_computed"] == 0
    assert two["embeddings_cached"] == one["embeddings_computed"]


@needs_cache
def test_the_build_opens_its_own_repository_when_it_is_not_given_one(staged):
    """The command line calls build with no repository, and that path has to close what it opened."""
    report = index.build(verbose=False)
    assert report["table_counts"]["chunks"] == report["chunks"]
    assert (staged / "knowledge.db").exists()


@needs_cache
def test_the_build_works_against_any_repository_not_only_the_sqlite_one(staged):
    """`counts()` is not part of the KnowledgeRepository contract, so a conforming adapter need not have it."""
    class RecordingRepository:
        """The indexing half of the boundary, and nothing else.

        Deliberately not a SQLite repository: the build must work against
        anything implementing the Protocol, which is what makes the PostgreSQL
        adapter a drop-in rather than a rewrite. It has no `counts()` either, so
        the report has to cope with a store that cannot introspect itself.
        """

        def __init__(self):
            self.applied = None
            self._snapshot = None

        def active_content_hashes(self):
            return {}                      # an empty store: everything is new

        def apply_delta(self, updates, removed, snapshot,
                        excluded=None, crawl_run=None):
            self.applied = (updates, removed, snapshot, crawl_run)
            self._snapshot = replace(
                snapshot,
                document_count=len(updates),
                chunk_count=sum(len(u.chunks) for u in updates))
            return snapshot.snapshot_id

        def snapshot(self):
            return self._snapshot

    repo = RecordingRepository()
    report = index.build(repo=repo, verbose=False)

    assert report["table_counts"] == {}
    updates, removed, snapshot, crawl_run = repo.applied
    assert removed == []
    assert len(updates) == report["documents"]
    assert snapshot.embedding_dimensions == index.ollama.EMBED_DIMENSIONS
    chunks = [c for u in updates for c in u.chunks]
    assert all(len(c.embedding) == index.ollama.EMBED_DIMENSIONS for c in chunks)
    # The crawl run counts crawled documents; `updates` additionally carries the
    # evaluation fixtures, which are indexed but never crawled.
    assert crawl_run is not None
    assert crawl_run.documents_checked == crawl_run.documents_new + crawl_run.documents_failed
    assert crawl_run.documents_new <= len(updates)


# ---------------------------------------------------------------------- main


def test_the_command_reports_a_missing_model_instead_of_a_traceback(monkeypatch, capsys):
    """The most likely reader of this error is an assessor running the project for the first time."""
    def refuse(**_kwargs):
        raise OllamaUnavailable("Ollama is not answering. Run `ollama serve`.")

    monkeypatch.setattr(index, "build", refuse)
    assert index.main([]) == 1
    assert "ollama serve" in capsys.readouterr().err


def test_the_command_prints_the_summary_on_success(monkeypatch, capsys):
    """The summary is what the transcript quotes; a silent success leaves the build unevidenced."""
    monkeypatch.setattr(index, "build", lambda **_kwargs: base_report())
    assert index.main([]) == 0
    assert "snap-20260915T120000Z" in capsys.readouterr().out

def test_the_rebuild_flag_reaches_the_build(monkeypatch, capsys):
    """A chunking change moves no content hash, so --rebuild is the only way to reindex."""
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return base_report()

    monkeypatch.setattr(index, "build", record)
    assert index.main(["--rebuild"]) == 0
    assert seen["rebuild"] is True

    seen.clear()
    assert index.main([]) == 0
    assert seen["rebuild"] is False
