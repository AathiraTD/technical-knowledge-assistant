"""Extraction, tested against the failures that would poison the index.

Extraction is upstream of everything. A figure rewritten here prints wrong
downstream and no post-generation check can tell, because the check compares
the answer against the passage and both would be wrong together. A heading
missed here turns a datasheet into one flat blob and a thickness gets cut away
from the substrate it applies to. A "Related products" block kept here puts
Duro and Warmshell into every passage retrieved about Solo.

So these tests run against the real shipped cache wherever a real file shows
the behaviour: the two datasheet families, the FAQ, the contact page, the
supplier map, a technical note. Synthetic HTML and synthetic PDFs appear only
for the edges the crawled corpus happens not to contain, which is honest about
which behaviours are evidenced by real data and which are evidenced by design.

Each test names the failure it stands for rather than the function it calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.extract import (                                # noqa: E402
    Section,
    _bold_sections,
    _faq_sections,
    _html_sections,
    _is_break,
    _is_heading_text,
    _pdf_lines,
    _repeated_lines,
    _soup,
    caveats,
    clean,
    contact_details,
    extract_html,
    extract_pdf,
    harvest,
)

PAGES = ROOT / "data" / "cache" / "pages"
DOCS = ROOT / "data" / "cache" / "documents"

SOLO_PAGE = PAGES / "products-lime-plaster-solo-onecoat-plaster.html"
ASHLAR_PAGE = PAGES / "products-lime-mortar-ashlar-mortar.html"
FAQ_PAGE = PAGES / "support-faq.html"
CONTACT_PAGE = PAGES / "contact.html"
SUPPLIER_PAGE = PAGES / "find-a-supplier.html"
TECHNICAL_NOTE = PAGES / "support-knowledgebase-background-preparation-for-rendering.html"
GLOSSARY_PAGE = PAGES / "support-knowledgebase-glossary-of-terms.html"

FINE_STUFF = DOCS / "Fine-Stuff-TDS.pdf"          # font-marked family
MEDIUM_MORTAR = DOCS / "medium-mortar-tds.pdf"    # layout-marked family
SILGUARD = DOCS / "silgaurd-datasheet-oct-25.pdf"  # layout signal only
SOLO_TDS = DOCS / "solo-one-coat-lime-plaster-tds.pdf"
IWI_GUIDE = DOCS / "Installation-Guide---IWI.pdf"  # 24 pages, running footer

pytestmark = pytest.mark.skipif(
    not SOLO_PAGE.exists(),
    reason="the shipped cache is missing; extraction has nothing honest to run against",
)


# --------------------------------------------------------------- test helpers


def soup_of(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def main_of(html: str):
    s = soup_of(html)
    return s.find("main") or s.find("body") or s


def write_page(tmp_path: Path, name: str, html: str) -> str:
    path = tmp_path / name
    path.write_text(html, encoding="utf-8")
    return str(path)


def write_pdf(tmp_path: Path, name: str, lines, pages: int = 1) -> str:
    """A PDF built line by line, so a detector can be shown what it does and does not see.

    `lines` is a list of (text, size, bold) per page when `pages` is 1, or a
    list of such lists when more than one page is wanted.
    """
    path = tmp_path / name
    doc = pymupdf.open()
    per_page = lines if pages > 1 else [lines]
    for page_lines in per_page:
        page = doc.new_page()
        y = 60.0
        for text, size, bold in page_lines:
            page.insert_text((50, y), text, fontsize=size,
                             fontname="hebo" if bold else "helv")
            y += size + 6
    doc.save(str(path))
    doc.close()
    return str(path)


def headings(extracted) -> list[str]:
    return [s.heading for s in extracted.sections]


# ------------------------------------------------------------------- clean()


def test_a_published_figure_is_not_rewritten_by_normalisation():
    """Every printed number must survive extraction word for word, or check 2 compares two wrong things."""
    raw = "Add\xa05–6\xa0litres per 25\xa0kg sack at 8°C, 2 to 4N/mm2."
    assert clean(raw) == "Add 5–6 litres per 25 kg sack at 8°C, 2 to 4N/mm2."


def test_a_zero_width_space_does_not_split_a_figure():
    """An invisible character inside '5-6' would make the figure unfindable in its own passage."""
    assert clean("5​-​6 litres") == "5-6 litres"


def test_runs_of_blank_lines_collapse_without_joining_two_sections():
    """Paragraph boundaries are what a bullet-less section is split on, so they must not be erased."""
    assert clean("Mixing\n\n\n\nAdd water") == "Mixing\n\nAdd water"
    assert clean("  spaced\t\tout  ") == "spaced out"


# ---------------------------------------------------------- heading shape test


@pytest.mark.parametrize("text, is_heading", [
    ("Mixing", True),
    ("Colours:", True),                      # a plural label still reads as a heading
    ("Mixing:", False),                      # a single trailing colon is a label, not a heading
    ("Preparation & Application", True),
    ("ab", False),                           # too short to be anything
    ("x" * 61, False),                       # too long to be a heading
    ("mixing", False),                       # headings on these sheets start capitalised
    ("Add between five and six litres of clean water per sack", False),  # a sentence
    ("Apply the coat.", False),              # ends like a sentence
    ("12345", False),                        # a page number carries no letters
])
def test_the_shape_test_separates_a_heading_from_a_sentence(text, is_heading):
    """Both detectors run this first; a loose shape test turns body sentences into section breaks."""
    assert _is_heading_text(text) is is_heading


# ---------------------------------------------------------------- PDF families


def test_a_font_marked_sheet_yields_its_named_sections():
    """Fine Stuff marks headings with a heavier font; losing them loses the 8 degree limit's section."""
    got = extract_pdf(str(FINE_STUFF))
    assert got.quality == "clean"
    assert got.detector == "both"
    assert {"Mixing", "Application", "Curing", "Coverage"} <= set(headings(got))


def test_a_layout_marked_sheet_is_not_reported_as_flat():
    """The newer sheets use one font throughout; a font-only detector returned two 3,000-character blobs."""
    got = extract_pdf(str(MEDIUM_MORTAR))
    assert got.quality == "clean"
    assert {"Description", "Mixing", "Application", "Aftercare"} <= set(headings(got))


def test_the_one_sheet_with_no_font_signal_at_all_still_splits():
    """Silguard carries no font signal whatsoever, so it is the proof that the layout detector is load-bearing."""
    got = extract_pdf(str(SILGUARD))
    assert got.detector == "layout"
    assert got.quality == "clean"
    assert {"Description", "Application", "Aftercare"} <= set(headings(got))


def test_a_sheet_with_no_layout_signal_still_splits(tmp_path):
    """The mirror case: short body lines give the layout detector nothing, and the font detector must carry it."""
    lines = [("Mixing", 11, True)]
    lines += [(f"add water slowly to the mix number {i}.", 10, False) for i in range(12)]
    lines += [("Aftercare", 11, True)]
    lines += [(f"keep the wall damp on day {i}.", 10, False) for i in range(12)]
    got = extract_pdf(write_pdf(tmp_path, "font-only.pdf", lines))
    assert got.detector == "font"
    assert headings(got) == ["Mixing", "Aftercare"]


def test_a_sheet_with_no_headings_is_chunked_by_page_and_says_so(tmp_path):
    """A document with no structure must be labelled flat, not silently indexed as though it had sections."""
    lines = [(f"this line is deliberately long enough to fail every heading shape test {i}.",
              10, False) for i in range(8)]
    got = extract_pdf(write_pdf(tmp_path, "flat.pdf", lines))
    assert got.detector == "none"
    assert got.quality == "flat"
    assert headings(got) == ["Page 1"]
    assert "no headings detected" in got.note


def test_headings_nobody_recognises_still_score_the_document_as_flat(tmp_path):
    """Quality is what the ingestion report is read for; unrecognised headings must not be reported as clean."""
    lines = []
    for name in ("Zephyr", "Quokka", "Tumbrel"):
        lines.append((name, 11, True))
        lines += [(f"{name.lower()} body line number {i} written out at length.", 10, False)
                  for i in range(6)]
    got = extract_pdf(write_pdf(tmp_path, "unknown-headings.pdf", lines))
    assert headings(got) == ["Zephyr", "Quokka", "Tumbrel"]
    assert got.quality == "flat"


def test_a_document_with_one_recognised_heading_is_partial(tmp_path):
    """Partial is the honest middle: some structure found, not enough to trust the split."""
    lines = [("Mixing", 11, True)]
    lines += [(f"add water slowly to the mix number {i}.", 10, False) for i in range(12)]
    lines += [("Zephyr", 11, True)]
    lines += [(f"zephyr body line number {i} written out at length.", 10, False)
              for i in range(6)]
    got = extract_pdf(write_pdf(tmp_path, "partial.pdf", lines))
    assert got.quality == "partial"


def test_a_blank_page_does_not_become_an_empty_passage(tmp_path):
    """An empty chunk retrieves on nothing and prints nothing, so it must never be written."""
    body = [(f"this line is deliberately long enough to fail every heading shape test {i}.",
             10, False) for i in range(8)]
    got = extract_pdf(write_pdf(tmp_path, "flat-blank.pdf", [body, []], pages=2))
    assert got.quality == "flat"
    assert headings(got) == ["Page 1"]


def test_a_sheet_that_ends_on_a_heading_does_not_publish_an_empty_section(tmp_path):
    """A trailing heading with nothing under it is a citation that leads to a blank passage."""
    lines = [("abc", 10, False)] * 11
    lines += [("Mixing", 11, True)]
    lines += [(f"add water slowly to the mix number {i}.", 10, False) for i in range(12)]
    lines += [("Aftercare", 11, True)]
    got = extract_pdf(write_pdf(tmp_path, "ends-on-heading.pdf", lines))
    assert "Aftercare" not in headings(got)
    assert all(s.text.strip() for s in got.sections)


def test_a_sheet_whose_opening_lines_are_too_short_gets_no_invented_title(tmp_path):
    """The title becomes a section name when nothing else is available; guessing one misnames the passage."""
    lines = [("abc", 10, False)] * 11
    lines += [("Mixing", 11, True)]
    lines += [(f"add water slowly to the mix number {i}.", 10, False) for i in range(12)]
    got = extract_pdf(write_pdf(tmp_path, "no-title.pdf", lines))
    assert got.title == ""


def test_a_missing_file_is_reported_rather_than_raised(tmp_path):
    """One unreadable document must not end a ninety-four document build."""
    got = extract_pdf(str(tmp_path / "does-not-exist.pdf"))
    assert got.quality == "failed"
    assert "could not open" in got.note
    assert got.sections == []


def test_a_pdf_with_no_text_layer_is_reported_as_failed(tmp_path):
    """A scanned sheet has no extractable text; indexing it silently would publish an empty document."""
    got = extract_pdf(write_pdf(tmp_path, "scan.pdf", [("short", 10, False)]))
    assert got.quality == "failed"
    assert "no usable text layer" in got.note


def test_the_printed_date_is_read_off_the_sheet():
    """Newest wins within a document type, and a sheet with no parsed date cannot take part in that."""
    assert extract_pdf(str(MEDIUM_MORTAR)).printed_date == "9/12/24"
    assert extract_pdf(str(FINE_STUFF)).printed_date == "7/6/19"


def test_a_sheet_that_prints_no_date_returns_an_empty_one():
    """An invented date is worse than a missing one, so the field stays empty rather than guessing."""
    assert extract_pdf(str(SOLO_TDS)).printed_date == ""


def test_a_running_footer_never_becomes_a_passage():
    """The IWI guide prints 'lime|green' on 23 of its 24 pages; indexed, it is 23 near-identical chunks."""
    doc = pymupdf.open(str(IWI_GUIDE))
    pages_with_footer = sum(1 for page in doc if "lime|green" in clean(page.get_text()))
    assert pages_with_footer >= 3

    got = extract_pdf(str(IWI_GUIDE))
    assert not any("lime|green" in s.text for s in got.sections)
    assert not any("lime|green" in s.heading for s in got.sections)


def test_a_two_page_sheet_has_no_running_furniture_to_strip():
    """On a short sheet a repeated line is content, and stripping it would delete a real passage."""
    doc = pymupdf.open(str(FINE_STUFF))
    assert _repeated_lines(_pdf_lines(doc), 2) == set()


# ----------------------------------------------------------------- harvesting


def test_twenty_four_colours_are_harvested_from_a_product_page():
    """Check 5 refuses an invented colour, and it can only do that against a list read off the site."""
    colours = harvest(_soup(str(SOLO_PAGE)))["colours"]
    assert len({c.lower() for c in colours}) == 24
    assert "York" in colours and "Cotswold" in colours


def test_a_sample_or_brochure_order_is_not_offered_as_a_render_colour():
    """'Colour brochure' sits under the same URL prefix as the colours and is not one."""
    colours = harvest(_soup(str(SOLO_PAGE)))["colours"]
    assert "Colour brochure" not in colours
    assert "Solo sample" not in colours
    assert "Warmshell Internal" not in colours


def test_the_navigation_tail_is_stripped_from_the_last_colour():
    """The site renders the final swatch as 'Chalk More products >'; indexed whole it is not a colour name."""
    colours = harvest(_soup(str(ASHLAR_PAGE)))["colours"]
    assert "Chalk" in colours
    assert not any("More products" in c for c in colours)


def test_thirty_nine_merchants_are_read_from_the_map_pin_alt_text():
    """The stockists are published as image alt text, so rendered-text extraction reported none at all."""
    merchants = harvest(_soup(str(SUPPLIER_PAGE)))["merchants"]
    assert len(merchants) == 39
    assert "The Lime Centre" in merchants


def test_product_links_are_harvested_for_the_real_names_check():
    """Check 5 also refuses an invented product, which needs the product list the site publishes."""
    products = harvest(_soup(str(SOLO_PAGE)))["products"]
    assert "Solo Onecoat Lime Plaster" in products
    assert "Duro Lime Plaster Base Coat" in products


def test_a_sentence_length_link_label_is_not_taken_as_a_name():
    """A call-to-action link under a colour URL would otherwise become a colour the assistant may name."""
    html = ('<a href="/products-by-colour/york">York</a>'
            f'<a href="/products-by-colour/long">{"X" * 70}</a>'
            '<a href="/products-by-colour/blank"></a>')
    assert harvest(soup_of(html))["colours"] == ["York"]


def test_a_map_pin_with_no_usable_alt_text_is_skipped():
    """An empty or sentence-length alt attribute is not a merchant, and check 5 must not learn it as one."""
    html = ('<img class="map-pin" alt="Womersley\'s">'
            '<img class="map-pin">'
            f'<img class="map-pin" alt="{"Y" * 80}">')
    assert harvest(soup_of(html))["merchants"] == ["Womersley's"]


# ------------------------------------------------------------ contact details


def test_the_only_phone_number_is_the_one_printed_on_the_contact_page():
    """A phone number an assistant invents is close to the worst thing it can print."""
    details = contact_details(str(CONTACT_PAGE))
    assert details["phone"] == "0800 538 5746"
    assert details["hours"].startswith("Mon - Fri")
    assert "9:00am" in details["hours"]
    assert "Lime Green Products Ltd" in details["address"]


def test_a_page_with_no_contact_details_returns_empty_strings(tmp_path):
    """A hand-off with a blank number is visibly broken; a hand-off with a guessed one is not."""
    path = write_page(tmp_path, "nocontact.html",
                      "<html><body><main><p>Nothing useful here at all.</p></main></body></html>")
    details = contact_details(path)
    assert details["phone"] == ""
    assert details["hours"] == ""


def test_opening_hours_with_no_times_after_them_do_not_invent_times(tmp_path):
    """'Mon - Fri' on its own is what the page said; appending a plausible time would be inventing it."""
    path = write_page(tmp_path, "hours.html",
                      "<html><body><main><p>Office Hours: Mon - Fri</p></main></body></html>")
    assert contact_details(path)["hours"] == "Mon - Fri"


# ------------------------------------------------------------------- the FAQ


def test_the_faq_yields_thirty_two_question_and_answer_sections():
    """The FAQ is 32 citable answers, not one page of prose; as one page none of them retrieve."""
    got, _ = extract_html(str(FAQ_PAGE), "faq")
    assert len(got.sections) == 32
    assert got.quality == "clean"
    assert got.detector == "dom"


def test_a_faq_answer_is_labelled_with_the_section_it_sits_under():
    """'Renders: how long before painting?' is a better citation than the question alone."""
    got, _ = extract_html(str(FAQ_PAGE), "faq")
    assert all(s.text for s in got.sections)
    assert any("—" in s.heading for s in got.sections)


def test_the_faq_page_without_its_doc_type_does_not_split_into_questions():
    """The question split is chosen by the crawl's classification, not guessed from the markup."""
    got, _ = extract_html(str(FAQ_PAGE))
    assert len(got.sections) < 32


def test_a_question_with_no_answer_beneath_it_is_dropped():
    """An empty answer indexed as a passage retrieves on the question and prints nothing."""
    html = ('<main><section class="faq-section"><p class="h2-style">Mortars</p>'
            '<dl><dt>Question with an answer?</dt><dd>Yes it does.</dd>'
            '<dt>Question with no answer?</dt></dl></section></main>')
    sections = _faq_sections(main_of(html))
    assert [s.heading for s in sections] == ["Mortars — Question with an answer?"]


def test_an_uncategorised_question_keeps_its_own_text_as_the_heading():
    """A missing category must not produce a heading that starts with a dangling dash."""
    html = ('<main><section class="faq-section"><dl>'
            '<dt>Uncategorised question?</dt><dd>An answer with no category above it.</dd>'
            '</dl></section></main>')
    sections = _faq_sections(main_of(html))
    assert [s.heading for s in sections] == ["Uncategorised question?"]


def test_a_page_classed_as_faq_that_has_no_questions_falls_back(tmp_path):
    """Classification can be wrong; when it is, the page must still extract rather than return nothing."""
    path = write_page(tmp_path, "notreallyfaq.html", (
        "<html><head><title>FAQ</title></head><body><main>"
        "<h2>General</h2><p>An answer paragraph long enough to be kept as a section body.</p>"
        "</main></body></html>"))
    got, _ = extract_html(path, "faq")
    assert "General" in headings(got)


# ---------------------------------------------------------- HTML page content


def test_related_products_never_reach_a_passage_about_solo():
    """The Solo page names Duro and Warmshell in its footer; indexed, a Solo question retrieves aerogel."""
    got, _ = extract_html(str(SOLO_PAGE), "product_page")
    body = " ".join(s.heading + " " + s.text for s in got.sections)
    assert "Related products" not in body
    assert "Duro" not in body
    assert "Warmshell" not in body


def test_the_download_block_does_not_publish_a_false_deferral():
    """'If you require any other information please contact us' fired router step 2 on every product page."""
    raw = SOLO_PAGE.read_text(encoding="utf-8", errors="replace").lower()
    assert "contact us" in raw          # the furniture really is on the page

    got, _ = extract_html(str(SOLO_PAGE), "product_page")
    body = " ".join(s.text for s in got.sections).lower()
    assert "contact us" not in body
    assert "click on a button to download" not in body


def test_download_furniture_inside_a_kept_section_is_removed(tmp_path):
    """The lines survive under headings that are not themselves furniture, so the text is filtered too."""
    path = write_page(tmp_path, "furniture.html", (
        "<html><body><main><h2>Mixing</h2><p>"
        "Add five to six litres of clean water per twenty five kilogram sack.\n"
        "Click on a button to download the data sheet\n"
        "View technical information about this product here\n"
        "SDS\nFind a supplier</p></main></body></html>"))
    got, _ = extract_html(path)
    mixing = next(s for s in got.sections if s.heading == "Mixing")
    assert mixing.text == (
        "Add five to six litres of clean water per twenty five kilogram sack.")


def test_collection_stops_at_the_first_tail_marker():
    """Everything below 'Related products' is furniture that names other products."""
    html = ("<main><h2>Related products</h2><p>Duro and Warmshell live here and must "
            "never be indexed as Solo content at all.</p></main>")
    assert _html_sections(main_of(html)) == []


def test_a_two_line_block_under_a_heading_is_not_a_section():
    """A heading with a stub under it retrieves on the heading and answers nothing."""
    html = "<main><h2>Short</h2><p>too short</p></main>"
    assert _html_sections(main_of(html)) == []


def test_a_download_heading_is_not_indexed_as_a_section():
    """'Downloads' is page furniture; kept, it carries the site's stock 'contact us' line into the index."""
    html = ("<main><h2>Downloads</h2><p>Click on a button to download the data sheet "
            "for this product now.</p>"
            "<h2></h2><p>a block with no heading at all, long enough to be kept</p></main>")
    assert _html_sections(main_of(html)) == []


def test_a_bare_text_node_under_a_heading_is_still_that_section_body():
    """The site emits loose text between tags; dropping it loses real published sentences."""
    html = ("<main><h2>Mixing</h2>Loose text sitting directly under the heading, "
            "long enough to keep as a section body.<h2>Aftercare</h2></main>")
    sections = _html_sections(main_of(html))
    assert [s.heading for s in sections] == ["Mixing"]
    assert "Loose text" in sections[0].text


def test_a_paragraph_styled_as_a_heading_opens_a_section():
    """The site marks some headings with a styled paragraph, not a heading tag."""
    html = ('<main><p class="lead-style">Pseudo heading</p>'
            "<p>A paragraph styled as a heading opens a section, which is how the site "
            "marks them.</p></main>")
    assert [s.heading for s in _html_sections(main_of(html))] == ["Pseudo heading"]


@pytest.mark.parametrize("html, breaks", [
    ("<h2>Mixing</h2>", True),
    ("<h5>Mixing</h5>", True),
    ('<p class="h2-style">Mixing</p>', True),
    ('<p class="body">Mixing</p>', False),
    ("<p>Mixing</p>", False),
    ("<div>Mixing</div>", False),
])
def test_only_real_breaks_end_the_section_above_them(html, breaks):
    """Treating an ordinary paragraph as a break shatters a section into uncitable fragments."""
    node = soup_of(html).find(True, recursive=True)
    node = node if node.name != "html" else node.find(True)
    while node is not None and node.name in ("html", "body"):
        node = node.find(True)
    assert _is_break(node) is breaks


# ------------------------------------------------------------- bold sections


def test_a_technical_note_marked_with_bold_is_not_one_block():
    """Background preparation has sixteen numbered sections and no heading tags at all."""
    got, _ = extract_html(str(TECHNICAL_NOTE), "knowledge_base")
    assert len(got.sections) > 10
    assert any(h.startswith("1) Construction Issues") for h in headings(got))


def test_the_published_date_is_read_off_the_article():
    """The site prints '4 th April 2017'; left as published, newest-wins cannot compare two articles."""
    got, _ = extract_html(str(TECHNICAL_NOTE), "knowledge_base")
    assert got.printed_date == "4 April 2017"


def test_a_bold_run_that_is_the_whole_paragraph_still_opens_a_section():
    """Some notes put the heading in its own paragraph; treated as body it becomes a one-word passage."""
    html = ("<main>"
            "<p><strong>Curing</strong></p>"
            "<p>Keep the wall damp for five days and protect it from direct sunlight.</p>"
            "</main>")
    assert [s.heading for s in _bold_sections(main_of(html))] == ["Curing"]


def test_a_bold_lead_inside_a_paragraph_keeps_the_rest_of_that_paragraph():
    """'Mixing: add 5-6 litres' loses its figure if only the lead is kept."""
    html = ("<main>"
            "<p><strong>Mixing:</strong> add between five and six litres of clean water.</p>"
            "<p>More mixing detail that follows the lead under the same heading here.</p>"
            "</main>")
    sections = _bold_sections(main_of(html))
    assert [s.heading for s in sections] == ["Mixing:"]
    assert "five and six litres" in sections[0].text


def test_a_bold_section_with_almost_no_body_is_dropped():
    """A heading with three words under it is not a passage anyone can be answered from."""
    html = "<main><p><strong>Tiny</strong> short</p></main>"
    assert _bold_sections(main_of(html)) == []


def test_a_bold_heading_whose_body_is_a_stub_is_not_kept():
    """Two bold leads in a row leave the first with three words, which answers nothing."""
    html = ("<main>"
            "<p><strong>Curing:</strong> short.</p>"
            "<p><strong>Mixing:</strong> add between five and six litres of clean water.</p>"
            "</main>")
    assert [s.heading for s in _bold_sections(main_of(html))] == ["Mixing:"]


def test_a_whole_paragraph_bold_lead_closes_the_section_before_it():
    """Without the flush, Mixing's figure lands under the Curing heading and check 4 cannot save it."""
    html = ("<main>"
            "<p><strong>Mixing:</strong> add between five and six litres of clean water.</p>"
            "<p><strong>Curing</strong></p>"
            "<p>Keep the wall damp for five days and protect it from direct sunlight.</p>"
            "</main>")
    sections = _bold_sections(main_of(html))
    assert [s.heading for s in sections] == ["Mixing:", "Curing"]
    assert "five and six litres" in sections[0].text
    assert "damp for five days" in sections[1].text


def test_a_stub_section_is_dropped_rather_than_merged_into_the_next_heading():
    """Merging it would put Mixing's words under Curing, which is exactly the attribution check 3 refuses."""
    html = ("<main>"
            "<p><strong>Mixing:</strong> short.</p>"
            "<p><strong>Curing</strong></p>"
            "<p>Keep the wall damp for five days and protect it from direct sunlight.</p>"
            "</main>")
    sections = _bold_sections(main_of(html))
    assert [s.heading for s in sections] == ["Curing"]
    assert "short." not in sections[0].text


def test_prose_before_the_first_bold_heading_is_not_promoted():
    """An article's opening paragraph belongs to the lead section, not to the first named one."""
    html = ("<main>"
            "<p>Opening paragraph with no bold lead at all, buffered until a heading appears.</p>"
            "<p><strong>Curing</strong></p>"
            "<p>Keep the wall damp for five days and protect it from direct sunlight.</p>"
            "</main>")
    sections = _bold_sections(main_of(html))
    assert [s.heading for s in sections] == ["Curing"]
    assert "Opening paragraph" not in sections[0].text


# -------------------------------------------------------------- extract_html


def test_a_page_that_already_splits_cleanly_is_not_re_split_by_bold_runs():
    """The bold fallback exists for pages with no heading tags; applied to a good split it fragments it."""
    got, _ = extract_html(str(PAGES / "support-knowledgebase-render-checklist.html"),
                          "knowledge_base")
    assert [s.heading for s in got.sections] == [
        "Design", "Planning and Sequencing", "Preparing the wall",
        "Application", "Inspection",
    ]


def test_a_page_with_no_headings_still_yields_its_lead_paragraphs():
    """The glossary has no structure at all, and dropping it would lose the vocabulary page entirely."""
    got, _ = extract_html(str(GLOSSARY_PAGE), "knowledge_base")
    assert got.sections
    assert got.chars > 300


def test_a_product_page_keeps_the_intro_that_says_what_the_product_is_for():
    """Selection questions are answered from the intro, which sits above the first heading."""
    got, _ = extract_html(str(SOLO_PAGE), "product_page")
    assert got.sections[0].heading == "Solo Onecoat Lime Plaster"
    assert "one-coat lime plaster" in got.sections[0].text


def test_a_page_with_no_title_tag_still_names_its_lead_section(tmp_path):
    """A blank heading is not citable; the fallback name keeps the passage referenceable."""
    path = write_page(tmp_path, "notitle.html", (
        "<html><body><p>Only a body, no main and no title tag anywhere on this page "
        "at all, and it runs long enough to be kept.</p></body></html>"))
    got, _ = extract_html(path)
    assert got.title == ""
    assert headings(got) == ["Description"]


def test_a_page_with_nothing_worth_indexing_is_reported_as_flat(tmp_path):
    """An empty extraction must be visible in the ingestion report rather than inferred from a bad answer."""
    path = write_page(tmp_path, "empty.html", (
        "<html><head><title>Nothing | Lime Green</title></head>"
        "<body><main><p>tiny</p></main></body></html>"))
    got, _ = extract_html(path)
    assert got.sections == []
    assert got.quality == "flat"
    assert got.title == "Nothing | Lime Green"


# ---------------------------------------------------------------- caveats


def test_a_temperature_limit_is_tagged_against_the_document():
    """Fine Stuff states its 8 degree limit under Mixing while the user asks about Application."""
    got = extract_pdf(str(FINE_STUFF))
    tagged = caveats(got.sections)
    temperature = [c for c in tagged if c["type"] == "temperature"]
    assert temperature
    assert any("8" in c["sentence"] for c in temperature)


def test_a_diy_warning_is_tagged():
    """'It is not suitable for DIY plastering' must travel with every Fine Stuff passage printed."""
    tagged = caveats(extract_pdf(str(FINE_STUFF)).sections)
    assert any(c["type"] == "diy" for c in tagged)


def test_an_incompatible_substrate_warning_is_tagged():
    """'Many MgO boards are not suitable for Solo' is the caveat with the largest cost attached."""
    tagged = caveats(extract_pdf(str(SOLO_TDS)).sections)
    assert any(c["type"] == "incompatibility" and "MgO" in c["sentence"] for c in tagged)


def test_a_caveat_records_the_section_it_came_from():
    """A caveat printed without its section cannot be checked against the sheet by a plasterer."""
    tagged = caveats(extract_pdf(str(FINE_STUFF)).sections)
    assert all(c["section"] for c in tagged)


def test_a_deferral_is_not_also_tagged_as_a_caveat():
    """A 'contact us' sentence has its own router path; tagged here it prints twice in one answer."""
    sections = [Section("Preparation", (
        "Do not use on gypsum plasterboard; please contact our technical team first."))]
    assert caveats(sections) == []


def test_the_real_sheet_deferral_is_absent_from_its_caveats():
    """The Solo sheet's MgO deferral is the worked example of decision 11 and must not be duplicated."""
    tagged = caveats(extract_pdf(str(SOLO_TDS)).sections)
    assert not any(c["sentence"].lower().startswith("contact us") for c in tagged)


def test_a_bullet_character_is_stripped_from_a_quoted_caveat():
    """A caveat is printed on its own line under the answer, where a stray bullet reads as a list item."""
    sections = [Section("Aftercare", "• Do not apply in frost or direct sunlight.")]
    tagged = caveats(sections)
    assert tagged[0]["sentence"] == "Do not apply in frost or direct sunlight."


def test_the_same_caveat_in_two_sections_is_tagged_once():
    """Fine Stuff repeats its temperature limit; appended twice, the answer reads as two different limits."""
    line = "Do not use below 8 C or above 25 C."
    tagged = caveats([Section("Mixing", line), Section("Curing", line)])
    assert len(tagged) == 1
    assert tagged[0]["section"] == "Mixing"


def test_a_fragment_too_short_to_stand_alone_is_not_a_caveat():
    """'Never.' appended under an answer qualifies nothing and reads as a refusal."""
    assert caveats([Section("Notes", "Never.")]) == []


def test_a_whole_paragraph_is_not_appended_as_a_caveat():
    """A caveat is a sentence; a 400-character paragraph under an answer buries the answer."""
    long_sentence = "Do not use this product " + ("in any circumstance whatsoever " * 15)
    assert caveats([Section("Notes", long_sentence)]) == []
