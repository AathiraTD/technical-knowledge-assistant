"""Extraction: cached bytes to citable sections.

Two things happen here, and the order of them is load-bearing.

**Harvesting runs before stripping.** The colour names live in the site's
navigation block, which is boilerplate on all 57 pages — so the check that
refuses an invented colour depends on reading a region that content extraction
exists to throw away. Harvest first, strip second. Decision 12 records why.

**Headings are found by two detectors, not one.** Decision 5 chose structure-
aware chunking on evidence from three probed datasheets. Probing all 37 showed
the corpus splits into two families: 17 sheets mark headings with a heavier or
larger font, and the rest use the same font throughout and mark a heading by
putting it alone on a short line above a paragraph. A font-only detector finds
nothing on the second family — it reports a flat document and chunking falls
back to whole pages. Running both detectors and taking the union recovers
Description / Mixing / Application / Aftercare on every sheet in the corpus.

Quality is recorded per document rather than assumed, so the ingestion report
can say which documents extracted badly instead of indexing them silently.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pymupdf
from bs4 import BeautifulSoup

# ---------------------------------------------------------------- vocabulary

# Section names seen across the corpus. Used to score a document's extraction
# quality, never to restrict what is kept: an unrecognised heading is still a
# heading, it just does not count towards the confidence score.
KNOWN_SECTIONS = (
    "description", "colour", "color", "coverage", "mixing", "application",
    "preparation", "background", "curing", "drying", "storage", "packaging",
    "aftercare", "finishing", "painting", "textures", "performance", "safety",
    "health", "composition", "usage", "disclaimer", "cleaning", "disposal",
    "properties", "technical", "specification", "limitations", "suitable",
    "grade", "choice", "surface", "further information",
)

# Headings inside <main> that are page furniture, not content. A product page
# carries its own "Related products" block naming three other products; indexed,
# it puts Duro and Warmshell into every passage retrieved about Solo.
HTML_BOILERPLATE = (
    "product gallery", "related case studies", "related products",
    "find your nearest supplier", "downloads", "download", "case studies",
    "product advice and expert help", "more knowledge base", "login",
    "read case study", "view all case studies",
    # Singular and plural both appear, and missing the singular was not
    # cosmetic: the "Data Sheet" block carries the line "If you require any
    # other information please contact us", which is indistinguishable from a
    # real published deferral. Indexed, it made router step 2 fire on the
    # download furniture of every product page, so "which plaster should I
    # use?" returned a quoted hand-off instead of asking which wall.
    "data sheet", "data sheets", "product data sheet",
)

# Furniture that survives inside a kept section. Matched against the text, not
# the heading, because the site repeats these lines under several headings.
HTML_FURNITURE_LINES = re.compile(
    r"^\s*(?:"
    r"click on a button to download.*"
    r"|the document will save to your downloads folder"
    r"|view technical information about this product here.*"
    r"|view all \d+ images"
    r"|downloads?"
    r"|read case study\s*>?"
    r"|find a supplier"
    r"|photographs showing examples of projects.*"
    r"|don't just take our word for it.*"
    r"|more fantastic products.*"
    r"|our products are sold nationwide.*"
    r"|(?:eu |uk )?dop|sds|epd|lrv light value|carbon footprint"
    r")\s*$",
    re.I | re.M,
)

# Everything from here down a product page is furniture, and it is furniture
# that names other products. The "Related products" block on the Solo page
# carries headings for Duro, Warmshell Internal and Solo Primer; indexed, those
# headings attach themselves to Solo's passages and a question about Solo
# retrieves a page about aerogel. Collection stops at the first of these.
HTML_TAIL_MARKERS = (
    "related case studies", "related products", "find your nearest supplier",
    "more knowledge base", "case studies", "product gallery",
)

# Sentences that qualify an instruction rather than continue it. Tagged per
# document and appended by code wherever that document is used — decision 11.
CAVEAT_PATTERNS = (
    ("temperature", re.compile(
        r"\b(?:below|above|under|over)\s*\d+\s*°?\s*[cC]\b"
        r"|\bfrost\b|\bfreez\w*\b|\bdirect sunlight\b|\bhot weather\b"
        r"|\bdo not apply\b.{0,40}\b(?:temperature|cold|frost|sun)\b", re.I)),
    ("diy", re.compile(
        r"\bnot suitable for (?:DIY|the inexperienced|amateur)\w*\b"
        r"|\bexperienced plasterer\b|\bskilled\b.{0,30}\brequired\b"
        r"|\bprofessional (?:application|applicator|plasterer)\b"
        r"|\brequires? (?:skill|experience)\b", re.I)),
    ("incompatibility", re.compile(
        r"\bnot suitable for\b|\bdo not use\b|\bmust not be\b|\bnever\b"
        r"|\bincompatible\b|\bavoid\b.{0,30}\b(?:cement|gypsum|plasterboard)\b"
        r"|\bunsuitable\b", re.I)),
)

# A sentence that tells the reader to ask a human is a deferral, not a caveat.
# It has its own path through the router, and tagging it here prints it twice.
DEFERRAL = re.compile(
    r"\b(?:contact|call|ask|consult|speak to|get in touch with)\b"
    r".{0,40}\b(?:us|our|technical|team|department|advice|support)\b"
    r"|\bplease (?:contact|call|ask|enquire)\b",
    re.I,
)

_DATE = re.compile(r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b")


# -------------------------------------------------------------------- output


@dataclass
class Section:
    """A heading and the text under it: the unit of retrieval and of citation."""

    heading: str
    text: str
    page: int = 0

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Extracted:
    """Everything one document yields, including how well it went."""

    sections: list[Section] = field(default_factory=list)
    quality: str = "unknown"        # clean | partial | flat | failed
    note: str = ""
    printed_date: str = ""
    detector: str = ""              # font | layout | both | none | dom
    title: str = ""

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


# --------------------------------------------------------------- normalising


def clean(text: str) -> str:
    """Normalise whitespace without touching the characters that carry meaning.

    Units, dashes and degree signs are left exactly as published: a figure is
    only allowed to print if it is found word-for-word in a cited passage, so
    rewriting '5-6 litres' here would break the check that depends on it.
    """
    text = text.replace("\xa0", " ").replace("​", "")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading_text(s: str) -> bool:
    """Shape test, applied by both detectors before either font or layout."""
    s = s.strip()
    if not 2 < len(s) <= 60:
        return False
    if not re.match(r"^[A-Z0-9]", s):
        return False
    if s.endswith((".", ",", ";", ":")) and not s.endswith("s:"):
        return False
    # a heading is not a sentence
    if len(s.split()) > 8:
        return False
    return bool(re.search(r"[A-Za-z]", s))


# ---------------------------------------------------------------------- PDFs


def _pdf_lines(doc) -> list[tuple[str, float, bool, int]]:
    """Text lines with the font facts the detectors need: (text, size, bold, page)."""
    out = []
    for pno, page in enumerate(doc):
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = [sp for sp in line.get("spans", []) if sp["text"].strip()]
                if not spans:
                    continue
                text = clean(" ".join(sp["text"] for sp in spans))
                if not text:
                    continue
                size = max(round(sp["size"], 1) for sp in spans)
                bold = any(sp["flags"] & 2 ** 4 or "bold" in sp["font"].lower()
                           for sp in spans)
                out.append((text, size, bold, pno))
    return out


def _repeated_lines(lines, pages: int) -> set[str]:
    """Running headers and footers: the same short line on most pages."""
    if pages < 3:
        return set()
    seen: dict[str, set[int]] = {}
    for text, _size, _bold, pno in lines:
        if len(text) <= 90:
            seen.setdefault(text, set()).add(pno)
    return {t for t, ps in seen.items() if len(ps) >= max(3, int(pages * 0.6))}


def extract_pdf(path: str) -> Extracted:
    """Sections from a PDF, using the font detector and the layout detector."""
    try:
        doc = pymupdf.open(path)
    except Exception as exc:                        # unreadable file
        return Extracted(quality="failed", note=f"could not open: {exc}")

    lines = _pdf_lines(doc)
    if sum(len(t) for t, *_ in lines) < 200:
        return Extracted(quality="failed",
                         note="no usable text layer — likely a scan")

    # Body size carries the most characters; headings stand off from it.
    weight: dict[float, int] = {}
    for text, size, _b, _p in lines:
        weight[size] = weight.get(size, 0) + len(text)
    body = max(weight, key=weight.get)

    furniture = _repeated_lines(lines, len(doc))

    # -- the two detectors --------------------------------------------------
    font_hits, layout_hits = set(), set()
    for i, (text, size, bold, _p) in enumerate(lines):
        if text in furniture or not _is_heading_text(text):
            continue
        if size > body + 0.4 or (bold and size >= body - 0.2):
            font_hits.add(i)
        nxt = lines[i + 1][0] if i + 1 < len(lines) else ""
        if len(nxt) > 55 and not _is_heading_text(nxt):
            layout_hits.add(i)

    heads = font_hits | layout_hits
    if font_hits and layout_hits:
        detector = "both"
    elif font_hits:
        detector = "font"
    elif layout_hits:
        detector = "layout"
    else:
        detector = "none"

    # -- title and printed date --------------------------------------------
    title = ""
    for text, _s, _b, _p in lines[:12]:
        if len(text) > 12 and re.search(r"[A-Za-z]", text):
            title = text
            break
    printed = ""
    for text, _s, _b, _p in lines[:25]:
        m = _DATE.search(text)
        if m:
            printed = m.group(1)
            break

    # -- assemble sections --------------------------------------------------
    sections: list[Section] = []
    current, buf, cpage = "", [], 0
    for i, (text, _s, _b, pno) in enumerate(lines):
        if text in furniture:
            continue
        if i in heads:
            if buf:
                body_text = clean("\n".join(buf))
                if body_text:
                    sections.append(Section(current or title or "Introduction",
                                            body_text, cpage))
            current, buf, cpage = text, [], pno
        else:
            buf.append(text)
    if buf:
        body_text = clean("\n".join(buf))
        if body_text:
            sections.append(Section(current or title or "Introduction", body_text, cpage))

    # Nothing to split on: fall back to one section per page and say so, rather
    # than pretending the document has structure it does not.
    if not sections or detector == "none":
        sections = []
        for pno, page in enumerate(doc):
            t = clean(page.get_text())
            if t:
                sections.append(Section(f"Page {pno + 1}", t, pno))
        return Extracted(sections, "flat", "no headings detected; chunked by page",
                         printed, "none", title)

    named = sum(1 for s in sections
                if any(k in s.heading.lower() for k in KNOWN_SECTIONS))
    quality = "clean" if named >= 3 else ("partial" if named >= 1 else "flat")
    return Extracted(sections, quality, "", printed, detector, title)


# ---------------------------------------------------------------------- HTML


def _soup(path: str) -> BeautifulSoup:
    with open(path, encoding="utf-8") as fh:
        return BeautifulSoup(fh.read(), "lxml")


# The colour block links to sample and brochure orders from the same URL prefix.
# They are order items, not colours, and an assistant that offers "Colour
# brochure" as a render colour is exactly the failure check 5 exists to stop.
_NOT_A_COLOUR = re.compile(r"\b(?:sample|brochure|warmshell|more products)\b", re.I)
_NAV_TAIL = re.compile(r"\s*more products\s*>?\s*$", re.I)


def harvest(soup: BeautifulSoup) -> dict:
    """Name lists, taken from the page *before* boilerplate stripping.

    Everything here lives in the navigation block that content extraction
    removes. Check 5 — real names only — depends on these lists, and typing
    them by hand would be inventing them.
    """
    colours, products = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = _NAV_TAIL.sub("", clean(a.get_text(" ", strip=True)))
        if not text or len(text) > 60:
            continue
        if "/products-by-colour/" in href:
            if not _NOT_A_COLOUR.search(text):
                colours.append(text)
        elif re.search(r"/products/[a-z0-9-]+/[a-z0-9-]+", href):
            products.append(text)

    # The stockists are the alt text of the pins on the supplier map, so they
    # are invisible to any extraction that reads rendered text — which is how a
    # first pass concluded the page named no merchants at all. The names are
    # published; they are just not published as prose.
    merchants = []
    for img in soup.find_all("img", class_="map-pin"):
        name = clean(img.get("alt") or "")
        if name and len(name) < 70:
            merchants.append(name)

    return {"colours": colours, "products": products, "merchants": merchants}


def contact_details(path: str) -> dict:
    """The contact line and opening hours, verbatim. Every refusal carries them.

    Read from the page rather than typed into configuration. A phone number an
    assistant invents is close to the worst thing it can print, so the only
    number it is able to give is one it read on the site.
    """
    soup = _soup(path)
    text = clean((soup.find("main") or soup).get_text("\n", strip=True))
    phone = re.search(r"\b0\d{3}\s?\d{3}\s?\d{4}\b|\b0\d{4}\s?\d{6}\b", text)
    hours = re.search(r"(Mon[^\n]{0,12}-\s*Fri[^\n]{0,40})", text, re.I)
    hours_line = ""
    if hours:
        after = text[hours.end():hours.end() + 60].split("\n")
        times = next((a for a in after if re.search(r"\d", a)), "")
        hours_line = clean(hours.group(1) + " " + times)
    return {"phone": phone.group(0) if phone else "",
            "hours": hours_line,
            "address": " ".join(text.split("\n")[3:10])[:200]}


def _faq_sections(main) -> list[Section]:
    """The FAQ is 32 question-and-answer pairs, not one page of prose.

    A question is the citable unit here: the heading is what the customer asked,
    which is also the thing a retrieval has to match.
    """
    out = []
    for sec in main.find_all("section", class_="faq-section"):
        head = sec.find(["h2", "h3"]) or sec.find("p", class_="h2-style")
        category = clean(head.get_text(" ", strip=True)) if head else ""
        for dt in sec.find_all("dt"):
            question = clean(dt.get_text(" ", strip=True))
            dd = dt.find_next_sibling("dd")
            answer = clean(dd.get_text("\n", strip=True)) if dd else ""
            if question and answer:
                label = f"{category} — {question}" if category else question
                out.append(Section(label, answer))
    return out


def _is_break(tag) -> bool:
    """A node that ends the section above it."""
    name = getattr(tag, "name", None)
    if name in ("h1", "h2", "h3", "h4", "h5"):
        return True
    if name == "p" and tag.get("class"):
        return any(c.endswith("-style") for c in tag.get("class"))
    return False


def _html_sections(main) -> list[Section]:
    """Split at heading tags and the site's paragraph-styled pseudo-headings."""
    out: list[Section] = []
    for node in main.find_all(_is_break):
        heading = clean(node.get_text(" ", strip=True))
        if heading.lower() in HTML_TAIL_MARKERS:
            break
        if not heading or heading.lower() in HTML_BOILERPLATE:
            continue
        body = []
        for sib in node.next_siblings:
            if _is_break(sib):
                break
            text = sib.get_text("\n", strip=True) if hasattr(sib, "get_text") else str(sib)
            if text:
                body.append(text)
        text = clean(HTML_FURNITURE_LINES.sub("", "\n".join(body)))
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if len(text) >= 40:
            out.append(Section(heading, text))
    return out


def _bold_sections(main) -> list[Section]:
    """Split where a short bold run opens a block — how the technical notes mark sections."""
    out: list[Section] = []
    heading, buf = "", []
    for para in main.find_all("p"):
        text = clean(para.get_text(" ", strip=True))
        if not text:
            continue
        lead = para.find(["strong", "b"])
        lead_text = clean(lead.get_text(" ", strip=True)) if lead else ""
        opens = bool(lead_text) and text.startswith(lead_text) and len(lead_text) <= 70

        if opens and len(lead_text) < len(text) * 0.9:
            if heading and buf:
                body = clean("\n".join(buf))
                if len(body) >= 40:
                    out.append(Section(heading, body))
            heading, buf = lead_text, [text[len(lead_text):].strip()]
        elif opens:
            if heading and buf:
                body = clean("\n".join(buf))
                if len(body) >= 40:
                    out.append(Section(heading, body))
            heading, buf = lead_text, []
        else:
            buf.append(text)
    if heading and buf:
        body = clean("\n".join(buf))
        if len(body) >= 40:
            out.append(Section(heading, body))
    return out


def extract_html(path: str, doc_type: str = "") -> tuple[Extracted, dict]:
    """Sections plus the harvest. Harvest first: stripping destroys its source."""
    soup = _soup(path)
    harvested = harvest(soup)

    title = clean(soup.title.get_text()) if soup.title else ""
    main = soup.find("main") or soup.find("body") or soup
    for tag in main.find_all(["script", "style", "noscript", "form", "svg"]):
        tag.decompose()

    if doc_type == "faq":
        sections = _faq_sections(main)
        if sections:
            return Extracted(sections, "clean", "", "", "dom", title), harvested

    # The published date the site prints on knowledge-base articles. Decision 12
    # lists it as a classification axis; it is also what "newest wins within a
    # type" resolves conflicts on.
    pub = main.find("p", class_="pub-date")
    printed = clean(pub.get_text(" ", strip=True)).replace(" th ", " ") if pub else ""

    sections = _html_sections(main)

    # Technical notes mark their sections with <strong>, not with a heading tag,
    # so a heading-only split returns one 5,000-character block for an article
    # that actually has sixteen numbered sections.
    if len(sections) <= 2:
        bold = _bold_sections(main)
        if len(bold) > len(sections):
            sections = bold

    # A product page's intro sits above its first heading, and it is the part
    # that says what the product is for. Take it from the lead paragraphs.
    if not sections or sum(len(s) for s in sections) < 300:
        para = [clean(p.get_text(" ", strip=True)) for p in main.find_all("p")]
        para = [p for p in para
                if len(p) > 60 and not any(b in p.lower() for b in HTML_BOILERPLATE)]
        if para:
            lead = Section(title.split("|")[0].strip() or "Description",
                           clean("\n\n".join(para[:8])))
            sections.insert(0, lead)

    quality = "clean" if len(sections) >= 3 else ("partial" if sections else "flat")
    return Extracted(sections, quality, "", printed, "dom", title), harvested


# ------------------------------------------------------------------- caveats


def caveats(sections: list[Section]) -> list[dict]:
    """Caveat sentences, tagged per document, deferrals excluded.

    Fine Stuff repeats its 8 °C limit under Mixing and again under Curing while
    the steps a user asks about sit elsewhere, so no chunking keeps them
    together. Tagging them against the document and appending by code does.
    """
    found, seen = [], set()
    for sec in sections:
        for sentence in re.split(r"(?<=[.!?])\s+", sec.text):
            # Sections keep their bullet characters, which is right for a
            # printed passage and wrong for a caveat quoted on its own line.
            s = re.sub(r"^[\s•·*\-–—]+", "", sentence).strip()
            s = re.sub(r"\s+", " ", s)
            if not 15 < len(s) < 320 or DEFERRAL.search(s):
                continue
            for kind, pattern in CAVEAT_PATTERNS:
                if pattern.search(s):
                    key = s.lower()
                    if key not in seen:
                        seen.add(key)
                        found.append({"type": kind, "sentence": s,
                                      "section": sec.heading})
                    break
    return found
