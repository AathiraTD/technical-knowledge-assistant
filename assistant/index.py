"""Build the index: extract, chunk, tag caveats, embed, publish a snapshot.

Run it with `python -m assistant.index`. It reads the cached crawl, so it needs
no network — only Ollama, running locally, for the embeddings.

Three things here are deliberate rather than incidental.

**Bullets are kept whole.** Solo's application section is one bullet per
background at roughly 2,000 characters. A size-based splitter cuts a thickness
away from the substrate it applies to, which is the single most dangerous thing
this corpus can do. So an over-long section is split at bullet boundaries, and
a bullet longer than the target is left over-long rather than cut.

**Caveats are tagged against the document, not the chunk.** Decision 11.

**The report is written whether or not the run went well.** An extraction that
produced two sections from a twenty-page guide is a fact an assessor should be
able to read, not one they should have to infer from a bad answer.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import ollama, use_utf8
from .embedcache import EmbeddingCache
from .extract import (
    Section,
    caveats as find_caveats,
    contact_details,
    extract_html,
    extract_pdf,
)
from .model import (
    AUTHORITY,
    Caveat,
    Chunk,
    Document,
    DocumentVersion,
    Excluded,
    Snapshot,
)
from .store import EmbeddedRepository

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
INDEX_DIR = ROOT / "data" / "index"

# The chunking rules, versioned. The engine records this against every answer
# and refuses to run against an index built under a different one, because a
# retrieval tuned to one chunk shape is not valid against another.
CHUNKING_VERSION = "structure-aware/1.0"

TARGET_CHARS = 1800     # a comfortable passage; sections under this stay whole
MIN_CHARS = 120         # below this a section is merged forward, not indexed alone
HARD_MAX = 4000         # above this, split at bullets even mid-section

BULLET = re.compile(r"^\s*(?:[-–—•*·]|\(?[a-z0-9]{1,3}[.)])\s+", re.M)


# ------------------------------------------------------------------ chunking


def _split_on_bullets(text: str) -> list[str]:
    """Split at bullet starts, never inside one."""
    marks = [m.start() for m in BULLET.finditer(text)]
    if len(marks) < 2:
        return [text]
    pieces, start = [], 0
    for mark in marks:
        if mark > start:
            pieces.append(text[start:mark])
            start = mark
    pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]


def chunk_sections(sections: list[Section]) -> list[tuple[str, str]]:
    """Sections to (heading, text) passages, bullets kept whole.

    A short section is merged into the one after it — a two-line "Colours"
    heading retrieves better attached to its context than alone — and a long
    one is split only where the document already has a boundary.
    """
    out: list[tuple[str, str]] = []
    carry_heading, carry_text = "", ""

    for sec in sections:
        heading = sec.heading.strip() or "Introduction"
        text = sec.text.strip()
        if not text:
            continue

        if carry_text:
            heading = f"{carry_heading} / {heading}" if carry_heading else heading
            text = f"{carry_text}\n{text}"
            carry_heading, carry_text = "", ""

        if len(text) < MIN_CHARS:
            carry_heading, carry_text = heading, text
            continue

        if len(text) <= TARGET_CHARS:
            out.append((heading, text))
            continue

        # Long section: prefer the document's own bullet boundaries.
        parts, buf = _split_on_bullets(text), ""
        if len(parts) == 1:
            # No bullets to split on. Split at blank lines, and only if it is
            # over the hard ceiling — an over-long passage costs context, a
            # badly cut one costs correctness.
            if len(text) <= HARD_MAX:
                out.append((heading, text))
                continue
            parts = [p for p in re.split(r"\n{2,}", text) if p.strip()]

        for part in parts:
            if buf and len(buf) + len(part) > TARGET_CHARS:
                out.append((heading, buf.strip()))
                buf = part
            else:
                buf = f"{buf}\n{part}" if buf else part
        if buf.strip():
            out.append((heading, buf.strip()))

    if carry_text:
        if out:
            h, t = out[-1]
            out[-1] = (h, f"{t}\n{carry_text}")
        else:
            out.append((carry_heading or "Introduction", carry_text))
    return out


# ------------------------------------------------------------------- tidying


_PRODUCT_NOISE = re.compile(r"\s*\|\s*Lime Green\s*$|\s*\bfor\b.*$", re.I)


def product_name(entry: dict, title: str) -> str:
    """The product a document belongs to, as a name rather than a page title.

    The crawl records a datasheet's product as the full title of the page that
    linked to it — 'Natural Lime Mortar for Old Buildings, Repointing, Pointing
    Repair and Stone Walls | Lime Green'. Check 3 compares a figure against the
    product it was published for, and this name is also prefixed onto the chunk
    before embedding, so a page title dressed as a product name costs twice.
    """
    raw = entry.get("product") or title or ""
    raw = raw.split("|")[0]
    # 'Aerogel wall insulation, Solo Onecoat Lime Plaster / Silic8 AeroGel
    # Adhesive' is three names in a trench coat. Take the first.
    raw = re.split(r"\s*[/,]\s*", raw)[0]
    raw = _PRODUCT_NOISE.sub("", raw).strip(" -–:")
    if not raw or len(raw) < 3:
        slug = entry["url"].rstrip("/").rsplit("/", 1)[-1]
        raw = re.sub(r"\.(pdf|html?)$", "", slug, flags=re.I)
        raw = raw.replace("%20", " ").replace("-", " ").replace("_", " ").title()
    return raw[:80].strip()


def iso_date(printed: str, fallback: str = "") -> str:
    """Printed dates to ISO, so 'newest wins' can actually compare two of them.

    The sheets print 9/12/24 and the articles print 4 April 2017. Left as
    published, a lexical sort puts 9/12/24 before 5/9/24, which inverts the
    rule it exists to serve.
    """
    printed = (printed or "").strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d.%m.%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(printed, fmt).date().isoformat()
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", printed)
    if m:
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(" ".join(m.groups()), fmt).date().isoformat()
            except ValueError:
                pass
    return (fallback or "")[:10]


def embedding_text(chunk) -> str:
    """The retrieval view of a chunk: product and section, then the passage."""
    head = " — ".join(p for p in (chunk.product, chunk.section) if p)
    return head + chr(10) + chunk.content if head else chunk.content


# --------------------------------------------------------------------- build


def build(repo=None, verbose: bool = True) -> dict:
    """Extract, chunk, embed and publish. Returns the ingestion report."""
    started = time.perf_counter()
    log = json.loads((CACHE / "crawl-log.json").read_text(encoding="utf-8"))
    ledger = json.loads((CACHE / "versions.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ollama.require(ollama.EMBED_MODEL)

    documents: list[Document] = []
    versions: list[DocumentVersion] = []
    chunks: list[Chunk] = []
    caveat_rows: list[Caveat] = []
    report_rows: list[dict] = []
    colours: list[str] = []
    products: list[str] = []
    merchants: list[str] = []
    contact: dict = {}

    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    say(f"Indexing {len(log['fetched'])} documents from the cache\n")

    for entry in log["fetched"]:
        url, path, kind = entry["url"], entry["path"], entry["kind"]
        doc_type = entry["doc_type"]
        full = ROOT / path

        if not full.exists():
            report_rows.append({"url": url, "quality": "failed",
                                "note": "cached file missing", "chunks": 0})
            continue

        if kind == "page":
            extracted, harvested = extract_html(str(full), doc_type)
            colours += harvested["colours"]
            products += harvested["products"]
            merchants += harvested.get("merchants", [])
            if url.rstrip("/").endswith("/contact"):
                contact = contact_details(str(full))
        else:
            extracted = extract_pdf(str(full))

        title = entry.get("title") or extracted.title or ""
        record = ledger.get(url, {})
        source_date = iso_date(extracted.printed_date,
                               record.get("fetched_at", "")[:10])

        doc = Document(
            canonical_url=url,
            title=title,
            document_type=doc_type,
            authority=AUTHORITY.get(doc_type, 9),
            audience="public",          # everything crawled is published material
            product=product_name(entry, title),
            link_text=entry.get("link_text", ""),
            active_version=record.get("version", 1),
        )
        documents.append(doc)

        versions.append(DocumentVersion(
            canonical_url=url,
            version=record.get("version", 1),
            content_hash=record.get("content_hash", ""),
            source_path=path,
            etag=record.get("etag", ""),
            source_last_modified=record.get("last_modified", ""),
            first_seen_at=record.get("first_seen_at", now),
            fetched_at=record.get("fetched_at", now),
            checked_at=record.get("checked_at", now),
            is_active=True,
            extraction_quality=extracted.quality,
            notes=extracted.note,
        ))

        passages = chunk_sections(extracted.sections)
        for i, (heading, text) in enumerate(passages):
            chunks.append(Chunk(
                canonical_url=url, version=doc.active_version, chunk_index=i,
                section=heading, content=text, audience="public",
                product=doc.product, document_type=doc_type,
                authority=doc.authority, source_date=source_date,
            ))

        found = find_caveats(extracted.sections)
        for cav in found:
            caveat_rows.append(Caveat(url, cav["type"], cav["sentence"], cav["section"]))

        report_rows.append({
            "url": url, "name": Path(path).name, "type": doc_type,
            "quality": extracted.quality, "detector": extracted.detector,
            "sections": len(extracted.sections), "chunks": len(passages),
            "chars": extracted.chars, "date": source_date,
            "caveats": len(found), "note": extracted.note,
        })

    # Evaluation fixtures. Lime Green publishes nothing that is not public, so
    # the audience filter would otherwise be a claim with no test behind it.
    # These are indexed as staff-only, and a public caller cannot retrieve them
    # because the filter is a WHERE clause rather than a prompt instruction.
    fixture_count = 0
    for fixture in sorted((ROOT / "eval" / "fixtures").glob("*.json")):
        spec = json.loads(fixture.read_text(encoding="utf-8"))
        url = spec["canonical_url"]
        documents.append(Document(
            canonical_url=url, title=spec["title"],
            document_type=spec["document_type"],
            authority=AUTHORITY.get(spec["document_type"], 9),
            audience=spec["audience"], product=spec.get("product", ""),
            link_text=spec.get("link_text", ""), active_version=1,
        ))
        versions.append(DocumentVersion(
            canonical_url=url, version=1, content_hash="fixture",
            source_path=str(fixture.relative_to(ROOT)).replace("\\", "/"),
            first_seen_at=now, fetched_at=now, checked_at=now, is_active=True,
            extraction_quality="clean", notes="evaluation fixture, not real data",
        ))
        for i, sec in enumerate(spec["sections"]):
            chunks.append(Chunk(
                canonical_url=url, version=1, chunk_index=i,
                section=sec["heading"], content=sec["text"],
                audience=spec["audience"], product=spec.get("product", ""),
                document_type=spec["document_type"],
                authority=AUTHORITY.get(spec["document_type"], 9),
                source_date=spec.get("source_date", ""),
            ))
        fixture_count += 1
        report_rows.append({
            "url": url, "name": fixture.name, "type": spec["document_type"],
            "quality": "clean", "detector": "fixture",
            "sections": len(spec["sections"]), "chunks": len(spec["sections"]),
            "chars": sum(len(s["text"]) for s in spec["sections"]),
            "date": spec.get("source_date", ""), "caveats": 0,
            "note": f"evaluation fixture, audience={spec['audience']}",
        })

    excluded = [Excluded(s["url"], s.get("reason", "")) for s in log.get("skipped", [])]

    # ------------------------------------------------------------- embedding
    say(f"Extracted {len(chunks)} chunks from {len(documents)} documents.")
    say(f"Embedding with {ollama.EMBED_MODEL} ...")

    def tick(done: int, total: int) -> None:
        pct = 100 * done / total if total else 100
        print(f"\r  {done}/{total} ({pct:.0f}%)", end="", flush=True)

    # What gets embedded is not what gets printed. A chunk reading "Add between
    # 5 and 6 litres of clean water per 25kg sack" never says which product it
    # belongs to, so a question naming Solo could not reach it and a page about
    # aerogel outranked the Solo datasheet. Prefixing the product and section
    # gives the passage the context a reader already has from the page it sits
    # on. This is the retrieval view only: citations and printed text are
    # untouched, so no published figure is altered by it.
    to_embed = [embedding_text(c) for c in chunks]
    embed_started = time.perf_counter()

    with EmbeddingCache() as cache:
        known = cache.get_many(to_embed, ollama.EMBED_MODEL, ollama.EMBED_DIMENSIONS)
        todo = [i for i in range(len(to_embed)) if i not in known]
        say(f"  {len(known)} already embedded, {len(todo)} to compute")

        if todo:
            fresh = ollama.embed(
                [to_embed[i] for i in todo],
                progress=tick if verbose else None,
            )
            cache.put_many(
                [(to_embed[i], v) for i, v in zip(todo, fresh)],
                ollama.EMBED_MODEL, ollama.EMBED_DIMENSIONS,
            )
            known.update(dict(zip(todo, fresh)))
        cache_hits, cache_misses = cache.hits, cache.misses

    embed_seconds = time.perf_counter() - embed_started
    say(f"\n  embeddings ready in {embed_seconds:.1f}s "
        f"({cache_hits} from cache, {cache_misses} computed)")
    for i, c in enumerate(chunks):
        c.embedding = known[i]

    # -------------------------------------------------------------- publish
    def dedupe(names: list[str]) -> list[str]:
        seen, out = set(), []
        for n in names:
            if n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out

    snapshot = Snapshot(
        snapshot_id=f"snap-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        created_at=now,
        embedding_model=ollama.EMBED_MODEL,
        embedding_dimensions=ollama.EMBED_DIMENSIONS,
        chunking_version=CHUNKING_VERSION,
        document_count=len(documents),
        chunk_count=len(chunks),
        notes={
            "colours": dedupe(colours),
            "products": dedupe(products),
            "merchants": dedupe(merchants),
            "contact": contact,
            "embed_seconds": round(embed_seconds, 1),
            "generation_model": ollama.GENERATION_MODEL,
        },
    )

    owned = repo is None
    repo = repo or EmbeddedRepository(INDEX_DIR / "knowledge.db")
    try:
        repo.publish(documents, versions, chunks, snapshot, caveat_rows, excluded)
        counts = repo.counts() if hasattr(repo, "counts") else {}
    finally:
        if owned:
            repo.close()

    report = {
        "snapshot": snapshot.snapshot_id,
        "built_at": now,
        "embedding_model": snapshot.embedding_model,
        "embedding_dimensions": snapshot.embedding_dimensions,
        "chunking_version": CHUNKING_VERSION,
        "documents": len(documents),
        "public_documents": len(documents) - fixture_count,
        "evaluation_fixtures": fixture_count,
        "chunks": len(chunks),
        "caveats": len(caveat_rows),
        "excluded": len(excluded),
        "colours": snapshot.notes["colours"],
        "products": snapshot.notes["products"],
        "merchants": snapshot.notes["merchants"],
        "contact": contact,
        "embed_seconds": round(embed_seconds, 1),
        "embeddings_cached": cache_hits,
        "embeddings_computed": cache_misses,
        "total_seconds": round(time.perf_counter() - started, 1),
        "rows": report_rows,
        "table_counts": counts,
    }
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "ingestion-report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


# ---------------------------------------------------------------------- main


def summarise(report: dict) -> str:
    """The ingestion report as a person reads it."""
    rows = report["rows"]
    by_quality: dict[str, int] = {}
    for r in rows:
        by_quality[r["quality"]] = by_quality.get(r["quality"], 0) + 1

    lines = [
        "",
        f"Snapshot {report['snapshot']}",
        f"  documents        {report['public_documents']} published"
        f" + {report['evaluation_fixtures']} evaluation fixture"
        f"{'s' if report['evaluation_fixtures'] != 1 else ''} (staff-only)",
        f"  chunks           {report['chunks']}",
        f"  caveats tagged   {report['caveats']}",
        f"  excluded on file {report['excluded']}",
        f"  colours          {len(report['colours'])}",
        f"  products         {len(report['products'])}",
        f"  merchants        {len(report['merchants'])}",
        f"  embedding        {report['embedding_model']} "
        f"({report['embedding_dimensions']}d) in {report['embed_seconds']}s",
        f"  total            {report['total_seconds']}s",
        "",
        "  extraction quality: " + ", ".join(
            f"{k} {v}" for k, v in sorted(by_quality.items())),
    ]
    poor = [r for r in rows if r["quality"] in ("failed", "flat")]
    if poor:
        lines += ["", "  documents that extracted badly:"]
        lines += [f"    [{r['quality']}] {r['name']}  {r['note']}" for r in poor]
    else:
        lines += ["", "  no document failed extraction."]
    lines += ["", f"  report: data/index/ingestion-report.json", ""]
    return "\n".join(lines)


def main() -> int:
    use_utf8()
    try:
        report = build()
    except ollama.OllamaUnavailable as exc:
        print(f"\nCannot build the index.\n\n{exc}\n", file=sys.stderr)
        return 1
    print(summarise(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
