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

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import ollama, use_utf8
from .embedcache import EmbeddingCache
from .crawl import archive_source, atomic_write
from .extract import (
    Section,
    _soup,
    caveats as find_caveats,
    contact_details,
    extract_html,
    extract_pdf,
    harvest,
)
from .model import (
    AUTHORITY,
    Caveat,
    Chunk,
    CrawlRun,
    Document,
    DocumentUpdate,
    DocumentVersion,
    Excluded,
    Snapshot,
)
from .store.factory import open_repository

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


# A citation is read by a person, so a heading has to stay the length of a
# heading. Merging a short section forward joins the two names, and joining
# three FAQ questions produced a 200-character "section" that nobody could look
# up. Past this limit the first name is kept and the rest becomes an ellipsis,
# because the first is the one the passage actually opens with.
MAX_HEADING = 70


def _merge_headings(carried: str, following: str) -> str:
    """Join two section names, or keep the first when the join is too long."""
    if not carried:
        return following
    joined = f"{carried} / {following}"
    if len(joined) <= MAX_HEADING:
        return joined
    if len(carried) <= MAX_HEADING:
        return f"{carried} …"
    # Cut on a word boundary. "anything else to your p …" reads as a defect;
    # "anything else to your …" reads as a heading that was too long.
    clipped = carried[:MAX_HEADING].rsplit(" ", 1)[0]
    return (clipped or carried[:MAX_HEADING]).rstrip(" ,;/-") + " …"


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
            heading = _merge_headings(carry_heading, heading)
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


def _fixture_hash(path: Path) -> str:
    """A fixture changes when its file does, which is all the signal needed."""
    return "fixture:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _classify(log: dict, ledger: dict, known: dict[str, str]
              ) -> tuple[list[str], list[str], list[str], list[str], dict]:
    """Split the crawl into new, changed, unchanged and failed.

    The comparison is against the hashes of what is **currently being served**,
    read from the store, not against a local file. A ledger can drift from the
    index; the index cannot drift from itself.
    """
    new, changed, unchanged = [], [], []
    failed = [entry["url"] for entry in log.get("errors", [])]
    entries: dict[str, dict] = {}
    for entry in log["fetched"]:
        url = entry["url"]
        entries[url] = entry
        if not (ROOT / entry["path"]).exists():
            entry["failure"] = "cached file missing"
            failed.append(url)
            continue
        fresh = "sha256:" + hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
        record = ledger.setdefault(url, {})
        expected = record.get("content_hash", "")
        if len(expected) == 71 and expected != fresh:
            # The shipped legacy ledger hashed HTML after Python's universal
            # newline decoding. Migrate only when that exact old digest matches;
            # immutable releases always require a byte-for-byte hash match.
            legacy = entry["kind"] == "page" and not record.get("source_path")
            normalized = ("sha256:" + hashlib.sha256(
                (ROOT / entry["path"]).read_text(encoding="utf-8").encode("utf-8")).hexdigest()) if legacy else ""
            if expected != normalized:
                entry["failure"] = "source hash mismatch; previous version retained"
                failed.append(url)
                continue
        record["content_hash"] = fresh
        prior = known.get(url)
        if prior is None:
            new.append(url)
        elif prior != fresh:
            changed.append(url)
        else:
            unchanged.append(url)
    return new, changed, unchanged, failed, entries


def _harvest_all(log: dict) -> tuple[list[str], list[str], list[str], dict]:
    """Name lists and the contact line, over every cached page.

    Deliberately not part of the delta. Harvesting parses the navigation block
    and costs about a second for the whole corpus, where extraction, chunking
    and embedding cost minutes — and carrying stale lists forward would leave a
    withdrawn colour in the vocabulary that check 5 trusts.
    """
    colours: list[str] = []
    products: list[str] = []
    merchants: list[str] = []
    contact: dict = {}
    for entry in log["fetched"]:
        if entry["kind"] != "page":
            continue
        full = ROOT / entry["path"]
        if not full.exists():
            continue
        harvested = harvest(_soup(str(full)))
        colours += harvested["colours"]
        products += harvested["products"]
        merchants += harvested.get("merchants", [])
        if entry["url"].rstrip("/").endswith("/contact"):
            contact = contact_details(str(full))
    return colours, products, merchants, contact


def _dedupe(names: list[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _extract_one(entry: dict, ledger: dict, now: str) -> tuple[DocumentUpdate, dict]:
    """One crawled document into a unit of work, plus its report row."""
    url, path, kind = entry["url"], entry["path"], entry["kind"]
    doc_type = entry["doc_type"]
    full = ROOT / path
    original = full.read_bytes()
    expected = ledger.get(url, {}).get("content_hash", "")
    if expected and expected != "sha256:" + hashlib.sha256(original).hexdigest():
        raise ValueError(f"Source changed during indexing: {url}")
    # Archive before extraction: a subsequent fetch must not overwrite the
    # original file referenced by an older database version.
    archived = archive_source(url, original, cache=CACHE)

    if kind == "page":
        extracted, _harvested = extract_html(str(archived), doc_type)
    else:
        extracted = extract_pdf(str(archived))

    title = entry.get("title") or extracted.title or ""
    record = ledger.get(url, {})
    source_date = iso_date(extracted.printed_date, record.get("fetched_at", "")[:10])

    document = Document(
        canonical_url=url, title=title, document_type=doc_type,
        authority=AUTHORITY.get(doc_type, 9),
        audience="public",                  # everything crawled is published
        product=product_name(entry, title),
        link_text=entry.get("link_text", ""),
        active_version=record.get("version", 1),
    )
    version = DocumentVersion(
        canonical_url=url, version=record.get("version", 1),
        content_hash=record.get("content_hash", ""),
        source_path=os.path.relpath(archived, ROOT).replace("\\", "/"),
        etag=record.get("etag", ""),
        source_last_modified=record.get("last_modified", ""),
        first_seen_at=record.get("first_seen_at", now),
        fetched_at=record.get("fetched_at", now),
        checked_at=record.get("checked_at", now),
        is_active=True, extraction_quality=extracted.quality, notes=extracted.note,
    )

    passages = chunk_sections(extracted.sections)
    chunks = [
        Chunk(canonical_url=url, version=document.active_version, chunk_index=i,
              section=heading, content=text, audience="public",
              product=document.product, document_type=doc_type,
              authority=document.authority, source_date=source_date)
        for i, (heading, text) in enumerate(passages)
    ]
    found = find_caveats(extracted.sections)
    caveats = [Caveat(url, c["type"], c["sentence"], c["section"]) for c in found]

    row = {
        "url": url, "name": Path(path).name, "type": doc_type,
        "quality": extracted.quality, "detector": extracted.detector,
        "sections": len(extracted.sections), "chunks": len(passages),
        "chars": extracted.chars, "date": source_date, "caveats": len(found),
        "note": extracted.note,
    }
    return DocumentUpdate(document, version, chunks, caveats), row


def _fixture_updates(known: dict[str, str], now: str, directory: Path | None = None,
                     approved: bool = False,
                     ) -> tuple[list[DocumentUpdate], list[dict], set[str]]:
    """Evaluation fixtures, treated as documents with the same delta rules.

    Lime Green publishes nothing that is not public, so without these the
    audience filter would be a claim with no test behind it. They are indexed as
    staff-only and a public caller cannot retrieve them, because the filter is a
    WHERE clause rather than an instruction.
    """
    updates, rows, urls = [], [], set()
    directory = directory if directory is not None else ROOT / "eval" / "fixtures"
    for path in sorted(directory.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if approved and (spec.get("approved") is not True or not spec.get("approved_by")):
            raise ValueError(f"Staff source requires explicit approval and reviewer: {path.name}")
        if (spec.get("audience") not in ("public", "trade", "staff")
                or not spec.get("sections")
                or any(not section.get("text", "").strip() for section in spec["sections"])):
            raise ValueError(f"Invalid authored source: {path.name}")
        url = spec["canonical_url"]
        if url in urls:
            raise ValueError(f"Duplicate authored source identity: {url}")
        urls.add(url)
        digest = _fixture_hash(path)
        if known.get(url) == digest:
            continue
        document = Document(
            canonical_url=url, title=spec["title"],
            document_type=spec["document_type"],
            authority=AUTHORITY.get(spec["document_type"], 9),
            audience=spec["audience"], product=spec.get("product", ""),
            link_text=spec.get("link_text", ""), active_version=1,
        )
        version = DocumentVersion(
            canonical_url=url, version=1, content_hash=digest,
            source_path=os.path.relpath(archive_source(url, path.read_bytes(), cache=CACHE), ROOT).replace("\\", "/"),
            first_seen_at=now, fetched_at=now, checked_at=now, is_active=True,
            extraction_quality="clean", notes=(f"approved by {spec['approved_by']}" if approved else "evaluation fixture, not real data"),
        )
        sections = [Section(sec["heading"], sec["text"]) for sec in spec["sections"]]
        passages = chunk_sections(sections) if approved else [(s.heading, s.text) for s in sections]
        chunks = [
            Chunk(canonical_url=url, version=1, chunk_index=i,
                  section=heading, content=text,
                  audience=spec["audience"], product=spec.get("product", ""),
                  document_type=spec["document_type"],
                  authority=AUTHORITY.get(spec["document_type"], 9),
                  source_date=spec.get("source_date", ""))
            for i, (heading, text) in enumerate(passages)
        ]
        caveats = [Caveat(url, kind, sentence, heading)
                   for kind, sentence, heading in find_caveats(sections)] if approved else []
        updates.append(DocumentUpdate(document, version, chunks, caveats))
        rows.append({
            "url": url, "name": path.name, "type": spec["document_type"],
            "quality": "clean", "detector": "approved-source" if approved else "fixture",
            "sections": len(sections), "chunks": len(chunks),
            "chars": sum(len(x["text"]) for x in spec["sections"]),
            "date": spec.get("source_date", ""), "caveats": len(caveats),
            "note": version.notes,
        })
    return updates, rows, urls


def build(repo=None, verbose: bool = True, rebuild: bool = False,
          staff_dir: Path | None = None) -> dict:
    """Build a release and leave an explicit failure record if any stage raises."""
    try:
        return _build(repo, verbose, rebuild, staff_dir)
    except Exception as exc:
        atomic_write(INDEX_DIR / "ingestion-failure.json", json.dumps({
            "status": "failed", "error": str(exc),
            "at": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8"))
        raise


def _validate_vectors(vectors):
    for vector in vectors:
        if (len(vector) != ollama.EMBED_DIMENSIONS
                or not all(math.isfinite(n) and abs(n) <= 3.4028234e38 for n in vector)
                or not any(vector)):
            raise ValueError("Embedding model or cache returned an invalid vector")


def _build(repo, verbose, rebuild, staff_dir) -> dict:
    """Index what changed, and leave the rest alone.

    A rebuild is a special case of a delta against an empty store, not the other
    way round. `rebuild=True` forces that by ignoring what is already indexed,
    which is the escape hatch for a chunking change: the content hashes are
    unchanged, so nothing would otherwise be reprocessed even though every
    passage boundary moved.
    """
    started = time.perf_counter()
    pointer = CACHE / "crawl-current.json"
    if pointer.exists():
        release = CACHE / json.loads(pointer.read_text(encoding="utf-8"))["release"]
        bundle = json.loads(release.read_text(encoding="utf-8"))
        log, ledger = bundle["log"], bundle["ledger"]
    else:
        log = json.loads((CACHE / "crawl-log.json").read_text(encoding="utf-8"))
        ledger = json.loads((CACHE / "versions.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ollama.require(ollama.EMBED_MODEL)

    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    owned = repo is None
    repo = repo or open_repository(INDEX_DIR / "knowledge.db")
    try:
        previous = repo.snapshot()
        known = repo.active_content_hashes()
        incompatible = previous is not None and (
            previous.embedding_model != ollama.EMBED_MODEL
            or previous.embedding_dimensions != ollama.EMBED_DIMENSIONS
            or previous.chunking_version != CHUNKING_VERSION)
        reprocess = rebuild or incompatible
        comparison = {} if reprocess else known
        new, changed, unchanged, failed, entries = _classify(log, ledger, comparison)

        fixture_updates, fixture_rows, fixture_urls = _fixture_updates(comparison, now)
        staff_updates, staff_rows, staff_urls = _fixture_updates(
            comparison, now, staff_dir if staff_dir is not None else ROOT / "data" / "staff", True)
        if staff_urls & (fixture_urls | set(entries)):
            raise ValueError("Staff source identity collides with another source")

        # Removed: served now, absent from this crawl, and not a fixture.
        crawled = set(entries)
        # Incomplete discovery is not proof of withdrawal (a failed product
        # page may also hide every datasheet linked from it).
        removed = ([] if log.get("errors") else sorted(u for u in known
                         if u not in crawled and u not in fixture_urls and u not in staff_urls))

        say(f"Crawl of {len(log['fetched'])} documents against the live index:")
        say(f"  new {len(new)} · changed {len(changed)} · "
            f"unchanged {len(unchanged)} · removed {len(removed)} · "
            f"failed {len(failed)}")
        if not (new or changed or removed or fixture_updates):
            say("  nothing to reprocess")

        # -------------------------------------------------- extract and chunk
        updates: list[DocumentUpdate] = []
        report_rows: list[dict] = []
        for url in new + changed:
            update, row = _extract_one(entries[url], ledger, now)
            row["change"] = "new" if url in set(new) else "changed"
            if update.version.extraction_quality == "failed" or not update.chunks:
                failed.append(url)
                row["change"] = "failed"
            else:
                updates.append(update)
            report_rows.append(row)
        for url in unchanged:
            report_rows.append({"url": url, "name": Path(entries[url]["path"]).name,
                                "type": entries[url]["doc_type"],
                                "quality": "unchanged", "detector": "-",
                                "sections": 0, "chunks": 0, "chars": 0,
                                "date": "", "caveats": 0, "change": "unchanged",
                                "note": "skipped: content hash unchanged"})
        for url in failed:
            if url in {row["url"] for row in report_rows}:
                continue
            report_rows.append({"url": url, "name": "", "type": "",
                                "quality": "failed", "detector": "-",
                                "sections": 0, "chunks": 0, "chars": 0,
                                "date": "", "caveats": 0, "change": "failed",
                                "note": entries.get(url, {}).get("failure", "source fetch failed; previous version retained")})

        updates += fixture_updates
        report_rows += fixture_rows
        updates += staff_updates
        report_rows += staff_rows
        if incompatible and failed:
            raise ValueError("Cannot change index configuration with failed sources")

        say(f"  extracted {sum(len(u.chunks) for u in updates)} passages from "
            f"{len(updates)} document(s)")

        # ----------------------------------------------------------- harvest
        valid_log = {**log, "fetched": [e for e in log["fetched"] if e["url"] not in failed]}
        colours, products, merchants, contact = _harvest_all(valid_log)

        # ----------------------------------------------------------- embed
        to_embed = [embedding_text(c) for u in updates for c in u.chunks]
        embed_started = time.perf_counter()
        cache_hits = cache_misses = 0
        if to_embed:
            def tick(done: int, total: int) -> None:
                print(f"\r  {done}/{total} ({100 * done / total:.0f}%)",
                      end="", flush=True)

            with EmbeddingCache() as cache:
                found = cache.get_many(to_embed, ollama.EMBED_MODEL,
                                       ollama.EMBED_DIMENSIONS)
                todo = [i for i in range(len(to_embed)) if i not in found]
                say(f"  {len(found)} already embedded, {len(todo)} to compute")
                if todo:
                    fresh = ollama.embed([to_embed[i] for i in todo],
                                         progress=tick if verbose else None)
                    if len(fresh) != len(todo):
                        raise ValueError("Embedding model returned the wrong vector count")
                    _validate_vectors(fresh)
                    cache.put_many([(to_embed[i], v) for i, v in zip(todo, fresh)],
                                   ollama.EMBED_MODEL, ollama.EMBED_DIMENSIONS)
                    found.update(dict(zip(todo, fresh)))
                    if verbose:
                        print()
                cache_hits, cache_misses = cache.hits, cache.misses
            _validate_vectors(found.values())
            position = 0
            for update in updates:
                for c in update.chunks:
                    c.embedding = found[position]
                    position += 1
        embed_seconds = time.perf_counter() - embed_started

        # --------------------------------------------------------- publish
        excluded = [Excluded(x["url"], x.get("reason", ""))
                    for x in log.get("skipped", [])]
        snapshot = Snapshot(
            snapshot_id=f"snap-{uuid.uuid4().hex}",
            created_at=now,
            embedding_model=ollama.EMBED_MODEL,
            embedding_dimensions=ollama.EMBED_DIMENSIONS,
            chunking_version=CHUNKING_VERSION,
            document_count=0,          # counted by the store; see apply_delta
            chunk_count=0,
            notes={
                "parent_snapshot": previous.snapshot_id if previous else None,
                "colours": _dedupe(colours),
                "products": _dedupe(products),
                "merchants": _dedupe(merchants),
                "contact": contact,
                "embed_seconds": round(embed_seconds, 1),
                "generation_model": ollama.GENERATION_MODEL,
            },
        )
        run = CrawlRun(
            started_at=now, completed_at=now,
            documents_checked=len(log["fetched"]), documents_new=len(new),
            documents_changed=len(changed), documents_unchanged=len(unchanged),
            documents_removed=len(removed), documents_failed=len(failed),
            snapshot_id=snapshot.snapshot_id,
        )

        repo.apply_delta(updates, removed, snapshot, excluded, run)

        live = repo.snapshot()
        counts = repo.counts() if hasattr(repo, "counts") else {}
    finally:
        if owned:
            repo.close()

    fixture_count = len(fixture_urls)
    report = {
        "snapshot": live.snapshot_id,
        "built_at": now,
        "embedding_model": live.embedding_model,
        "embedding_dimensions": live.embedding_dimensions,
        "chunking_version": CHUNKING_VERSION,
        "documents": live.document_count,
        "public_documents": live.document_count - fixture_count,
        "evaluation_fixtures": fixture_count,
        "chunks": live.chunk_count,
        "delta": {
            "new": len(new), "changed": len(changed), "unchanged": len(unchanged),
            "removed": len(removed), "failed": len(failed),
            "reprocessed": len(updates),
            "removed_urls": removed,
        },
        "caveats": sum(len(u.caveats) for u in updates),
        "excluded": len(excluded),
        "colours": live.notes.get("colours", []),
        "products": live.notes.get("products", []),
        "merchants": live.notes.get("merchants", []),
        "contact": live.notes.get("contact", {}),
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


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(
        prog="assistant.index",
        description="Index what changed since the last run.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="reprocess every document, ignoring what is already indexed. "
             "Needed after a chunking change, because that moves every passage "
             "boundary without moving a single content hash.")
    parser.add_argument("--staff-dir", type=Path, help="Directory of explicitly approved staff JSON sources")
    args = parser.parse_args(argv)
    try:
        options = {"staff_dir": args.staff_dir} if args.staff_dir else {}
        report = build(rebuild=args.rebuild, **options)
    except ollama.OllamaUnavailable as exc:
        print(f"\nCannot build the index.\n\n{exc}\n", file=sys.stderr)
        return 1
    print(summarise(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
