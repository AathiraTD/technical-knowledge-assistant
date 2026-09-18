"""Crawl Lime Green's published material once, to disk.

Run once, by hand. The cache ships with the submission so the indexer — and the
assessors — work offline. Politeness is not optional here: this is a partner's
website, so one crawl, rate limited, identifying itself, respecting robots.txt.

    python -m assistant.indexing.crawl

Corpus boundary and the exclusion rules live in config/sources.json, so the
question "why isn't the safety data sheet in here?" has an answer on file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from .. import paths

ROOT = paths.ROOT
CONFIG = ROOT / "config" / "sources.json"
CACHE = ROOT / "data" / "cache"
LEDGER = CACHE / "versions.json"


@dataclass
class Fetched:
    """One retrieved file, plus why it is in the corpus."""

    url: str
    path: str
    kind: str  # page | datasheet | system_guide
    doc_type: str  # product_page | knowledge_base | faq | commercial | datasheet | system_guide
    title: str = ""
    link_text: str = ""  # the text of the link that pointed here, for documents
    product: str = ""  # the page a document was linked from
    status: int = 0
    bytes: int = 0
    # Version identity. The same fields become columns in the production
    # schema (documents / document_versions); only the substrate changes.
    content_hash: str = ""
    etag: str = ""
    last_modified: str = ""
    version: int = 1
    first_seen_at: str = ""
    fetched_at: str = ""
    is_active: bool = True


@dataclass
class Skipped:
    url: str
    reason: str


@dataclass
class CrawlLog:
    """The record of what was taken and what was left, written beside the cache."""

    fetched: list[Fetched] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(blob: bytes | str) -> str:
    data = blob.encode("utf-8") if isinstance(blob, str) else blob
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_ledger() -> dict:
    """The version ledger: one record per source URL, carried across crawls.

    This is what makes a re-crawl a delta rather than a replacement, and what
    lets the index hold exactly one active version of a document — the rule the
    answer engine depends on when it refuses to blend two versions.
    """
    pointer = CACHE / "crawl-current.json"
    if pointer.exists():
        release = CACHE / json.loads(pointer.read_text(encoding="utf-8"))["release"]
        return json.loads(release.read_text(encoding="utf-8"))["ledger"]
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {}


def reconcile(ledger: dict, url: str, content: bytes | str, headers: dict) -> dict:
    """Decide whether this fetch is a new version, and return its record.

    A content hash, not a timestamp: sites routinely change banners, tracking
    markup and footers without touching the technical content, and re-embedding
    on that would be waste. Equally, a changed hash is a real change even when
    the server's Last-Modified has not moved.
    """
    digest = sha256(content)
    prior = ledger.get(url)
    stamp = now_iso()

    if prior is None:
        record = {
            "version": 1,
            "content_hash": digest,
            "etag": headers.get("etag", ""),
            "last_modified": headers.get("last-modified", ""),
            "first_seen_at": stamp,
            "fetched_at": stamp,
            "checked_at": stamp,
            "is_active": True,
            "supersedes": None,
            "change": "new",
        }
    elif prior["content_hash"] == digest:
        record = {**prior, "checked_at": stamp, "change": "unchanged"}
    else:
        record = {
            "version": prior["version"] + 1,
            "content_hash": digest,
            "etag": headers.get("etag", ""),
            "last_modified": headers.get("last-modified", ""),
            "first_seen_at": prior["first_seen_at"],
            "fetched_at": stamp,
            "checked_at": stamp,
            "is_active": True,
            "supersedes": prior["content_hash"],
            "change": "changed",
        }
    history = list(prior.get("history", [])) if prior else []
    if prior and prior["content_hash"] != digest:
        history.append({k: v for k, v in prior.items() if k != "history"})
    record["history"] = history
    source = archive_source(url, content)
    record["source_path"] = os.path.relpath(source, ROOT).replace("\\", "/")
    record["is_active"] = True
    for header, key in (("etag", "etag"), ("last-modified", "last_modified")):
        if header in headers:
            record[key] = headers[header]
    ledger[url] = record
    return record


def archive_source(url: str, content: bytes | str, *, cache: Path | None = None) -> Path:
    """Keep original source bytes under immutable URL/content identities."""
    blob = content.encode("utf-8") if isinstance(content, str) else content
    identity = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    digest = hashlib.sha256(blob).hexdigest()
    suffix = ".pdf" if urlparse(url).path.lower().endswith(".pdf") else ".html"
    path = (cache if cache is not None else CACHE) / "versions" / identity / (digest + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        atomic_write(path, blob)
    elif path.read_bytes() != blob:
        raise ValueError(f"Archived source is corrupt: {path}")
    return path


def atomic_write(path: Path, blob: bytes) -> None:
    """Publish a complete file; interrupted writes cannot truncate its predecessor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(uuid.uuid4().hex + ".tmp")
    try:
        staging.write_bytes(blob)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def cache_path(url: str) -> Path:
    """A readable, stable filename for a URL. Documents keep their own name."""
    parsed = urlparse(url)
    path = unquote(parsed.path).strip("/")
    if path.lower().endswith(".pdf"):
        return CACHE / "documents" / re.sub(r"[^A-Za-z0-9._-]+", "-", Path(path).name)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", path) or "index"
    return CACHE / "pages" / f"{slug}.html"


def classify_page(path: str, cfg: dict) -> str:
    """What kind of page this is, decided by URL prefix and nothing cleverer.

    The document type is not cosmetic: it sets the authority rank that resolves
    a conflict between two passages — datasheet, then product page, then
    knowledge-base article, then FAQ — and it is the noun that appears in a
    citation. So it has to be derived from something stable, and on this site
    the URL structure is the stable thing. Page titles are not (they carry
    marketing copy), and filenames are not (they carry typos and date
    prefixes); see `document_kind` for the same problem solved by link text.

    Warmshell pages classify as product pages because that is what they are —
    a system rather than a single product, but read for selection the same way.

    Everything unmatched is `commercial`, which covers contact,
    find-a-supplier and order-a-sample. That default is safe only because the
    corpus boundary has already run: `wanted()` decides what is crawled at all,
    so an unrecognised path reaching here is inside the boundary by rule rather
    than an accident being given a type.

    `cfg` is accepted and unused. It keeps the signature uniform with the other
    classifiers in this module so the rules can move into `crawl.json` without
    touching every call site.
    """
    if path.startswith("/products/"):
        return "product_page"
    if path.startswith("/support/knowledgebase/"):
        return "knowledge_base"
    if path.startswith("/support/faq"):
        return "faq"
    if path.startswith("/warmshell-natural-insulation/"):
        return "product_page"
    return "commercial"


def wanted(path: str, cfg: dict) -> tuple[bool, str]:
    """Apply the corpus boundary. Returns (keep, reason-if-not)."""
    if path in cfg["exclude_exact"]:
        return False, cfg["exclusion_reasons"].get(path, "excluded by rule")
    for prefix in cfg["exclude_prefixes"]:
        if path.startswith(prefix):
            return False, cfg["exclusion_reasons"].get(prefix, "excluded by rule")
    for prefix in cfg["include_prefixes"]:
        if path.startswith(prefix):
            return True, ""
    return False, "outside the corpus boundary"


def document_kind(link_text: str, cfg: dict) -> str | None:
    """Classify a linked document by its link text, never by its filename.

    Filenames on this site carry spaces, typos ('Insualtion', 'silgaurd',
    'Peformance') and date prefixes. The link text is what a person reads and
    what the site is consistent about.
    """
    text = " ".join(link_text.split()).lower()
    if not text:
        return None
    rules = cfg["document_link_text"]
    for token in rules["exclude"]:
        if token in text:
            return None
    for token in rules["include"]:
        if token in text:
            return "datasheet"
    return None


def _origin(url):
    parsed = httpx.URL(url)
    return parsed.scheme, parsed.host, parsed.port


def _get(client, url, headers):
    target = url
    for _ in range(6):
        response = client.get(target, headers=headers, follow_redirects=False)
        if not response.has_redirect_location:
            return response
        target = urljoin(target, response.headers["location"])
        if _origin(target) != _origin(url):
            raise ValueError("Cross-publisher redirect rejected")
    raise ValueError("Too many source redirects")


def fetch(client: httpx.Client, url: str, delay: float,
          headers: dict | None = None) -> httpx.Response:
    """Bounded retries for transient transport/server failures."""
    attempt = 0
    while True:
        time.sleep(delay * (2 ** attempt))
        try:
            response = _get(client, url, headers)
        except httpx.TransportError:
            if attempt == 2:
                raise
            attempt += 1
            continue
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            return response
        attempt += 1


def load_or_fetch(
    client: httpx.Client, url: str, dest: Path, delay: float, binary: bool = False,
    *, refresh: bool = False, prior: dict | None = None,
) -> tuple[bytes | str | None, bool, int, dict]:
    """Return cached content if we already have it, otherwise fetch once.

    The cache ships with the submission, so a second run must not hit the
    partner's site again. Returns (content, from_cache, status).
    """
    cached = dest.exists() and dest.stat().st_size > 0
    if cached and not refresh:
        content = dest.read_bytes() if binary else dest.read_text(encoding="utf-8")
        return content, True, 200, {}
    conditional = {}
    if cached and prior:
        if prior.get("etag"):
            conditional["If-None-Match"] = prior["etag"]
        if prior.get("last_modified"):
            conditional["If-Modified-Since"] = prior["last_modified"]
    r = fetch(client, url, delay, headers=conditional)
    headers = {k.lower(): v for k, v in r.headers.items()}
    if r.status_code == 304 and cached:
        content = dest.read_bytes() if binary else dest.read_text(encoding="utf-8")
        return content, True, 304, headers
    if r.status_code != 200:
        return None, False, r.status_code, {}
    return (r.content if binary else r.text), False, r.status_code, headers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and publish a versioned source release")
    parser.add_argument("--refresh", action="store_true", help="Revalidate cached sources with HTTP validators")
    args = parser.parse_args(argv)
    return run(refresh=args.refresh)


def run(refresh: bool = False) -> int:
    """Crawl, and leave evidence behind when it does not finish.

    A thin wrapper by design. All the work is in `_run`; what this adds is the
    guarantee that **a failed crawl is visible on disk rather than only in the
    terminal that was running it.** `crawl-failure.json` records the error and
    the time, which is what stops a half-finished fetch from being read later
    as a site that withdrew its documents — the difference between "the crawl
    broke" and "these pages are gone" is the difference between retrying and
    deactivating ninety documents.

    The exception is re-raised rather than swallowed. The marker is for the
    next person to look; the non-zero exit and the traceback are for the caller
    that has to decide whether to index on top of this, and
    `assistant/indexing/pipeline.py` treats either as a reason to retry rather
    than publish.

    Nothing here rolls anything back, because nothing needs to: the previous
    source release stays current until a complete crawl replaces it, so the
    failure path is to leave everything exactly as it was.
    """
    try:
        return _run(refresh)
    except Exception as exc:
        atomic_write(CACHE / "crawl-failure.json", json.dumps({
            "status": "failed", "error": str(exc),
            "at": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8"))
        raise


def _run(refresh: bool) -> int:
    cfg = load_config()
    site = cfg["site"].rstrip("/")
    log = CrawlLog()
    from_cache = 0
    ledger = load_ledger()
    # Upgrade legacy URL caches before fetching new bytes, while originals
    # still exist. New release records always point at immutable evidence.
    for url, record in ledger.items():
        old = cache_path(url)
        if not record.get("source_path") and old.exists():
            record["source_path"] = os.path.relpath(archive_source(url, old.read_bytes()), ROOT).replace("\\", "/")
    changes: dict[str, int] = {}

    (CACHE / "pages").mkdir(parents=True, exist_ok=True)
    (CACHE / "documents").mkdir(parents=True, exist_ok=True)

    robots = urllib.robotparser.RobotFileParser()
    robots.set_url(f"{site}/robots.txt")
    try:
        robots.read()
    except Exception as exc:  # a missing robots.txt is not permission to ignore politeness
        print(f"  ! could not read robots.txt ({exc}); continuing rate limited")

    headers = {"User-Agent": cfg["user_agent"]}
    with httpx.Client(
        headers=headers,
        timeout=cfg["timeout_seconds"],
        follow_redirects=True,
    ) as client:
        print(f"Sitemap: {cfg['sitemap']}")
        sitemap = fetch(client, cfg["sitemap"], 0)
        sitemap.raise_for_status()
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", sitemap.text)
        if not urls:
            raise ValueError("Empty or malformed sitemap; previous release retained")
        print(f"  {len(urls)} URLs listed\n")

        keep: list[str] = []
        for url in urls:
            if _origin(url) != _origin(site):
                raise ValueError("Sitemap contains a URL outside the configured publisher")
            path = urlparse(url).path.rstrip("/") or "/"
            ok, reason = wanted(path, cfg)
            if ok:
                keep.append(url)
            else:
                log.skipped.append(Skipped(url, reason))

        print(f"In the corpus boundary: {len(keep)} pages")
        print(f"Excluded by rule:       {len(log.skipped)} pages\n")

        # ---- pages, and the documents they link to -------------------------
        seen_docs: set[str] = set()
        pending_docs: list[tuple[str, str, str]] = []  # url, link text, product page

        # Seed the explicitly included system guides first. They are linked with
        # text like "Download our design guide here", which correctly fails the
        # datasheet link-text test — so without seeding, the general rule would
        # claim them as skipped and the explicit include could never add them.
        for rel in cfg["system_guides"]["include_urls"]:
            doc_url = urljoin(site + "/", rel.lstrip("/"))
            seen_docs.add(doc_url)
            pending_docs.append((doc_url, "system guide", "Warmshell system"))

        for i, url in enumerate(keep, 1):
            path = urlparse(url).path.rstrip("/") or "/"
            if not robots.can_fetch(cfg["user_agent"], url):
                log.skipped.append(Skipped(url, "disallowed by robots.txt"))
                continue
            dest = cache_path(url)
            try:
                prior = ledger.get(url)
                source = ROOT / prior["source_path"] if prior and prior.get("source_path") else dest
                text, cached, status, headers = load_or_fetch(client, url, source, cfg["delay_seconds"], refresh=refresh, prior=prior)
            except Exception as exc:
                log.errors.append({"url": url, "error": str(exc)})
                print(f"  [{i:>3}/{len(keep)}] ERROR {url}: {exc}")
                continue

            if text is None:
                if status in (404, 410):
                    continue
                log.errors.append({"url": url, "status": status})
                print(f"  [{i:>3}/{len(keep)}] {status} {path}")
                continue

            if not cached:
                dest.write_text(text, encoding="utf-8")
            else:
                from_cache += 1
            rec = reconcile(ledger, url, text, headers)
            changes[rec["change"]] = changes.get(rec["change"], 0) + 1
            soup = BeautifulSoup(text, "lxml")
            title = soup.title.get_text(strip=True) if soup.title else ""
            doc_type = classify_page(path, cfg)

            log.fetched.append(
                Fetched(
                    url=url,
                    path=rec["source_path"],
                    kind="page",
                    doc_type=doc_type,
                    title=title,
                    status=status,
                    bytes=len(text),
                    content_hash=rec["content_hash"],
                    etag=rec["etag"],
                    last_modified=rec["last_modified"],
                    version=rec["version"],
                    first_seen_at=rec["first_seen_at"],
                    fetched_at=rec["fetched_at"],
                    is_active=rec["is_active"],
                )
            )
            print(f"  [{i:>3}/{len(keep)}] {doc_type:<15} {path}")

            # queue linked documents, classified by link text
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" not in href.lower():
                    continue
                link_text = a.get_text(" ", strip=True)
                kind = document_kind(link_text, cfg)
                doc_url = urljoin(url, href)
                if _origin(doc_url) != _origin(site):
                    log.skipped.append(Skipped(doc_url, "outside the configured publisher"))
                    continue
                if kind is None:
                    if doc_url not in seen_docs:
                        log.skipped.append(
                            Skipped(doc_url, f"document link text '{link_text or '(none)'}' is not a technical datasheet")
                        )
                        seen_docs.add(doc_url)
                    continue
                if doc_url in seen_docs:
                    continue
                seen_docs.add(doc_url)
                pending_docs.append((doc_url, link_text, title))

        print(f"\nDocuments to fetch: {len(pending_docs)}\n")

        for i, (doc_url, link_text, product) in enumerate(pending_docs, 1):
            if not robots.can_fetch(cfg["user_agent"], doc_url):
                log.skipped.append(Skipped(doc_url, "disallowed by robots.txt"))
                continue
            dest = cache_path(doc_url)
            try:
                prior = ledger.get(doc_url)
                source = ROOT / prior["source_path"] if prior and prior.get("source_path") else dest
                blob, cached, status, headers = load_or_fetch(
                    client, doc_url, source, cfg["delay_seconds"], binary=True,
                    refresh=refresh, prior=prior,
                )
            except Exception as exc:
                log.errors.append({"url": doc_url, "error": str(exc)})
                print(f"  [{i:>3}/{len(pending_docs)}] ERROR {doc_url}: {exc}")
                continue
            if blob is None:
                if status in (404, 410):
                    continue
                log.errors.append({"url": doc_url, "status": status})
                print(f"  [{i:>3}/{len(pending_docs)}] {status} {doc_url}")
                continue

            if not cached:
                dest.write_bytes(blob)
            else:
                from_cache += 1
            rec = reconcile(ledger, doc_url, blob, headers)
            changes[rec["change"]] = changes.get(rec["change"], 0) + 1
            is_guide = link_text == "system guide"
            log.fetched.append(
                Fetched(
                    url=doc_url,
                    path=rec["source_path"],
                    kind="system_guide" if is_guide else "datasheet",
                    doc_type="system_guide" if is_guide else "datasheet",
                    link_text=link_text,
                    product=product,
                    status=status,
                    bytes=len(blob),
                    content_hash=rec["content_hash"],
                    etag=rec["etag"],
                    last_modified=rec["last_modified"],
                    version=rec["version"],
                    first_seen_at=rec["first_seen_at"],
                    fetched_at=rec["fetched_at"],
                    is_active=rec["is_active"],
                )
            )
            print(f"  [{i:>3}/{len(pending_docs)}] {dest.name}")

    # ---- the crawl log -----------------------------------------------------
    counts: dict[str, int] = {}
    for f in log.fetched:
        counts[f.doc_type] = counts.get(f.doc_type, 0) + 1

    out = {
        "site": site,
        "counts": counts,
        "total_fetched": len(log.fetched),
        "total_skipped": len(log.skipped),
        "errors": log.errors,
        "fetched": [asdict(f) for f in log.fetched],
        "skipped": [asdict(s) for s in log.skipped],
    }
    atomic_write(CACHE / "crawl-report.json", json.dumps(out, ensure_ascii=False).encode("utf-8"))
    if not log.errors:
        fetched_urls = {entry.url for entry in log.fetched}
        for url, record in ledger.items():
            if url not in fetched_urls:
                record["is_active"] = False
        release_name = "releases/" + uuid.uuid4().hex + ".json"
        atomic_write(CACHE / release_name, json.dumps({"log": out, "ledger": ledger}, ensure_ascii=False).encode("utf-8"))
        atomic_write(CACHE / "crawl-current.json", json.dumps({"release": release_name}).encode("utf-8"))
        # Compatibility reports are not the publication boundary. Readers use
        # the single release pointer above, so these cannot be read half-paired.
        atomic_write(CACHE / "crawl-log.json", json.dumps(out, ensure_ascii=False).encode("utf-8"))
        atomic_write(LEDGER, json.dumps(ledger, ensure_ascii=False).encode("utf-8"))

    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    for doc_type, n in sorted(counts.items()):
        print(f"  {doc_type:<18} {n:>3}")
    print(f"  {'TOTAL':<18} {len(log.fetched):>3}")
    print(f"\n  excluded by rule   {len(log.skipped):>3}")
    print(f"  errors             {len(log.errors):>3}")
    print(f"  served from cache  {from_cache:>3}")
    print()
    for change in ("new", "changed", "unchanged"):
        if changes.get(change):
            print(f"  {change:<18} {changes[change]:>3}")
    print(f"\n  cache:  {CACHE.relative_to(ROOT)}")
    print(f"  log:    {(CACHE / 'crawl-log.json').relative_to(ROOT)}")
    print(f"  ledger: {LEDGER.relative_to(ROOT)}")
    return 1 if log.errors else 0


if __name__ == "__main__":
    sys.exit(main())
