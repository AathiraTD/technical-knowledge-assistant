"""Crawl Lime Green's published material once, to disk.

Run once, by hand. The cache ships with the submission so the indexer — and the
assessors — work offline. Politeness is not optional here: this is a partner's
website, so one crawl, rate limited, identifying itself, respecting robots.txt.

    python -m assistant.crawl

Corpus boundary and the exclusion rules live in config/sources.json, so the
question "why isn't the safety data sheet in here?" has an answer on file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
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
    ledger[url] = record
    return record


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


def fetch(client: httpx.Client, url: str, delay: float) -> httpx.Response:
    time.sleep(delay)
    return client.get(url)


def load_or_fetch(
    client: httpx.Client, url: str, dest: Path, delay: float, binary: bool = False
) -> tuple[bytes | str | None, bool, int, dict]:
    """Return cached content if we already have it, otherwise fetch once.

    The cache ships with the submission, so a second run must not hit the
    partner's site again. Returns (content, from_cache, status).
    """
    if dest.exists() and dest.stat().st_size > 0:
        content = dest.read_bytes() if binary else dest.read_text(encoding="utf-8")
        return content, True, 200, {}
    r = fetch(client, url, delay)
    if r.status_code != 200:
        return None, False, r.status_code, {}
    headers = {k.lower(): v for k, v in r.headers.items()}
    return (r.content if binary else r.text), False, r.status_code, headers


def main() -> int:
    cfg = load_config()
    site = cfg["site"].rstrip("/")
    log = CrawlLog()
    from_cache = 0
    ledger = load_ledger()
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
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", sitemap.text)
        print(f"  {len(urls)} URLs listed\n")

        keep: list[str] = []
        for url in urls:
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
                text, cached, status, headers = load_or_fetch(client, url, dest, cfg["delay_seconds"])
            except Exception as exc:
                log.errors.append({"url": url, "error": str(exc)})
                print(f"  [{i:>3}/{len(keep)}] ERROR {url}: {exc}")
                continue

            if text is None:
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
                    path=str(dest.relative_to(ROOT)).replace("\\", "/"),
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

        # explicitly included system guides
        for rel in cfg["system_guides"]["include_urls"]:
            doc_url = urljoin(site + "/", rel.lstrip("/"))
            if doc_url not in seen_docs:
                seen_docs.add(doc_url)
                pending_docs.append((doc_url, "system guide", "Warmshell system"))

        print(f"\nDocuments to fetch: {len(pending_docs)}\n")

        for i, (doc_url, link_text, product) in enumerate(pending_docs, 1):
            if not robots.can_fetch(cfg["user_agent"], doc_url):
                log.skipped.append(Skipped(doc_url, "disallowed by robots.txt"))
                continue
            dest = cache_path(doc_url)
            try:
                blob, cached, status, headers = load_or_fetch(
                    client, doc_url, dest, cfg["delay_seconds"], binary=True
                )
            except Exception as exc:
                log.errors.append({"url": doc_url, "error": str(exc)})
                print(f"  [{i:>3}/{len(pending_docs)}] ERROR {doc_url}: {exc}")
                continue
            if blob is None:
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
                    path=str(dest.relative_to(ROOT)).replace("\\", "/"),
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
    (CACHE / "crawl-log.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

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
