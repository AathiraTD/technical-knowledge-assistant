"""Source freshness, immutable provenance, and isolated crawler behavior."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from assistant.indexing import crawl

HTTP_CLIENT = httpx.Client


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(crawl, "ROOT", tmp_path)
    monkeypatch.setattr(crawl, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(crawl, "LEDGER", tmp_path / "cache" / "versions.json")
    monkeypatch.setattr(crawl.time, "sleep", lambda _: None)
    return tmp_path / "cache"


@pytest.fixture
def config():
    return {
        "site": "https://example.test", "sitemap": "https://example.test/sitemap.xml",
        "user_agent": "test-crawler", "timeout_seconds": 1, "delay_seconds": 0,
        "exclude_exact": ["/products/excluded", "/products/fallback"],
        "exclude_prefixes": ["/private/", "/excluded/"],
        "include_prefixes": ["/products/", "/support/", "/warmshell-natural-insulation/"],
        "exclusion_reasons": {"/products/excluded": "outdated", "/private/": "private"},
        "document_link_text": {"include": ["datasheet", "technical data"], "exclude": ["safety"]},
        "system_guides": {"include_urls": ["/docs/guide.pdf"]},
    }


@pytest.mark.parametrize("path,expected", [
    ("/products/test", "product_page"), ("/support/knowledgebase/test", "knowledge_base"),
    ("/support/faq", "faq"), ("/warmshell-natural-insulation/test", "product_page"),
    ("/other", "commercial"),
])
def test_page_classification(path, expected, config):
    assert crawl.classify_page(path, config) == expected


@pytest.mark.parametrize("path,expected", [
    ("/products/excluded", (False, "outdated")),
    ("/products/fallback", (False, "excluded by rule")),
    ("/private/a", (False, "private")), ("/excluded/a", (False, "excluded by rule")),
    ("/products/good", (True, "")), ("/other", (False, "outside the corpus boundary")),
])
def test_corpus_boundaries(path, expected, config):
    assert crawl.wanted(path, config) == expected


@pytest.mark.parametrize("text,expected", [
    ("", None), ("  Technical\n DATA  ", "datasheet"), ("Product datasheet", "datasheet"),
    ("Safety datasheet", None), ("Download brochure", None),
])
def test_link_text_controls_document_classification(text, expected, config):
    assert crawl.document_kind(text, config) == expected


def test_cache_paths_and_hashes(isolated_cache):
    assert crawl.cache_path("https://example.test/") == isolated_cache / "pages/index.html"
    assert crawl.cache_path("https://example.test/products/a%20b") == isolated_cache / "pages/products-a-b.html"
    assert crawl.cache_path("https://example.test/docs/A%20B.PDF") == isolated_cache / "documents/A-B.PDF"
    assert crawl.sha256("café") == crawl.sha256("café".encode())
    assert crawl.sha256("a") != crawl.sha256("b")


def test_config_and_ledger_reading(isolated_cache, tmp_path, monkeypatch, config):
    cfg_path = tmp_path / "sources.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(crawl, "CONFIG", cfg_path)
    assert crawl.load_config() == config
    assert crawl.load_ledger() == {}
    isolated_cache.mkdir()
    crawl.LEDGER.write_text('{"a": {"version": 2}}', encoding="utf-8")
    assert crawl.load_ledger() == {"a": {"version": 2}}


@pytest.mark.parametrize("binary", [False, True])
def test_default_cache_reuse_never_contacts_network(tmp_path, binary):
    dest = tmp_path / "source"
    dest.write_bytes(b"original")
    def refuse(request):
        pytest.fail("offline cache reuse contacted the source")
    with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
        content, cached, status, _ = crawl.load_or_fetch(client, "https://example.test/a", dest, 0, binary)
    assert content == (b"original" if binary else "original")
    assert cached and status == 200


@pytest.mark.parametrize("binary", [False, True])
def test_initial_fetch_returns_content_and_http_validators(tmp_path, binary):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, content=b"fresh", headers={"ETag": '"v1"', "Last-Modified": "yesterday"},
    ))) as client:
        result = crawl.load_or_fetch(client, "https://example.test/a", tmp_path / "missing", 0, binary)
    assert result[:3] == ((b"fresh" if binary else "fresh"), False, 200)
    assert result[3]["etag"] == '"v1"'


def test_empty_cache_file_is_refetched(tmp_path):
    dest = tmp_path / "empty"
    dest.touch()
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="fresh"))) as client:
        assert crawl.load_or_fetch(client, "https://example.test/a", dest, 0)[:3] == ("fresh", False, 200)


@pytest.mark.parametrize("status", [404, 410, 429, 500, 503])
def test_failed_fetch_reports_status_without_overwriting_cache(tmp_path, status):
    dest = tmp_path / "missing"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        assert crawl.load_or_fetch(client, "https://example.test/a", dest, 0)[:3] == (None, False, status)
    assert not dest.exists()


@pytest.mark.parametrize("binary", [False, True])
def test_refresh_uses_conditional_get_and_reuses_304_body(tmp_path, binary):
    dest = tmp_path / "source"
    dest.write_bytes(b"original")
    prior = {"etag": '"v1"', "last_modified": "Mon, 01 Jun 2026 00:00:00 GMT"}
    def unchanged(request):
        assert request.headers["If-None-Match"] == prior["etag"]
        assert request.headers["If-Modified-Since"] == prior["last_modified"]
        return httpx.Response(304, headers={"ETag": '"v1"'})
    with httpx.Client(transport=httpx.MockTransport(unchanged)) as client:
        content, cached, status, headers = crawl.load_or_fetch(
            client, "https://example.test/a", dest, 0, binary, refresh=True, prior=prior,
        )
    assert content == (b"original" if binary else "original")
    assert cached and status in (200, 304)
    assert headers["etag"] == '"v1"'


def test_refresh_discovers_changed_content(tmp_path):
    dest = tmp_path / "source"
    dest.write_text("original", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, text="changed", headers={"ETag": '"v2"'},
    ))) as client:
        result = crawl.load_or_fetch(client, "https://example.test/a", dest, 0, refresh=True)
    assert result[:3] == ("changed", False, 200)
    assert result[3]["etag"] == '"v2"'


def test_transient_refresh_failure_preserves_previous_source(tmp_path):
    dest = tmp_path / "source"
    dest.write_text("original", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        result = crawl.load_or_fetch(client, "https://example.test/a", dest, 0, refresh=True)
    assert result[:3] == (None, False, 503)
    assert dest.read_text(encoding="utf-8") == "original"


def test_version_reconciliation_retains_every_immutable_original(isolated_cache):
    ledger = {}
    url = "https://example.test/products/a"
    first = crawl.reconcile(ledger, url, "first", {"etag": '"v1"'})
    first_path = Path(first["source_path"])
    if not first_path.is_absolute():
        first_path = crawl.ROOT / first_path
    assert first_path.read_text(encoding="utf-8") == "first"
    second = crawl.reconcile(ledger, url, "second", {"etag": '"v2"'})
    assert second["version"] == 2 and second["supersedes"] == first["content_hash"]
    assert second["history"][0]["source_path"] == first["source_path"]
    assert first_path.read_text(encoding="utf-8") == "first"
    third = crawl.reconcile(ledger, url, "third", {})
    assert [r["version"] for r in third["history"]] == [1, 2]
    assert all("history" not in r for r in third["history"])


def test_unchanged_source_updates_validators_without_creating_version(isolated_cache):
    ledger = {}
    url = "https://example.test/products/a"
    first = crawl.reconcile(ledger, url, "same", {"etag": '"v1"'})
    ledger[url]["is_active"] = False
    second = crawl.reconcile(ledger, url, "same", {"etag": '"v2"', "last-modified": "today"})
    assert second["version"] == 1
    assert second["change"] == "unchanged"
    assert second["is_active"] is True
    assert second["etag"] == '"v2"' and second["last_modified"] == "today"
    assert second["first_seen_at"] == first["first_seen_at"]
    assert second["fetched_at"] == first["fetched_at"]
    assert second.get("history", []) == []


@pytest.mark.parametrize("url,content,suffix", [
    ("https://example.test/products/a", "café", ".html"),
    ("https://example.test/docs/A.PDF", b"%PDF-original", ".pdf"),
])
def test_archive_is_content_addressed_and_repeatable(isolated_cache, url, content, suffix):
    first = crawl.archive_source(url, content)
    assert first == crawl.archive_source(url, content)
    assert first.is_relative_to(isolated_cache / "versions") and first.suffix == suffix
    assert first.read_bytes() == (content.encode("utf-8") if isinstance(content, str) else content)
    changed = crawl.archive_source(url, b"replacement")
    assert changed != first and first.exists()
    assert crawl.archive_source(url + "?variant=2", content) != first


def test_corrupted_archive_is_rejected(isolated_cache):
    path = crawl.archive_source('https://example.test/a', b'original')
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        crawl.archive_source('https://example.test/a', b'original')


def test_revalidation_with_only_last_modified(isolated_cache):
    path = isolated_cache / 'page.html'
    path.parent.mkdir(parents=True)
    path.write_text('old')
    def respond(request):
        assert 'If-None-Match' not in request.headers
        assert request.headers['If-Modified-Since'] == 'yesterday'
        return httpx.Response(304)
    with HTTP_CLIENT(transport=httpx.MockTransport(respond)) as client:
        assert crawl.load_or_fetch(client, 'https://example.test/a', path, 0,
                                   refresh=True, prior={'last_modified': 'yesterday'})[0] == 'old'


def test_legacy_source_archive_survives_confirmed_deletion(isolated_cache, monkeypatch, config):
    isolated_cache.mkdir(parents=True)
    url = 'https://example.test/products/a'
    path = crawl.cache_path(url)
    path.parent.mkdir(parents=True)
    path.write_text('original')
    crawl.LEDGER.write_text(json.dumps({url: {'version': 1, 'is_active': True}}))
    config['system_guides']['include_urls'] = []
    assert run_crawler(monkeypatch, config, {
        config['sitemap']: httpx.Response(200, text=f'<loc>{url}</loc>'),
        url: httpx.Response(404),
    }, refresh=True) == 0
    record = crawl.load_ledger()[url]
    assert not record['is_active']
    assert (crawl.ROOT / record['source_path']).read_text() == 'original'


def test_malformed_discovery_does_not_publish(isolated_cache, monkeypatch, config):
    with pytest.raises(ValueError, match='sitemap'):
        run_crawler(monkeypatch, config, {config['sitemap']: httpx.Response(200, text='<urlset/>')})
    assert not (isolated_cache / 'crawl-current.json').exists()


def run_crawler(monkeypatch, config, responses, *, robots_failure=False, refresh=False):
    """Use the real HTTP client with deterministic source responses."""
    monkeypatch.setattr(crawl, "load_config", lambda: config)
    class Robots:
        def set_url(self, url):
            pass
        def read(self):
            if robots_failure:
                raise OSError("robots unavailable")
        def can_fetch(self, agent, url):
            return "blocked" not in url
    monkeypatch.setattr(crawl.urllib.robotparser, "RobotFileParser", Robots)
    def response(request):
        result = responses[str(request.url)]
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(crawl.httpx, "Client", lambda **kwargs: HTTP_CLIENT(
        transport=httpx.MockTransport(response), **kwargs,
    ))
    return crawl.main(["--refresh"] if refresh else [])


def test_main_records_pages_datasheets_guides_and_exclusions(isolated_cache, monkeypatch, config):
    html = '''<html><title>Product A</title><a href="/other">Other</a>
      <a href="/docs/data.pdf">Technical datasheet</a><a href="/docs/data.pdf">Datasheet</a>
      <a href="/docs/safety.pdf">Safety datasheet</a><a href="/docs/safety.pdf">Safety</a>
      <a href="/docs/empty.pdf"></a><a href="/docs/guide.pdf">Download guide</a></html>'''
    responses = {
        config["sitemap"]: httpx.Response(200, text="""<urlset>
          <loc>https://example.test/products/a</loc><loc>https://example.test/products/blocked</loc>
          <loc>https://example.test/products/untitled</loc><loc>https://example.test/outside</loc></urlset>"""),
        "https://example.test/products/a": httpx.Response(200, text=html),
        "https://example.test/products/untitled": httpx.Response(200, text="<p>Untitled</p>"),
        "https://example.test/docs/data.pdf": httpx.Response(200, content=b"%PDF-data"),
        "https://example.test/docs/guide.pdf": httpx.Response(200, content=b"%PDF-guide"),
    }
    assert run_crawler(monkeypatch, config, responses) == 0
    log = json.loads((isolated_cache / "crawl-log.json").read_text(encoding="utf-8"))
    assert log["counts"] == {"product_page": 2, "datasheet": 1, "system_guide": 1}
    assert len(log["skipped"]) == 4 and not log["errors"]
    assert {entry["url"] for entry in log["fetched"]} == set(responses) - {config["sitemap"]}
    assert all(entry["content_hash"].startswith("sha256:") for entry in log["fetched"])


def test_main_uses_cached_page_and_document(isolated_cache, monkeypatch, config):
    (isolated_cache / "pages").mkdir(parents=True)
    (isolated_cache / "documents").mkdir()
    crawl.cache_path("https://example.test/products/a").write_text("<title>Cached</title>", encoding="utf-8")
    crawl.cache_path("https://example.test/docs/guide.pdf").write_bytes(b"%PDF-cached")
    assert run_crawler(monkeypatch, config, {
        config["sitemap"]: httpx.Response(200, text="<loc>https://example.test/products/a</loc>"),
    }) == 0
    assert len(crawl.load_ledger()) == 2


def test_main_reports_fetch_failures_without_publishing_success(isolated_cache, monkeypatch, config):
    config["system_guides"]["include_urls"] = ["/docs/failed.pdf", "/docs/error.pdf", "/docs/blocked.pdf"]
    responses = {
        config["sitemap"]: httpx.Response(200, text="""<loc>https://example.test/products/failed</loc>
          <loc>https://example.test/products/error</loc>"""),
        "https://example.test/products/failed": httpx.Response(503),
        "https://example.test/products/error": httpx.ConnectError("offline"),
        "https://example.test/docs/failed.pdf": httpx.Response(500),
        "https://example.test/docs/error.pdf": httpx.ConnectError("offline"),
    }
    assert run_crawler(monkeypatch, config, responses, robots_failure=True) == 1
    log = json.loads((isolated_cache / "crawl-report.json").read_text(encoding="utf-8"))
    assert len(log["errors"]) == 4 and log["total_fetched"] == 0
    assert log["skipped"][0]["reason"] == "disallowed by robots.txt"
    assert not (isolated_cache / "crawl-log.json").exists()
    assert not crawl.LEDGER.exists()


def test_complete_refresh_lifecycle_preserves_previous_release_on_failure(isolated_cache, monkeypatch, config):
    config["system_guides"]["include_urls"] = []
    page = "https://example.test/products/a"
    doc = "https://example.test/docs/data.pdf"
    html = '<title>Product A</title><a href="/docs/data.pdf">Datasheet</a>'
    def responses(page_result, doc_result):
        return {
            config["sitemap"]: httpx.Response(200, text=f"<loc>{page}</loc>"),
            page: page_result, doc: doc_result,
        }
    assert run_crawler(monkeypatch, config, responses(
        httpx.Response(200, text=html, headers={"ETag": '"page-v1"'}),
        httpx.Response(200, content=b"%PDF-v1", headers={"ETag": '"doc-v1"'}),
    )) == 0
    original = crawl.load_ledger()[doc]
    original_path = crawl.ROOT / original["source_path"]
    assert original_path.read_bytes() == b"%PDF-v1"

    assert run_crawler(monkeypatch, config, responses(
        httpx.Response(304), httpx.Response(304),
    ), refresh=True) == 0
    assert crawl.load_ledger()[doc]["version"] == 1

    assert run_crawler(monkeypatch, config, responses(
        httpx.Response(304), httpx.Response(200, content=b"%PDF-v2", headers={"ETag": '"doc-v2"'}),
    ), refresh=True) == 0
    updated = crawl.load_ledger()[doc]
    assert updated["version"] == 2 and len(updated["history"]) == 1
    assert original_path.read_bytes() == b"%PDF-v1"
    release_before_failure = (isolated_cache / "crawl-current.json").read_bytes()
    ledger_before_failure = crawl.LEDGER.read_bytes()
    log_before_failure = (isolated_cache / "crawl-log.json").read_bytes()

    assert run_crawler(monkeypatch, config, responses(
        httpx.Response(200, text=html + "changed page"), httpx.Response(503),
    ), refresh=True) == 1
    assert (isolated_cache / "crawl-current.json").read_bytes() == release_before_failure
    assert crawl.LEDGER.read_bytes() == ledger_before_failure
    assert (isolated_cache / "crawl-log.json").read_bytes() == log_before_failure

    assert run_crawler(monkeypatch, config, responses(
        httpx.Response(200, text=html), httpx.Response(410),
    ), refresh=True) == 0
    assert crawl.load_ledger()[doc]["is_active"] is False
    published = json.loads((isolated_cache / "crawl-log.json").read_text(encoding="utf-8"))
    assert doc not in {row["url"] for row in published["fetched"]}
    assert original_path.read_bytes() == b"%PDF-v1"
