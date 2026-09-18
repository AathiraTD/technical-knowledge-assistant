"""Everything indexed has to come from the one site the configuration names.

The crawler is the system's largest external boundary and the only place where
an attacker supplies bytes directly. Two risks are specific to it. The first is
SSRF by redirection: a sitemap entry or a link on a page is followed, and the
response is a `302` to somewhere else — an internal address, or simply a
publisher nobody approved — and its content is then archived, chunked and cited
as Lime Green's own material. The second is time-of-check to time-of-use: a
document is classified and hashed during the crawl and read again later during
indexing, and the file on disk need not be the file that was validated.

The rule these tests enforce is that the publisher check is **applied to the
final URL and re-applied after every redirect**, and that the content hash is
re-verified at extraction rather than trusted from the crawl log. A foreign
sitemap entry raises before anything is published; a foreign linked document is
skipped with `publisher` recorded as the reason, so the exclusion is
explicable; a redirect chain that leaves the site or loops raises, and the
parametrised case asserts what was *visited*, not merely what was returned,
because a request that is made and then discarded has already reached the
host.

What this file is not is a network test. Every response comes from
`httpx.MockTransport`, and the crawl fixtures are imported from
`test_crawl.py` so that the isolation of cache, ledger and configuration is
the same one the crawler's own suite uses. Nothing here contacts the real site,
and nothing here checks extraction quality.
"""
import httpx
import pytest
from assistant.indexing import crawl, index
from test_crawl import isolated_cache, config, run_crawler, HTTP_CLIENT
from test_pipeline_regressions import pipeline


def test_source_changed_between_classification_and_extraction_is_rejected(pipeline):
    """A cached file edited after the crawl validated it must not be extracted."""
    stage, repo = pipeline
    stage()
    import json
    log = json.loads((index.CACHE / 'crawl-log.json').read_text())
    ledger = json.loads((index.CACHE / 'versions.json').read_text())
    (index.CACHE / 'plaster.html').write_text('changed after validation')
    with pytest.raises(ValueError, match='during indexing'):
        index._extract_one(log['fetched'][0], ledger, '2026-01-01')


def test_foreign_sitemap_entry_cannot_publish(isolated_cache, monkeypatch, config):
    """Discovery naming another host fails the run rather than crawling it."""
    with pytest.raises(ValueError, match='publisher'):
        run_crawler(monkeypatch, config, {
            config['sitemap']: httpx.Response(200, text='<loc>https://other.test/products/a</loc>')})


def test_foreign_linked_document_is_excluded(isolated_cache, monkeypatch, config):
    """An off-site datasheet link is skipped, and the log says the publisher is why."""
    config['system_guides']['include_urls'] = []
    page = 'https://example.test/products/a'
    assert run_crawler(monkeypatch, config, {
        config['sitemap']: httpx.Response(200, text=f'<loc>{page}</loc>'),
        page: httpx.Response(200, text='<a href="https://other.test/a.pdf">Datasheet</a>'),
    }) == 0
    import json
    log = json.loads((isolated_cache / 'crawl-log.json').read_text())
    assert len(log['fetched']) == 1
    assert 'publisher' in log['skipped'][0]['reason']


@pytest.mark.parametrize('location', ['/new', 'https://other.test/private', '/loop'])
def test_redirects_remain_inside_publisher_and_are_bounded(isolated_cache, location):
    """Same-site hop is followed; off-site and looping hops raise, and nothing foreign is visited."""
    visited = []
    def respond(request):
        visited.append(str(request.url))
        if request.url.path == '/new':
            return httpx.Response(200, text='document')
        return httpx.Response(302, headers={'Location': location})
    with HTTP_CLIENT(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        if location == '/new':
            assert crawl.fetch(client, 'https://example.test/a', 0).text == 'document'
        else:
            with pytest.raises(ValueError, match='redirect'):
                crawl.fetch(client, 'https://example.test/a', 0)
    assert all(url.startswith('https://example.test/') for url in visited)
