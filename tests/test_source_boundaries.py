"""Only the configured publisher can supply public knowledge."""
import httpx
import pytest
from assistant import crawl, index
from test_crawl import isolated_cache, config, run_crawler, HTTP_CLIENT
from test_pipeline_regressions import pipeline


def test_source_changed_between_classification_and_extraction_is_rejected(pipeline):
    stage, repo = pipeline
    stage()
    import json
    log = json.loads((index.CACHE / 'crawl-log.json').read_text())
    ledger = json.loads((index.CACHE / 'versions.json').read_text())
    (index.CACHE / 'plaster.html').write_text('changed after validation')
    with pytest.raises(ValueError, match='during indexing'):
        index._extract_one(log['fetched'][0], ledger, '2026-01-01')


def test_foreign_sitemap_entry_cannot_publish(isolated_cache, monkeypatch, config):
    with pytest.raises(ValueError, match='publisher'):
        run_crawler(monkeypatch, config, {
            config['sitemap']: httpx.Response(200, text='<loc>https://other.test/products/a</loc>')})


def test_foreign_linked_document_is_excluded(isolated_cache, monkeypatch, config):
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
