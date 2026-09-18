"""Invalid source releases and vectors must never replace live evidence."""
import json
from dataclasses import replace

import pytest

from assistant.indexing import index
from assistant.model import DocumentUpdate
from assistant.repository import IndexMismatch
from test_pipeline_regressions import pipeline, URL, served_text
from test_repository_contract import A, doc, ver, chunk, snap


def test_source_hash_mismatch_keeps_original(pipeline):
    stage, repo = pipeline
    stage()
    index.build(repo, False)
    (index.CACHE / 'plaster.html').write_text('tampered', encoding='utf-8')
    report = index.build(repo, False)
    assert report['delta']['failed'] == 1
    assert '5 to 6 litres' in served_text(repo)


def test_configuration_change_with_failed_source_cannot_publish(pipeline, monkeypatch):
    stage, repo = pipeline
    stage()
    index.build(repo, False)
    before = repo.snapshot()
    stage(failed=True)
    monkeypatch.setattr(index.ollama, 'EMBED_MODEL', 'new-model')
    with pytest.raises(ValueError, match='failed sources'):
        index.build(repo, False)
    assert repo.snapshot() == before


@pytest.mark.parametrize('vectors', [[], [[1, 0]], [[float('nan'), 0, 0, 0]], [[0, 0, 0, 0]]])
def test_invalid_embedding_cannot_publish(pipeline, monkeypatch, vectors):
    stage, repo = pipeline
    stage()
    monkeypatch.setattr(index.ollama, 'embed', lambda *a, **k: vectors)
    with pytest.raises(ValueError, match='Embedding model'):
        index.build(repo, False)
    assert repo.snapshot() is None


def staff_file(directory, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    spec = dict(canonical_url='staff://mixing', title='Reviewed mixing note',
                audience='staff', approved=True, approved_by='Technical team',
                document_type='knowledge_base', sections=[dict(heading='Mixing', text='Approved internal mixing guidance.')])
    spec.update(overrides)
    path = directory / 'note.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    return path


def test_approved_staff_import_update_and_withdrawal(pipeline, tmp_path):
    stage, repo = pipeline
    stage()
    directory = tmp_path / 'staff'
    path = staff_file(directory)
    index.build(repo, False, staff_dir=directory)
    assert 'staff://mixing' not in {d.canonical_url for d in repo.manifest()}
    assert 'staff://mixing' in {d.canonical_url for d in repo.manifest(('staff',))}
    index.build(repo, False, staff_dir=directory)
    assert len(repo.versions('staff://mixing')) == 1
    staff_file(directory, title='Revised note')
    index.build(repo, False, staff_dir=directory)
    assert len(repo.versions('staff://mixing')) == 2
    assert repo.active_version('staff://mixing').notes == 'approved by Technical team'
    path.unlink()
    index.build(repo, False, staff_dir=directory)
    assert repo.active_version('staff://mixing') is None


@pytest.mark.parametrize('bad', [dict(approved=False), dict(approved_by=''), dict(audience='secret'),
                              dict(sections=[]), dict(sections=[dict(heading='Empty', text=' ')]),
                              dict(canonical_url=URL)])
def test_invalid_staff_submission_is_rejected_before_publication(pipeline, tmp_path, bad):
    stage, repo = pipeline
    stage()
    directory = tmp_path / 'staff'
    staff_file(directory, **bad)
    with pytest.raises(ValueError):
        index.build(repo, False, staff_dir=directory)
    assert repo.snapshot() is None


def test_build_reads_atomic_crawl_release(pipeline):
    stage, repo = pipeline
    stage()
    log = json.loads((index.CACHE / 'crawl-log.json').read_text())
    ledger = json.loads((index.CACHE / 'versions.json').read_text())
    (index.CACHE / 'release.json').write_text(json.dumps(dict(log=log, ledger=ledger)))
    (index.CACHE / 'crawl-current.json').write_text(json.dumps(dict(release='release.json')))
    (index.CACHE / 'crawl-log.json').write_text('invalid obsolete compatibility file')
    assert index.build(repo, False)['documents'] == 1


def test_stale_writer_cannot_publish_over_newer_release(repo):
    update = DocumentUpdate(doc('one'), ver('one'), [chunk('one', 0, 'original', A)])
    repo.apply_delta([update], [], replace(snap('one'), notes={'parent_snapshot': None}))
    repo.apply_delta([], [], replace(snap('two'), notes={'parent_snapshot': 'one'}))
    with pytest.raises(IndexMismatch, match='Another indexer'):
        repo.apply_delta([], ['one'], replace(snap('stale'), notes={'parent_snapshot': 'one'}))
    assert repo.snapshot().snapshot_id == 'two'
    assert repo.active_version('one') is not None


def test_cached_invalid_vector_is_rejected(pipeline, monkeypatch):
    stage, repo = pipeline
    stage()
    index.build(repo, False)
    original = repo.snapshot()
    with index.EmbeddingCache() as cache:
        cache.db.execute('UPDATE embedding_cache SET vector=?', (b'\0' * 16,))
        cache.db.commit()
    with pytest.raises(ValueError, match='invalid vector'):
        index.build(repo, False, rebuild=True)
    assert repo.snapshot() == original


def test_fatal_build_records_failure_report(pipeline, monkeypatch):
    stage, repo = pipeline
    stage()
    monkeypatch.setattr(index.ollama, 'require', lambda *a: (_ for _ in ()).throw(RuntimeError('model unavailable')))
    with pytest.raises(RuntimeError):
        index.build(repo, False)
    report = json.loads((index.INDEX_DIR / 'ingestion-failure.json').read_text())
    assert report['error'] == 'model unavailable'
    assert report['status'] == 'failed'


def test_duplicate_staff_identity_rejected(pipeline, tmp_path):
    stage, repo = pipeline
    stage()
    path = staff_file(tmp_path / 'staff')
    path.with_name('duplicate.json').write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='Duplicate'):
        index.build(repo, False, staff_dir=path.parent)


def test_retriever_rejects_chunking_mismatch():
    from test_retrieve import build
    from assistant.retrieve import Retriever
    with build() as repo:
        repo.apply_delta([], [], replace(repo.snapshot(), snapshot_id='changed', chunking_version='other'))
        with pytest.raises(IndexMismatch, match='chunking'):
            Retriever(repo)


def test_heading_without_carry_is_used_verbatim():
    assert index._merge_headings('', 'Mixing') == 'Mixing'


def test_name_harvesting_tolerates_a_source_disappearing(tmp_path):
    assert index._harvest_all({'fetched': [{'kind': 'page', 'path': str(tmp_path / 'missing.html')}]}) == ([], [], [], {})


def test_legacy_html_hash_migrates_only_line_ending_normalization(pipeline):
    import hashlib
    stage, repo = pipeline
    stage()
    source = index.CACHE / 'plaster.html'
    text = source.read_text() + '\n'
    source.write_bytes(text.replace('\n', '\r\n').encode())
    ledger_path = index.CACHE / 'versions.json'
    ledger = json.loads(ledger_path.read_text())
    ledger[URL]['content_hash'] = 'sha256:' + hashlib.sha256(text.encode()).hexdigest()
    ledger_path.write_text(json.dumps(ledger))
    report = index.build(repo, False)
    assert report['delta']['failed'] == 0
    assert repo.active_content_hashes()[URL] == 'sha256:' + hashlib.sha256(source.read_bytes()).hexdigest()


def test_immutable_source_never_accepts_legacy_hash_normalization(pipeline):
    import hashlib
    stage, repo = pipeline
    stage()
    source = index.CACHE / 'plaster.html'
    text = source.read_text() + '\n'
    source.write_bytes(text.replace('\n', '\r\n').encode())
    ledger_path = index.CACHE / 'versions.json'
    ledger = json.loads(ledger_path.read_text())
    ledger[URL].update(content_hash='sha256:' + hashlib.sha256(text.encode()).hexdigest(), source_path=str(source))
    ledger_path.write_text(json.dumps(ledger))
    assert index.build(repo, False)['delta']['failed'] == 1
