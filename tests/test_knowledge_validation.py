"""Publication is a gate, and this is the file that tries to get bad evidence past it.

Decision 19 says sources, extraction and vectors are validated *before* the
delta is activated. Stated that way it sounds like defensive programming; it is
not. Each test here corresponds to a way the index could end up serving
something that is not what the site published: a cached file edited after it
was hashed, a vector array of the wrong width or full of `NaN`, a cached
embedding that has been corrupted on disk, an unreviewed staff document, a
second indexer publishing over a release it never saw.

The shared assertion is always the same shape, and it is deliberately the
strong one: **the previous snapshot is still the snapshot**. Not that an
exception was raised — an exception raised after a partial write would satisfy
a weaker test and leave the store wrong. `repo.snapshot() == before`, or
`repo.snapshot() is None` where nothing was ever published, is what separates
validation from a message in a log.

Staff import gets the most space because it is the one path where material
enters the corpus without having been crawled. The lifecycle test walks
publication, re-import without change, revision, and withdrawal by deleting the
file, and asserts the audience boundary at the top of it: `staff://mixing` is
absent from `manifest()` and present in `manifest(("staff",))`. Approval is an
operator-controlled record rather than an authenticated signature, and the
rejection cases say what that record has to contain — approved, an approver,
a known audience, non-empty sections, and an identity that cannot collide with
a crawled URL.

The last two tests are a matched pair about hashing, and they only make sense
together. A source whose bytes differ from the ledger only by line endings is
migrated rather than treated as changed — that is tolerated for the mutable
cache file. The same difference is **not** tolerated once the ledger points at
an immutable archived original, where the hash is the content's identity and
any mismatch is a failure. Understanding which bytes are hashed before changing
hash behaviour is the standing rule here.

Fixtures are shared with `test_pipeline_regressions.py`, so the corpus is the
same disposable one page and the only substituted component is the model.
Nothing in this file establishes answer quality.
"""
import json
from dataclasses import replace

import pytest

from assistant.indexing import index
from assistant.knowledge.model import DocumentUpdate
from assistant.knowledge.repository import IndexMismatch
from test_pipeline_regressions import pipeline, URL, served_text
from test_repository_contract import A, doc, ver, chunk, snap


def test_source_hash_mismatch_keeps_original(pipeline):
    """A cached page edited after the crawl hashed it is counted as failed, not indexed."""
    stage, repo = pipeline
    stage()
    index.build(repo, False)
    (index.CACHE / 'plaster.html').write_text('tampered', encoding='utf-8')
    report = index.build(repo, False)
    assert report['delta']['failed'] == 1
    assert '5 to 6 litres' in served_text(repo)


def test_configuration_change_with_failed_source_cannot_publish(pipeline, monkeypatch):
    """A new embedding model cannot be adopted while a source is failing.

    Reprocessing under a changed configuration rewrites the whole corpus, so
    doing it while one document cannot be read would silently drop that
    document from the release it produces.
    """
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
    """Empty, wrong-width, `NaN` and all-zero vectors are each refused before publication.

    All four are returned as a successful embedding call, which is the point:
    the model answering without erroring is not evidence that what it returned
    can be searched.
    """
    stage, repo = pipeline
    stage()
    monkeypatch.setattr(index.ollama, 'embed', lambda *a, **k: vectors)
    with pytest.raises(ValueError, match='Embedding model'):
        index.build(repo, False)
    assert repo.snapshot() is None


def staff_file(directory, **overrides):
    """Write one reviewed staff document, with `overrides` making it invalid in one way."""
    directory.mkdir(parents=True, exist_ok=True)
    spec = dict(canonical_url='staff://mixing', title='Reviewed mixing note',
                audience='staff', approved=True, approved_by='Technical team',
                document_type='knowledge_base', sections=[dict(heading='Mixing', text='Approved internal mixing guidance.')])
    spec.update(overrides)
    path = directory / 'note.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    return path


def test_approved_staff_import_update_and_withdrawal(pipeline, tmp_path):
    """The staff import lifecycle, with the audience boundary asserted at the first step.

    Publish, re-import unchanged (no second version), revise (a second version,
    and the approver recorded in the version notes), then delete the file,
    which withdraws it. Throughout, the document is invisible to a public
    manifest and visible to a staff one.
    """
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
    """Six ways a staff document can be unacceptable, each failing the whole build.

    Not approved; no named approver; an audience that is not one of the three;
    no sections; a section with no text; and an identity colliding with a
    crawled URL. Rejecting the run rather than skipping the file is deliberate
    — a partial import is a corpus nobody can describe.
    """
    stage, repo = pipeline
    stage()
    directory = tmp_path / 'staff'
    staff_file(directory, **bad)
    with pytest.raises(ValueError):
        index.build(repo, False, staff_dir=directory)
    assert repo.snapshot() is None


def test_build_reads_atomic_crawl_release(pipeline):
    """The indexer follows the crawl pointer, not whatever `crawl-log.json` happens to hold.

    The obsolete compatibility file is deliberately filled with garbage: a
    build that still read it would fail rather than quietly pass.
    """
    stage, repo = pipeline
    stage()
    log = json.loads((index.CACHE / 'crawl-log.json').read_text())
    ledger = json.loads((index.CACHE / 'versions.json').read_text())
    (index.CACHE / 'release.json').write_text(json.dumps(dict(log=log, ledger=ledger)))
    (index.CACHE / 'crawl-current.json').write_text(json.dumps(dict(release='release.json')))
    (index.CACHE / 'crawl-log.json').write_text('invalid obsolete compatibility file')
    assert index.build(repo, False)['documents'] == 1


def test_stale_writer_cannot_publish_over_newer_release(repo):
    """An indexer publishing against a parent snapshot that has moved on is refused.

    The competing publisher would otherwise remove a document that the newer
    release still serves. The store is asserted unchanged afterwards, on both
    adapters.
    """
    update = DocumentUpdate(doc('one'), ver('one'), [chunk('one', 0, 'original', A)])
    repo.apply_delta([update], [], replace(snap('one'), notes={'parent_snapshot': None}))
    repo.apply_delta([], [], replace(snap('two'), notes={'parent_snapshot': 'one'}))
    with pytest.raises(IndexMismatch, match='Another indexer'):
        repo.apply_delta([], ['one'], replace(snap('stale'), notes={'parent_snapshot': 'one'}))
    assert repo.snapshot().snapshot_id == 'two'
    assert repo.active_version('one') is not None


def test_cached_invalid_vector_is_rejected(pipeline, monkeypatch):
    """A corrupted row in the embedding cache fails the rebuild instead of entering the index.

    Decision 17 makes the cache safe by keying on text, model and dimension —
    which says nothing about the bytes still being intact on disk.
    """
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
    """A build that dies leaves `ingestion-failure.json` behind, with the reason in it."""
    stage, repo = pipeline
    stage()
    monkeypatch.setattr(index.ollama, 'require', lambda *a: (_ for _ in ()).throw(RuntimeError('model unavailable')))
    with pytest.raises(RuntimeError):
        index.build(repo, False)
    report = json.loads((index.INDEX_DIR / 'ingestion-failure.json').read_text())
    assert report['error'] == 'model unavailable'
    assert report['status'] == 'failed'


def test_duplicate_staff_identity_rejected(pipeline, tmp_path):
    """Two staff files claiming one canonical URL fail the run rather than racing to win it."""
    stage, repo = pipeline
    stage()
    path = staff_file(tmp_path / 'staff')
    path.with_name('duplicate.json').write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='Duplicate'):
        index.build(repo, False, staff_dir=path.parent)


def test_retriever_rejects_chunking_mismatch():
    """A `Retriever` refuses to be built against an index chunked by a different version."""
    from test_retrieve import build
    from assistant.retrieval.retrieve import Retriever
    with build() as repo:
        repo.apply_delta([], [], replace(repo.snapshot(), snapshot_id='changed', chunking_version='other'))
        with pytest.raises(IndexMismatch, match='chunking'):
            Retriever(repo)


def test_heading_without_carry_is_used_verbatim():
    """With no parent heading to carry, the section path is the heading itself."""
    assert index._merge_headings('', 'Mixing') == 'Mixing'


def test_name_harvesting_tolerates_a_source_disappearing(tmp_path):
    """A source file gone from disk yields empty name lists rather than an exception.

    Name harvesting is a full pass every run, so it is the stage most exposed
    to a file the crawl recorded and something later removed.
    """
    assert index._harvest_all({'fetched': [{'kind': 'page', 'path': str(tmp_path / 'missing.html')}]}) == ([], [], [], {})


def test_legacy_html_hash_migrates_only_line_ending_normalization(pipeline):
    """A ledger hash that differs only by line endings is migrated, not treated as a change.

    The stored hash is then rewritten to the bytes actually on disk, so the
    migration happens once.
    """
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
    """The same tolerance is refused once the ledger points at an archived original.

    There the hash is the content's identity, so any mismatch is a failure —
    the pair with the test above, and the reason the tolerance is narrow rather
    than general.
    """
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
