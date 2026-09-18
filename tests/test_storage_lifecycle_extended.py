"""Adapter parity for concurrent readers, configuration migrations and failures."""
from dataclasses import replace
import pytest

from assistant.model import DocumentUpdate, Caveat
from assistant.store import SQLiteKnowledgeRepository
from test_repository_contract import A, doc, ver, chunk, snap
from test_pipeline_regressions import pipeline, URL


def second_connection(repo):
    if hasattr(repo, 'dsn'):
        from assistant.store.postgres import PostgresKnowledgeRepository
        return PostgresKnowledgeRepository(repo.dsn, apply_schema=False)
    return SQLiteKnowledgeRepository(repo.path)


def test_reader_pins_caveats_and_metadata_across_concurrent_publish(repo):
    first = DocumentUpdate(doc('one'), ver('one'), [chunk('one', 0, 'old', A)],
                           [Caveat('one', 'limitation', 'old caveat', 'Scope')])
    repo.apply_delta([first], [], snap('before'))
    with second_connection(repo) as reader:
        with reader.read_snapshot() as release:
            assert release.snapshot_id == 'before'
            repo.apply_delta([replace(first, caveats=[Caveat('one', 'limitation', 'new caveat', 'Scope')])], [], snap('after'))
            assert reader.caveats('one')[0].sentence == 'old caveat'
            assert reader.snapshot().snapshot_id == 'before'
        assert reader.caveats('one')[0].sentence == 'new caveat'
        with pytest.raises(ValueError):
            with reader.read_snapshot():
                raise ValueError('cancelled request')
        assert reader.snapshot().snapshot_id == 'after'


def test_vector_dimensions_can_change_while_history_remains(repo):
    first = DocumentUpdate(doc('one'), ver('one'), [chunk('one', 0, 'old', A)])
    repo.apply_delta([first], [], snap('before'))
    repo.apply_delta([replace(first, chunks=[chunk('one', 0, 'new', [1, 0])])], [],
                     replace(snap('after'), embedding_dimensions=2))
    assert len(repo.versions('one')) == 2
    assert repo.retrieve([1, 0])[0].chunk.content == 'new'
    assert repo.counts()['chunks'] == 2
    assert repo.active_version('absent') is None
    repo.apply_delta([], ['absent'], snap('unchanged'))
    assert repo.active_version('one') is not None


def test_bootstrap_ignores_orphan_rows(repo):
    repo.publish([doc('one')], [ver('one'), ver('orphan')],
                 [chunk('one', 0, 'valid', A), chunk('orphan', 0, 'orphan', A)], snap('s'),
                 [Caveat('orphan', 'limitation', 'orphan', 'Scope')])
    assert repo.counts()['chunks'] == 1
    assert repo.caveats('one') == []


def test_postgres_readiness_uses_live_embedded_chunks():
    from conftest import _postgres
    repo = _postgres()
    try:
        assert not repo.health()['ready']
        first = DocumentUpdate(doc('one'), ver('one'), [chunk('one', 0, 'text', A)])
        repo.apply_delta([first], [], snap('s'))
        assert repo.health()['ready']
        repo.apply_delta([], ['one'], snap('empty'))
        assert not repo.health()['ready']
        repo.conn.close()
        assert repo.health()['connected'] is False
    finally:
        repo.close()


def test_full_index_delta_against_each_backend(pipeline, repo):
    from assistant.indexing import index
    from assistant.model import Excluded
    stage, _ = pipeline
    stage()
    first = index.build(repo, False)
    assert first['documents'] == 1
    stage(water='7 to 8')
    second = index.build(repo, False)
    assert second['delta']['changed'] == 1
    assert len(repo.versions(URL)) == 2
    assert '7 to 8 litres' in repo.retrieve([1, 0, 0, 0])[0].chunk.content
    repo.apply_delta([], [], snap('excluded'), excluded=[Excluded('safety.pdf', 'read whole')])
    assert repo.excluded()[0].url == 'safety.pdf'
    repo.apply_delta([], [], snap('cleared'), excluded=[])
    assert repo.excluded() == []


def test_sqlite_upgrades_legacy_crawl_columns(tmp_path):
    import sqlite3
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as old:
        old.execute('''CREATE TABLE crawl_runs (id INTEGER PRIMARY KEY,
            started_at TEXT, completed_at TEXT, documents_checked INTEGER,
            documents_new INTEGER, documents_changed INTEGER,
            documents_unchanged INTEGER, documents_failed INTEGER)''')
    with SQLiteKnowledgeRepository(path) as repo:
        columns = {row[1] for row in repo.db.execute('PRAGMA table_info(crawl_runs)')}
        assert {'documents_removed', 'snapshot_id'} <= columns


def test_postgres_missing_dependency_explains_installation(monkeypatch):
    import sys
    from assistant.store.postgres import PostgresKnowledgeRepository
    monkeypatch.setitem(sys.modules, 'psycopg', None)
    with pytest.raises(RuntimeError, match='psycopg is not installed'):
        PostgresKnowledgeRepository('unused')
