"""Durable retries survive worker restarts and retain dead-letter evidence."""
import pytest
from assistant import pipeline


def test_queue_success_and_deduplication(tmp_path):
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('daily', now=0)
        queue.enqueue('daily', now=0)
        job = queue.claim(now=0)
        assert job['attempts'] == 1
        assert queue.claim(now=0) is None
        queue.finish(job, now=1, snapshot='release')
        assert queue.get('daily')['snapshot'] == 'release'
        assert queue.get('daily')['status'] == 'done'


def test_retry_restart_and_dead_letter(tmp_path):
    path = tmp_path / 'jobs.db'
    with pipeline.JobQueue(path) as queue:
        queue.enqueue('job', now=0)
        first = queue.claim(now=0)
        queue.finish(first, now=1, error='source unavailable')
    with pipeline.JobQueue(path) as queue:
        assert queue.claim(now=30) is None
        second = queue.claim(now=61)
        assert second['attempts'] == 2
        queue.finish(second, now=62, error='source unavailable')
        third = queue.claim(now=182)
        queue.finish(third, now=183, error='source unavailable')
        assert queue.get('job')['status'] == 'dead'
        assert queue.get('job')['error'] == 'source unavailable'
        assert queue.claim(now=10000) is None


def test_crashed_worker_lease_recovery_and_stale_completion(tmp_path):
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', now=0)
        old = queue.claim(now=0, lease_seconds=10)
        new = queue.claim(now=11, lease_seconds=10)
        with pytest.raises(RuntimeError, match='lease'):
            queue.finish(old, now=12, snapshot='stale')
        queue.finish(new, now=13, snapshot='new')
        assert queue.get('job')['snapshot'] == 'new'


def test_repeated_worker_crashes_eventually_dead_letter(tmp_path):
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', now=0)
        for now in (0, 11, 22):
            assert queue.claim(now=now, lease_seconds=10)
        assert queue.claim(now=33) is None
        assert queue.get('job')['status'] == 'dead'


@pytest.mark.parametrize('refresh,result', [(False, 0), (True, 0), (True, 1)])
def test_worker_only_indexes_successful_crawls(tmp_path, monkeypatch, refresh, result):
    calls = []
    monkeypatch.setattr(pipeline.crawl, 'run', lambda **k: calls.append('crawl') or result)
    monkeypatch.setattr(pipeline.index, 'build', lambda **k: calls.append('index') or {'snapshot': 's'})
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', refresh=refresh, now=0)
        outcome = pipeline.work(queue, now=0)
        assert outcome == (1 if refresh and result else 0)
        assert ('index' in calls) == (not refresh or not result)
        assert pipeline.work(queue, now=1) == 0


def test_worker_records_index_exceptions(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise ValueError('invalid vector')
    monkeypatch.setattr(pipeline.index, 'build', fail)
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', now=0)
        assert pipeline.work(queue, now=0) == 1
        assert queue.get('job')['error'] == 'invalid vector'


def test_cli_enqueues_runs_and_reports_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.index, 'build', lambda **k: {'snapshot': 's'})
    args = ['--queue', str(tmp_path / 'jobs.db')]
    assert pipeline.main(args + ['--job', 'nightly']) == 0
    assert pipeline.main(args + ['--status']) == 0
    assert 'nightly' in capsys.readouterr().out
    assert pipeline.main(args) == 0


def test_claim_failure_rolls_back_lock(tmp_path):
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.db.execute('DROP TABLE jobs')
        with pytest.raises(Exception):
            queue.claim(now=0)
        assert not queue.db.in_transaction


def test_staff_directory_and_partial_failure_reach_worker(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.index, 'build', lambda **k: seen.append(k) or {'delta': {'failed': 1}})
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('staff', staff_dir=tmp_path)
        assert pipeline.work(queue) == 1
        assert seen == [{'staff_dir': tmp_path}]
