"""A local job queue earns its place only if it survives the crash it exists for.

Decision 19 declines a broker: there is one ingestion host, cron or Task
Scheduler supplies the cadence, and `assistant/indexing/pipeline.py` keeps the
jobs in SQLite. That is a defensible choice precisely as long as the queue does
the things a broker would be adopted for — deduplicate, back off, expire the
lease of a worker that died, and stop retrying something that will never work —
and every one of those is asserted here rather than assumed from the schema.

The tests are written against **injected clock values**, not `time.sleep`.
`claim(now=30)` returning nothing and `claim(now=61)` returning the job is the
backoff schedule stated as an assertion, and the whole file runs in
milliseconds because no test ever waits for a delay it is testing. A queue
whose retry timing could only be verified by waiting would not be verified.

Two failures matter more than the happy path. A worker that is killed mid-job
leaves a lease that must expire — and when it does, the *old* claim must be
refused at `finish()`, or a process that wakes up after its work was
reassigned silently overwrites the newer result with a stale snapshot id. And a
crawl that failed must not be followed by an index: `work()` is asserted to
call `crawl.run` and then *not* `index.build` when the refresh returned
non-zero, because publishing an index over a half-fetched crawl is how a failed
run corrupts the serving snapshot.

Nothing here reaches the network, a model or a real corpus: `crawl.run` and
`index.build` are replaced with recorders. This file proves the queue's
mechanics and the worker's ordering, and claims nothing about what indexing
itself does — that is `test_pipeline_regressions.py`.
"""
import pytest
from assistant.indexing import pipeline


def test_queue_success_and_deduplication(tmp_path):
    """Enqueueing the same job twice yields one claimable job, and finishing records its release."""
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
    """Backoff and the attempt count outlive the process, and the third failure dead-letters.

    The queue is reopened between the first and second attempt, so what is
    being asserted is durability rather than in-memory bookkeeping. The final
    `claim(now=10000)` proves a dead job is never retried again, and its error
    text is retained as evidence of why.
    """
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
    """An expired lease is reclaimable, and the original holder's `finish` is refused.

    Without the second half a worker that came back from the dead would
    overwrite the reassigned run's snapshot with its own stale one.
    """
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', now=0)
        old = queue.claim(now=0, lease_seconds=10)
        new = queue.claim(now=11, lease_seconds=10)
        with pytest.raises(RuntimeError, match='lease'):
            queue.finish(old, now=12, snapshot='stale')
        queue.finish(new, now=13, snapshot='new')
        assert queue.get('job')['snapshot'] == 'new'


def test_repeated_worker_crashes_eventually_dead_letter(tmp_path):
    """Crashing counts as failing: a job nobody ever finishes still stops being retried."""
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', now=0)
        for now in (0, 11, 22):
            assert queue.claim(now=now, lease_seconds=10)
        assert queue.claim(now=33) is None
        assert queue.get('job')['status'] == 'dead'


@pytest.mark.parametrize('refresh,result', [(False, 0), (True, 0), (True, 1)])
def test_worker_only_indexes_successful_crawls(tmp_path, monkeypatch, refresh, result):
    """A failed refresh crawl must not be followed by an index build.

    The three cases are: no refresh (index anyway, from the shipped cache), a
    refresh that succeeded (index), and a refresh that failed (do not index,
    and report the job as failed).
    """
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
    """An exception out of `index.build` becomes recorded job evidence, not a lost traceback."""
    def fail(**kwargs):
        raise ValueError('invalid vector')
    monkeypatch.setattr(pipeline.index, 'build', fail)
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('job', now=0)
        assert pipeline.work(queue, now=0) == 1
        assert queue.get('job')['error'] == 'invalid vector'


def test_cli_enqueues_runs_and_reports_jobs(tmp_path, monkeypatch, capsys):
    """The scheduled entry point: enqueue, report status, and run pending work."""
    monkeypatch.setattr(pipeline.index, 'build', lambda **k: {'snapshot': 's'})
    args = ['--queue', str(tmp_path / 'jobs.db')]
    assert pipeline.main(args + ['--job', 'nightly']) == 0
    assert pipeline.main(args + ['--status']) == 0
    assert 'nightly' in capsys.readouterr().out
    assert pipeline.main(args) == 0


def test_claim_failure_rolls_back_lock(tmp_path):
    """A claim that raises leaves no open transaction holding the queue's write lock."""
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.db.execute('DROP TABLE jobs')
        with pytest.raises(Exception):
            queue.claim(now=0)
        assert not queue.db.in_transaction


def test_staff_directory_and_partial_failure_reach_worker(tmp_path, monkeypatch):
    """A staff import directory is passed through, and a partly failed build is reported as failed."""
    seen = []
    monkeypatch.setattr(pipeline.index, 'build', lambda **k: seen.append(k) or {'delta': {'failed': 1}})
    with pipeline.JobQueue(tmp_path / 'jobs.db') as queue:
        queue.enqueue('staff', staff_dir=tmp_path)
        assert pipeline.work(queue) == 1
        assert seen == [{'staff_dir': tmp_path}]
