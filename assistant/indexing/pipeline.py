"""Durable scheduled ingestion. Invoke from cron or Task Scheduler once a minute.

Use --job <unique schedule period> to enqueue once; subsequent invocations with
no job drain due retries. Jobs run at least once; indexing is content-idempotent.
The SQLite queue is local to one ingestion host, regardless of serving backend.

**A table in SQLite rather than a broker, because there is one ingestion host.**
Decision 19 makes that the explicit trade: Redis, RabbitMQ or Celery would buy
distribution this system has no second consumer for, and would cost a service
to run, monitor and secure beside the two — PostgreSQL and Ollama — that the
Compose stack already asks Lime Green to operate. What a broker genuinely
provides is durability across a crash, and a file on the same disk as the index
provides that too. If ingestion ever grows a second host the honest move is to
replace this module, not to distribute it: nothing above it depends on the queue
being local, and nothing in it pretends to be a broker.

The division of labour is deliberate and narrow. **Cron or Task Scheduler owns
cadence; this module owns durability.** There is no timer here, no daemon and no
loop — one invocation claims at most one job and exits, so a schedule that stops
firing stops ingestion visibly rather than leaving a wedged process that looks
alive. `--job` names the period (`daily-2026-09-18`, say) and enqueue is
`INSERT OR IGNORE` on that identity, so a scheduler that fires twice, or an
operator who runs the command again by hand, still enqueues once.

Delivery is **at least once, never exactly once**, and that is affordable only
because the work underneath it is content-idempotent: `assistant/indexing/index.py`
diffs against the content hashes of what is currently being served, so a job
that runs twice re-checks and reprocesses nothing. A repeat costs a crawl's
worth of conditional requests, not a duplicated version.

Failure has three stages rather than two. A job that raises goes back to
`pending` with an exponential backoff on `due` — roughly one minute, then two,
then four — so a source that is briefly unreachable is retried without a person
being woken. On the third failure it becomes `dead` with the error text kept on
the row, which is the dead-letter: the evidence stays in the queue where
`--status` prints it, rather than in a terminal that has since been closed.
Nothing is ever deleted, so "why did last night not index" is answerable in the
morning.

A crash is the third stage, and it is why every claim takes a lease. A worker
killed between claiming and finishing leaves the row `running` and nothing to
observe it; the lease — 24 hours, held in `due` — expires and the next
invocation reclaims the job. The window is deliberately long rather than tight,
because a legitimate full rebuild of this corpus is minutes and a lease shorter
than the work would hand the same job to a second worker while the first is
still doing it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import uuid
from pathlib import Path

from . import crawl, index


class JobQueue:
    """Ingestion work that survives the process that was doing it.

    One table, and the column list is the whole design. `id` is the schedule
    period rather than a generated key, so enqueue is idempotent against a
    scheduler that fires twice. `attempts` bounds retrying at three. `error`
    and `snapshot` keep the outcome on the row, so the queue is also the record
    of what happened: `--status` prints the snapshot a successful job published
    and the message a dead one failed with, long after the terminal has gone.

    `due` carries two meanings at once, which is worth knowing before reading
    `claim`: on a `pending` row it is the earliest time the job may run, and on
    a `running` row it is the moment the worker's lease expires. One column can
    serve both because a row is only ever in one of those states, and a single
    ordering column is what keeps the claim query a single statement.

    **Deliberately not a broker.** There are no topics, no consumer groups and
    no fan-out; `claim` hands out one job at a time across the entire queue.
    That is a fit for the real constraint rather than a simplification of a
    general design — two concurrent indexers would contend for the publication
    lock in `assistant/knowledge/store/postgres.py` and gain nothing, because
    the corpus is one crawl.

    The connection is opened once and closed by `__exit__`. Nothing here is
    thread-safe and nothing needs to be: the process claims one job and exits.
    """

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, refresh INTEGER NOT NULL, staff_dir TEXT,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            due REAL NOT NULL, lease TEXT, error TEXT NOT NULL DEFAULT '',
            snapshot TEXT NOT NULL DEFAULT '')''')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    def enqueue(self, job_id, refresh=False, staff_dir=None, *, now=None):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO jobs (id,refresh,staff_dir,due) VALUES (?,?,?,?)',
                            (job_id, refresh, str(staff_dir) if staff_dir else None,
                             time.time() if now is None else now))

    def claim(self, *, now=None, lease_seconds=86400):
        """The one job this invocation may run, leased, or nothing.

        Three things happen in one write transaction, and the order matters.

        First, dead-lettering. A row still marked `running` whose lease has
        expired is a worker that died holding it; if it has already been tried
        three times it becomes `dead` with `worker lease expired` recorded, so
        a job that crashes the indexer every time it runs stops being handed
        out rather than looping forever. A crash with attempts to spare is not
        dead-lettered — it simply becomes claimable again below, which is the
        crash recovery this queue exists for.

        Second, selection. A row qualifies when it is `pending` or `running`
        and `due` has passed — the two states share the column, so an expired
        lease and a due retry are the same condition. The `NOT EXISTS` clause
        is the interlock: **if any job anywhere in the queue holds an
        unexpired lease, nothing is claimed and `claim` returns `None`.** That
        makes the queue strictly one job at a time, which is what stops a
        scheduler firing every minute from starting a second indexer on top of
        a crawl that is still running. Ordering by `due` then `id` makes the
        choice deterministic rather than whatever SQLite scanned first.

        Third, the lease itself: a fresh `uuid4` token in `lease`, `attempts`
        incremented, and `due` pushed to `now + lease_seconds`. The token is
        what `finish` checks, so a worker whose lease expired and was reclaimed
        cannot later write its result over the newer attempt's.

        `lease_seconds` defaults to a day on purpose. The bound to beat is not
        "how long should a job take" but "how long could a legitimate run
        take", and a full rebuild of this corpus is minutes on a good day and
        much longer on a cold embedding cache. A tight lease would hand a
        running job to a second worker, which is the failure the lease is meant
        to prevent.

        `now` is injectable so the lifecycle tests can move time rather than
        sleep through a backoff.
        """
        now = time.time() if now is None else now
        try:
            self.db.execute('BEGIN IMMEDIATE')
            self.db.execute("""UPDATE jobs SET status='dead', error='worker lease expired'
                WHERE status='running' AND due<=? AND attempts>=3""", (now,))
            row = self.db.execute("""SELECT * FROM jobs WHERE status IN ('pending','running')
                AND due<=? AND NOT EXISTS (SELECT 1 FROM jobs WHERE status='running' AND due>?)
                ORDER BY due,id LIMIT 1""", (now, now)).fetchone()
            if row is None:
                self.db.commit()
                return None
            token = uuid.uuid4().hex
            self.db.execute("""UPDATE jobs SET status='running', attempts=attempts+1,
                lease=?, due=? WHERE id=?""", (token, now + lease_seconds, row['id']))
            self.db.commit()
            return dict(self.get(row['id']))
        except Exception:
            self.db.rollback()
            raise

    def finish(self, job, *, now=None, error='', snapshot=''):
        """Record the outcome, and decide whether this job gets another go.

        Success is `done` with the published snapshot id kept on the row, so
        the queue can be asked which release a given schedule period produced.
        Failure is `pending` with a backoff — one minute, then two, then four,
        doubling on the attempt count already incremented by `claim` — until
        the third attempt, which is `dead` with the error text retained.

        The `WHERE` clause is the interesting half. The update is conditional
        on the row still being `running` under **this** attempt's lease token,
        and a row count other than one raises rather than passing silently. A
        worker whose lease expired mid-job has already had its work reclaimed
        by someone else, and letting it write its result afterwards would
        overwrite a newer attempt with a stale verdict — quietly marking
        `done` a job that is at that moment being retried.
        """
        now = time.time() if now is None else now
        state = ('dead' if job['attempts'] >= 3 else 'pending') if error else 'done'
        with self.db:
            changed = self.db.execute("""UPDATE jobs SET status=?, due=?, error=?, snapshot=?, lease=NULL
                WHERE id=? AND status='running' AND lease=?""",
                (state, now + 60 * 2 ** (job['attempts'] - 1), error, snapshot, job['id'], job['lease']))
            if changed.rowcount != 1:
                raise RuntimeError('Worker lease no longer belongs to this attempt')

    def get(self, job_id):
        return self.db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()


def work(queue, *, now=None):
    """Run at most one due job, and treat a partial success as a failure.

    The body is short because the two stages it calls own their own safety:
    `crawl.run` leaves the previous source release intact when it fails, and
    `index.build` publishes a delta transactionally or not at all. What this
    adds is the judgement about what counts as having worked.

    Two conditions become exceptions rather than a logged warning. A non-zero
    return from the crawl means the fetch did not complete, and indexing on top
    of a half-fetched release would publish a snapshot that quietly lost
    documents the site still serves. A non-empty `delta.failed` means some
    sources did not survive extraction; those documents keep their previous
    version — nothing incorrect is being served — but the run has not delivered
    the corpus it was asked for, so it is retried rather than recorded as a
    success that nobody would look at again.

    **A failure is always reported through `finish`, never by leaving the row
    running.** That is what makes the backoff and the dead-letter reachable at
    all; a bare `raise` here would strand the job until its lease expired a day
    later. The return value is the exit code — zero when a job succeeded or
    there was nothing due, one when a job failed — so the scheduler's own
    failure reporting sees a bad night.
    """
    job = queue.claim(now=now)
    if job is None:
        return 0
    try:
        if job['refresh'] and crawl.run(refresh=True):
            raise RuntimeError('Crawl failed; previous source release retained')
        report = index.build(staff_dir=Path(job['staff_dir']) if job['staff_dir'] else None)
        if report.get('delta', {}).get('failed'):
            raise RuntimeError('Some sources failed indexing; retry required')
    except Exception as exc:
        queue.finish(job, now=now, error=str(exc))
        return 1
    queue.finish(job, now=now, snapshot=report['snapshot'])
    return 0


def main(argv=None):
    """One scheduler invocation: enqueue if asked, then drain one job.

    The shape is what makes this safe to call from cron every minute.
    `--job` is optional, and the two halves compose rather than exclude each
    other: an invocation that names a period enqueues it and then works it, and
    an invocation that names nothing works whatever became due in the meantime
    — a retry after a backoff, or a job abandoned by a crashed worker. So the
    schedule can be "enqueue nightly, invoke every minute", and retries need no
    schedule of their own.

    `--status` is the operator's view and prints the table as JSON, including
    the dead rows and the error each one carries. It deliberately returns
    before any work is claimed: reading the queue must never advance it, or
    checking on a stuck night would start another run of it.

    The default queue file sits beside the index rather than in a system-wide
    location, because the queue and the index it publishes into are one unit —
    copying the data directory takes the pending work with it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=index.INDEX_DIR / 'jobs.db')
    parser.add_argument('--job', help='Unique schedule period or manual job identity')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--staff-dir', type=Path)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    with JobQueue(args.queue) as queue:
        if args.status:
            print(json.dumps([dict(row) for row in queue.db.execute('SELECT * FROM jobs ORDER BY due,id')], indent=2))
            return 0
        if args.job:
            queue.enqueue(args.job, args.refresh, args.staff_dir)
        return work(queue)


if __name__ == '__main__':
    raise SystemExit(main())
