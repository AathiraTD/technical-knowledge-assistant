"""Durable scheduled ingestion. Invoke from cron or Task Scheduler once a minute.

Use --job <unique schedule period> to enqueue once; subsequent invocations with
no job drain due retries. Jobs run at least once; indexing is content-idempotent.
The SQLite queue is local to one ingestion host, regardless of serving backend.
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
