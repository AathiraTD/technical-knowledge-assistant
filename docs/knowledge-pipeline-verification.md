# Knowledge pipeline verification — 16 September 2026

## Measured result

- **519 tests passed, zero skipped**, running in Python 3.13 on Linux with real
  SQLite and PostgreSQL 16 plus pgvector. Configured PostgreSQL failures fail tests.
- **100% line and branch coverage** across the measured libraries, including
  `crawl`, `extract`, `index`, `pipeline`, embedding/cache clients, domain models,
  repository interface, both storage adapters, backend factory and answering code.
- CLI/UI presentation wrappers and module launch guards are outside the library
  coverage measurement. No crawler or PostgreSQL exclusions remain.
- CI enforces `missing_lines == 0` and `missing_branches == 0` in addition to
  `coverage report --fail-under=100`, so rounding cannot hide a missed branch.

## Environment and commands

Test service: `pgvector/pgvector:pg16`; Python runner: `python:3.13-slim`;
Psycopg: `3.3.3`; Coverage: `7.16.0`. Dependencies use `requirements.txt`.
The local PostgreSQL test database ran on loopback port 55432. Each test created
and removed its own schema, separate from application data.

```sh
pip install -r requirements.txt pytest coverage 'psycopg[binary]==3.3.3'
# Set ASSISTANT_POSTGRES_DSN to a disposable local pgvector database.
python -m coverage run --rcfile=.coveragerc -m pytest -q tests/
python -m coverage report --rcfile=.coveragerc --fail-under=100
python -m coverage json --rcfile=.coveragerc -o coverage.json
```

All website responses and model calls in tests are deterministic doubles. The
tests exercise actual parsing, chunking, embedding-cache persistence, transactions,
SQL/vector queries, migrations, request isolation and job recovery. They do not
establish model accuracy or live-site availability.

## Container evidence

The production Dockerfile builds. An instance running as UID **10001** indexed
the shipped corpus into a disposable PostgreSQL schema with a deterministic
two-dimensional embedding model double, then indexed it again:

| Observation | Result |
|---|---:|
| Active documents | 95 (94 website documents + one evaluation fixture) |
| Active chunks | 552 |
| Failed documents | 0 |
| Unchanged website documents on second run | 94 |
| Reprocessed documents on second run | 0 |
| PostgreSQL readiness | true |

`docker compose -f deploy/compose.yaml config --quiet` passes with the required
password supplied. Compose now uses writable persistent source/index volumes,
honours the Ollama endpoint and model settings, and finishes index initialization
before starting the UI. A full Compose deployment with downloaded, running Ollama
models was **not** part of this verification.

## Scope and review

The changed lifecycle is documented in [the architecture/runbook](knowledge-pipeline.md)
and [decision 19](../DECISIONS.md#19-controlled-knowledge-releases-and-operational-evidence).
The existing extraction work in the user's working tree was preserved and included
in the tested checkout. Coverage describes that checkout, not an older commit.

Local diff review and whitespace checks accompany the tests. Earlier independent
agents supplied failing lifecycle/crawler tests; their execution quota ended before
final review, so no independent final-review claim is made.

Remaining separate work includes real-model quality evaluation, authenticated
audiences and approval workflow, distributed scheduling, vision, answer-serving
load controls and production operations. Complete test coverage is not a proof
of perfect technical-answer correctness.
