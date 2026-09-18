# Knowledge pipeline verification — 16 September 2026

> **Read the date before the numbers.** This record was measured on
> 16 September 2026 and describes the checkout of that day. The package was
> restructured into
> `assistant/{answering,indexing,infrastructure,interfaces,knowledge,retrieval,turn}`
> on 18 September 2026, and the suite has grown: **2324 tests are collected
> today** (`python -m pytest tests/ --collect-only -q`), against the 519 recorded
> below. Nothing here has been re-measured against the current tree, so treat
> every figure as historical evidence of a lifecycle that was verified, not as a
> statement about what passes now. Re-run the commands under *Environment and
> commands* before quoting any of it.

## Measured result

- **519 tests passed, zero skipped** *(as at 16 September 2026; 2324 are
  collected today and the pass count has not been re-measured)*, running in
  Python 3.13 on Linux with real SQLite and PostgreSQL 16 plus pgvector.
  Configured PostgreSQL failures fail tests.
- **100% line and branch coverage** across the measured libraries, including
  `crawl`, `extract`, `index`, `pipeline`, embedding/cache clients, domain models,
  repository interface, both storage adapters, backend factory and answering code
  — again, of that checkout and its then-flat module layout.
- **The CLI and the web page are inside the measurement, not outside it.** An
  earlier version of this line excluded them; `.coveragerc` does not. It sets
  `source = assistant` and omits exactly one path, `assistant/__init__.py`, the
  package docstring module with no behaviour to measure. No crawler or
  PostgreSQL exclusions remain either.
- CI enforces `missing_lines == 0` and `missing_branches == 0` in addition to
  `coverage report --fail-under=100`, so rounding cannot hide a missed branch.
  That gate is in the full-library job and is current. **The separate
  safety-critical job in `.github/workflows/tests.yml` is not**: it names seven
  flat paths — `assistant/router.py`, `assistant/answer.py`,
  `assistant/retrieve.py`, `assistant/store/embedded.py`, `assistant/engine.py`,
  `assistant/model.py`, `assistant/repository.py` — none of which survived the
  restructuring. It fails loudly rather than quietly, which is the one merciful
  detail: a path absent from `coverage.json` is appended as `not measured` and
  the step exits 1, so that job is now red for a naming reason rather than a
  coverage one. Repointing the list at
  `assistant/answering/{router,answer,engine}.py`,
  `assistant/knowledge/{model,repository}.py` and the retrieval and store
  modules is outstanding work.

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

One consequence of the 18 September restructuring reaches this section too: the
indexing and readiness commands in `deploy/compose.yaml` are current
(`assistant.indexing.index`, `assistant.infrastructure.health`), so the container
evidence above still describes a path that works, but `deploy/Dockerfile`'s
`CMD` names `assistant.ui` and no compose `command:` overrides it, so the `app`
container cannot serve. That was not exercised here — this verification indexed
into PostgreSQL and read the result back, and never started the UI — which is
exactly why it was not caught. See [`deployment.md`](deployment.md).

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
