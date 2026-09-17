# Container deployment

The production-shaped stack: the application, PostgreSQL with pgvector, and
Ollama. Three services because three are needed.

**For a demonstration, do not use this.** Use [`demo-runbook.md`](demo-runbook.md)
§1 — a local Ollama that already holds the models starts in seconds, where a
first `compose up` pulls about 4 GB of them. This document is the deployment
path; that one is the demonstration path, and they are different on purpose.

---

## What this document used to say

It is worth naming, because an earlier version of this file was wrong in ways
an assessor would find by typing one command, and the repository's own standard
is that documentation describes reality rather than intention.

It documented the **root** `docker-compose.yml` — a stale parallel stack that
runs as root, never builds an index, pins nothing, and is referenced by no CI
job and no other document. It quoted metric names (`tka_questions_total`,
`tka_generation_seconds`) that **do not exist**; the real families are
`assistant_*`. It gave a `--index-store postgres` flag that **is not a flag**.
It said the index "will be created and populated on first query", which is
**false** — nothing builds an index lazily, and a server started without one
refuses to start. And it recommended Docker Swarm and Kubernetes, which
`CLAUDE.md` explicitly rules out.

All of that is corrected below against the stack that actually exists.

---

## Which stack

| | `deploy/` — **use this** | root `Dockerfile` / `docker-compose.yml` — stale |
|---|---|---|
| Referenced by | CI, README, DECISIONS 19 | nothing |
| User | non-root, uid 10001 | root |
| Base | `python:3.13-slim-bookworm` | `python:3.11-slim` |
| Database image | `pgvector/pgvector:pg16`, pinned | `ankane/pgvector:latest`, unpinned |
| Ollama image | `ollama/ollama:0.12.3`, pinned | `ollama/ollama:latest` |
| Models | `model-init` pulls both, then exits | never pulled |
| Index | `index-init` completes before the UI starts | never built |
| Healthcheck | `python -m assistant.health` — readiness | `/health` — liveness only |
| Port | 8765, bound to 127.0.0.1 | 8000, bound to every interface |

The stale pair is left in the tree rather than deleted — removing tracked files
is the repository owner's decision — but nothing should be run from it.

## Prerequisites

- Docker 20.10 or later with Compose v2
- About 10 GB of disk for the models and the database volume
- Enough memory to hold a 3.4 GB generation model alongside a 639 MB embedding
  model. The build machine has 24 GB; that is comfortable, not a floor that has
  been measured.

## Configuration

```bash
cp deploy/.env.example deploy/.env      # then set POSTGRES_PASSWORD
```

`POSTGRES_PASSWORD` has **no default**, deliberately: `up` fails on the missing
variable rather than starting a database with a password someone committed.

| Variable | Default | What it does |
|---|---|---|
| `POSTGRES_USER` | `assistant` | |
| `POSTGRES_PASSWORD` | *(none — required)* | |
| `POSTGRES_DB` | `assistant` | |
| `ASSISTANT_PORT` | `8765` | host port, bound to `127.0.0.1` |
| `GENERATION_MODEL` | `qwen3.5:4b` | recorded in the index header |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | changing it means rebuilding the index |
| `EMBED_DIMENSIONS` | `1024` | must match the model |
| `ASSISTANT_POSTGRES_DSN` | set by compose | selects PostgreSQL at every entry point |
| `OLLAMA_HOST` | `http://ollama:11434` | service DNS, never `localhost` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | opt-in OTLP export; empty disables it |

Never commit `deploy/.env`. Inside the compose network, services are reached as
`db:5432` and `ollama:11434` — `localhost` inside a container is the container.

**The engine refuses to query an index built with a different embedding model**,
so changing `EMBED_MODEL` is safe and forgetting to rebuild is not silently
possible. It fails closed rather than returning confident nonsense.

## Starting it

```bash
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml exec app python -m assistant.health
```

Startup is ordered and each step must complete before the next begins:

1. `db` becomes healthy (`pg_isready` — readiness, not "the server answered").
2. `ollama` becomes healthy.
3. `model-init` pulls both models and exits. Separate from the app so a restart
   does not re-check a 4 GB download, and a failed pull shows as a failed
   service rather than as a slow start.
4. `index-init` builds the index against PostgreSQL and exits.
5. `app` starts, and its healthcheck is readiness rather than liveness.

The corpus is bind-mounted from `../data` rather than held in a named volume. A
named volume starts empty, so the indexer had nothing to index and the stack
came up serving an empty store — a failure that looked like a model problem and
was a mount problem. It also means the shipped embedding cache is the one used,
so a first build takes seconds rather than re-embedding 552 passages on CPU.

## Verifying it

```powershell
.\scripts\verify-docker-deployment.ps1          # image-level checks
.\scripts\verify-docker-deployment.ps1 -Full    # the whole stack
```

Measured on the build machine, 17 September 2026 — image-level checks:

```
  ok    docker daemon responding
  ok    deploy/compose.yaml is valid
  ok    image built as lime-green-assistant:verify
  ok    runs as uid 10001
  ok    application  /app/assistant/health.py
  ok    authored configuration  /app/config/routing.json
  ok    deployment schema  /app/db/schema.postgres.sql
  ok    repository contract  /app/tests/test_repository_contract.py
  ok    shipped embedding cache  /app/data/index/embeddings.db
  ok    crawled corpus  /app/data/cache
  ok    the application imports inside the image
  ok    hosted tracing stays off under a hostile environment
```

**Not claimed:** a full live-model `up` has not been completed on this machine.
The image build, the compose validation, the non-root user, the copied contents
and the in-image import are verified; the running stack is not.

### On a TLS-intercepting network

A container reaching `pypi.org` directly fails with
`SSLV3_ALERT_HANDSHAKE_FAILURE` while the host installs perfectly well through
its own proxied index. Pass that index in:

```bash
docker build -f deploy/Dockerfile \
  --build-arg PIP_INDEX_URL="$(pip config list | grep index-url | cut -d\' -f2)" \
  -t lime-green-assistant .
```

It defaults to PyPI, so an unproxied network needs no argument.

## Health, readiness and metrics

| Endpoint | Question | Codes |
|---|---|---|
| `/health` | is the process running | always 200 |
| `/ready` | could it answer a question | 200 or **503** |
| `/ready?format=text` | the same, as a table | 200 or 503 |
| `/metrics` | Prometheus exposition | 200 |

The compose healthcheck runs `python -m assistant.health`, which is the same
readiness check — not `/health`, which would keep a container in rotation while
it had no index, a mismatched index, or no reachable Ollama.

Real metric families, from `assistant/metrics.py`:

```
assistant_answers_total          assistant_outcomes_total
assistant_span_duration_ms       assistant_span_errors_total
assistant_retrieval_top_score    assistant_cache_lookups_total
assistant_cache_hits_total       assistant_checks_runs_total
assistant_check_failures_total   assistant_window_spans
assistant_window_truncated
```

That is the whole list, read off `# TYPE` lines from a running server rather
than from the source — the previous version of this document listed names that
did not exist, so this one is quoted from the endpoint.

```bash
curl -s http://127.0.0.1:8765/metrics | grep assistant_answers_total
curl -s http://127.0.0.1:8765/ready?format=text
```

## Operating it

```bash
docker compose -f deploy/compose.yaml logs -f app        # structured JSON on stderr
docker compose -f deploy/compose.yaml ps
docker compose -f deploy/compose.yaml exec app python -m assistant.trace
docker compose -f deploy/compose.yaml exec app python -m assistant.index    # incremental refresh
docker compose -f deploy/compose.yaml down               # keeps volumes
docker compose -f deploy/compose.yaml down -v            # discards them
```

Refreshing the index is incremental: unchanged documents are not re-extracted,
re-chunked or re-embedded, a changed one supersedes the version it replaces and
that version is retained, and a withdrawn one is deactivated rather than
deleted. A failed run cannot replace a good release.

### Backups

The database and the versioned source archive must be backed up **together** —
PostgreSQL holds identity, versions, chunks and provenance; the filesystem holds
the original bytes those rows point at. A backup of one without the other cannot
reproduce an answer.

```bash
docker compose -f deploy/compose.yaml exec db \
  pg_dump -U assistant assistant > backup.sql
tar czf corpus.tgz data/cache
```

## Scaling — what is true

Generation is the bottleneck and it is serialised: one Ollama instance produces
one answer at a time. Measured at 1, 3 and 5 concurrent questions, wall time
scales with the number of questions and throughput does not improve; retrieval
degrades too, because embedding the question queues behind generation on the
same server. Numbers in [`demo-runbook.md`](demo-runbook.md) §7.

So the first move under load is **not** more application replicas — they would
queue on the same model server. It is the serving layer the architecture already
describes: a generation queue with a visible wait, per-session rate limiting, and
extract-only degradation, which drops the compose path rather than a safety
check. After that, more Ollama capacity.

**No Kubernetes, no Swarm.** `CLAUDE.md` rules both out for this system, and
nothing measured here argues for either.

## Troubleshooting

**`up` fails immediately on `POSTGRES_PASSWORD`.** Intended. `cp deploy/.env.example deploy/.env` and set one.

**`index-init` exits non-zero.** Read its logs. A failed indexing run leaves the
previous release serving; it cannot corrupt a good one.

**`app` never becomes healthy.** Run the readiness check directly — it names the
failing check and its remedy:
`docker compose -f deploy/compose.yaml exec app python -m assistant.health`

**`IndexMismatch` on start.** `EMBED_MODEL` changed since the index was built.
Rebuild: `docker compose -f deploy/compose.yaml run --rm index-init`.

**A `COPY` fails during build with `not found`.** A path the Dockerfile copies is
excluded by `.dockerignore`. See the comment at the top of that file.

**SQLite cross-thread errors.** Only reachable when a threaded server opens the
store without `thread_safe=True`; `assistant/ui.py` passes it. In the container
the store is PostgreSQL and this does not arise.

## References

- [`demo-runbook.md`](demo-runbook.md) — startup, readiness, traces, latency, recovery
- [`knowledge-pipeline.md`](knowledge-pipeline.md) — ingestion lifecycle and scheduling
- [`DECISIONS.md`](../DECISIONS.md) entry 19 — why the release model is shaped this way
- [pgvector](https://github.com/pgvector/pgvector) · [Ollama](https://github.com/ollama/ollama)
