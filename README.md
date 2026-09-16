# Lime Green Technical Assistant

A local-LLM technical knowledge assistant over Lime Green's published material. It retrieves passages from the company's own datasheets, guides and FAQ, cites every source by document name, and refuses — with a hand-off to the technical team — when the published material does not answer the question.

Built for the AEC Solution Architect (AI/LLM Developer) KTP take-home exercise, Birmingham City University with Lime Green Products.

## How it works, in a paragraph

A question is split by topic, gated against a routing table, and matched against slot vocabularies. What survives is embedded and retrieved from the knowledge store — active document versions only, filtered to the caller's audience in code rather than by prompt. A deterministic router — not the model — then picks one of five paths: route to a fixed referral, print a passage verbatim, compose over several passages, quote a published hand-off, or refuse. The model runs on the compose path only, at temperature zero, over retrieved passages delimited as data. Six checks run before anything prints, and a failure sends the part to a refusal that still carries whatever the site does publish.

## Status

The knowledge ingestion lifecycle is implemented. The architecture and decision
records distinguish working components from the remaining production roadmap.
See [the pipeline runbook](docs/knowledge-pipeline.md) and
[verification evidence](docs/knowledge-pipeline-verification.md).

Status is reported with a fixed vocabulary, so an intention is never mistaken for an implementation: **built and verified**, **built but weakly tested**, **partial**, **documented only**, **known limitation**.

| Area | Status | Evidence |
|---|---|---|
| Decisions | built and verified | 19 recorded with alternatives, costs and verification boundaries — `DECISIONS.md` |
| Crawler | built and verified | 94 documents, 0 errors; a re-run reports 94 unchanged |
| Extraction | built and verified | 37 PDFs probed; two heading detectors; per-document quality in the ingestion report |
| Indexer | built and verified | Container verification: 94 documents + 1 staff fixture, 552 passages, zero failures; unchanged second run reprocesses zero documents |
| Delta ingestion | built and verified | A second crawl of an unchanged site reprocesses **nothing**: 1.4s against 23.8s. Changed documents supersede and keep their history; withdrawn ones are deactivated, not deleted. Four-run proof in `DECISIONS.md` entry 18 |
| SQLite adapter | built and verified | The index builds, publishes atomically, and serves every answer in the transcript |
| Retrieval | built and verified | Audience filtering inside the repository, before anything is ranked — a `WHERE` clause in PostgreSQL, a Python row filter over the active chunks in SQLite; authority banding; index-mismatch refusal |
| Router | built and verified | 11-topic policy gate, 8 slots, 8 ordered steps |
| Six checks | built and verified | 30 unit tests across the six, each aimed at a failure they exist to catch — `tests/test_checks.py` |
| Answer engine | built and verified | Five paths plus the two composites the diagram names, 100% branch coverage, and a full evaluation transcript in `eval/results/` |
| CLI and web page | built and verified | Both over one library; the harness drives the library. `python -m assistant.health` reports ready |
| Evaluation harness | built and verified | **9/9 situations, 10/10 probes**, audience filter passing in both directions, threshold sweep. The two multi-source situations cover the brief's second test type. Transcript in `eval/results/transcript.txt` |
| Embedding model | **development default** | `qwen3-embedding:0.6b`. `eval/embedding_choice.py` is the benchmark that closes decision 6 |
| PostgreSQL adapter | built and verified | Repository contract, ingestion lifecycle, publication lock, concurrent reader and configuration migration, all against a real PostgreSQL 16 + pgvector in CI and in a container — [evidence](docs/knowledge-pipeline-verification.md). Verified against the contract, not deployed: the transcript is produced on SQLite, and the 43 PostgreSQL tests skip on a machine with no `ASSISTANT_POSTGRES_DSN` |
| Container deployment | partial | Image builds and runs ingestion as non-root against PostgreSQL with model doubles; Compose validates and initializes the index before the UI. Full live-model deployment is not claimed |
| Authentication | documented only | The audience set is asserted, not proved — the filter itself is real and tested. Over HTTP a request may only narrow the set the operator started the server with, never widen it; that is not authentication and is not described as such |
| Vision | documented only | Refused by policy, not by capability — decision 16 |
| Scheduled ingestion | built and verified | Conditional refresh, durable job deduplication, retries, crash recovery and dead-letter status; external scheduler supplies cadence |
| Approved staff ingestion | built and verified | Reviewed JSON sources, shared chunking/caveats, version history and audience enforcement |
| Answer cache | built and verified | Exact-key, audience- and snapshot-scoped; a repeated question drops from 40.78 s to 0.017 s. The template-keyed form decision 14 designs is still roadmap |
| Generation queue, rate limiting | documented only | Separate from the implemented ingestion queue. Concurrent serving is built and tested; queueing and degradation under load are not |
| Test coverage | built and verified | 100% line and branch target includes crawler, extraction, indexing, jobs, model clients and both stores; [current measurements](docs/knowledge-pipeline-verification.md) |

## Read this first

| If you want | Read |
|---|---|
| Why it is built this way — every decision, alternative and cost | [`DECISIONS.md`](DECISIONS.md) |
| What the system is — diagrams, component reference, corpus inventory | [`docs/architecture.md`](docs/architecture.md) |
| The analysis it came from — segments, jobs, question types, data, guardrails | [`docs/working-record.docx`](docs/working-record.docx) |
| The exercise itself | [`docs/brief.docx`](docs/brief.docx) |

## Requirements

- Python 3.11 or later
- [Ollama](https://ollama.com/download) installed and running locally, with two models pulled:
  ```
  ollama pull qwen3.5:4b            # 3.4 GB — generation, 256K context
  ollama pull qwen3-embedding:0.6b  # 639 MB — embeddings
  ```
  About 4 GB in total; allow time on a slow connection. `qwen3:4b-instruct` (2.5 GB) is the documented fallback if `qwen3.5` turns out to emit reasoning blocks — see `DECISIONS.md` entry 7. Tags were verified against the Ollama library on 15 September 2026. They are fixed in `config/`, recorded in the index header, and the engine refuses to run against a mismatch rather than returning confident nonsense.

## Running it

From a clean clone:

```
pip install -r requirements.txt
python -m assistant.index      # crawl is cached in the repo; builds the index offline
                               # a second run reprocesses only what changed
python -m assistant.cli        # ask a question
python -m assistant.ui         # the same library behind a web page
python -m eval.run             # nine situations, probe suite, threshold sweep
```

The crawled pages and PDFs ship in `data/cache/`, so the indexer runs without network access. Only the index is rebuilt locally, because it is tied to the embedding model on your machine.

`python -m assistant.crawl --refresh` revalidates sources using ETag and
Last-Modified. Without `--refresh`, available bodies are reused but discovery
still contacts the site. `python -m assistant.index` uses the cached source
release. Set `ASSISTANT_POSTGRES_DSN` to select PostgreSQL for all entry points.
See the [runbook](docs/knowledge-pipeline.md) for staff imports and scheduling.

## What ships in this repository, and on what footing

The crawled corpus ships with the code. `data/cache/` is 96 tracked files and
about 29 MB — 37 Lime Green PDFs (34 technical datasheets and 3 Warmshell system
guides) and the HTML of the pages that linked them, byte-for-byte as fetched —
and `docs/brief.docx` is the exercise brief itself. That
is a deliberate trade, and it is what makes the clean-clone run real: an assessor
with no network and no crawl still gets the same 95 documents, the same 552
passages and the same transcript, and can check any quoted figure against the
original file rather than taking the index's word for it. Shipping only the built
index would have removed the evidence and kept the claim.

The footing has to be stated plainly, because it is not a licence. This is
third-party material — Lime Green's published documents, and the brief itself —
included so the exercise can be assessed offline, not redistributed under any
grant. Copyright stays with its owners, this repository is private, and nothing
in it confers a right to republish. There is deliberately no `LICENSE` file:
adding one would imply a grant over content that is not ours to grant. Everything
derived from the corpus inherits the same footing — the index, the embedding
cache, and the harvested product, colour and merchant name lists. If this work is
ever made public, the corpus comes out first and the crawl becomes a build step
rather than a shipped artefact.

## Layout

```
README.md               what it is, how to run it
DECISIONS.md            why it is this way — 19 decisions
requirements.txt        five pinned dependencies

assistant/
  crawl.py              sitemap crawl, content hashing, the version ledger
  model.py              the domain model — storage-agnostic by design
  repository.py         KnowledgeRepository: the boundary the engine depends on
  store/embedded.py     the SQLite adapter behind it — the one that ships
  store/postgres.py     the PostgreSQL + pgvector adapter, same contract
  store/factory.py      which of the two a process gets, decided in one place
  extract.py            HTML and PDF to citable sections; harvest before stripping
  index.py              chunk, tag caveats, embed, publish a snapshot
  embedcache.py         content-addressed embeddings, so a rebuild is seconds
  ollama.py             two HTTP endpoints, retried; no client library
  retrieve.py           question embedding, synonyms, index-mismatch refusal
  router.py             policy gate, slot detection, the eight ordered steps
  answer.py             extract and compose, the six checks, hand-off, rendering
  engine.py             the assembled assistant, split by topic
  audience.py           a request may narrow the operator's audience set, never widen it
  cache.py              exact-key answer cache, scoped by audience and snapshot
  cli.py                canonical interface
  ui.py                 a web page over the same library, standard library only
  health.py             readiness: store, snapshot compatibility, model reachability

config/
  sources.json          the corpus boundary as executable configuration
  routing.json          the policy gate: eleven topics that never reach retrieval
  vocabularies.json     slot vocabularies, synonyms, deferral markers
db/
  schema.sqlite.sql     the assessment adapter — stdlib, ships, offline
  schema.postgres.sql   the deployment adapter — PostgreSQL + pgvector
data/
  cache/                original HTML and PDFs as fetched, shipped so it runs offline
  embeddings.db         the embedding cache, content-addressed by text and model
  index/                generated — the knowledge store and the ingestion report
eval/
  situations.json       nine situations with mechanical expectations
  probes.json           ten guardrail probes
  fixtures/             a synthetic staff-tagged document that must stay invisible
  run.py                the harness
  embedding_choice.py   the measurement behind decision 6
tests/
  test_checks.py        the six checks, against the failures they exist to catch
docs/
  architecture.md       diagrams, component reference, corpus inventory
  diagrams/             Mermaid sources for the three diagrams
```

## Tests

```
pip install pytest coverage
python -m pytest -q tests/                 # no Ollama, no index, no network
python -m coverage run --rcfile=.coveragerc -m pytest -q tests/
python -m coverage report --rcfile=.coveragerc
```

Tests replace the website transport and model calls with deterministic doubles.
SQLite always runs. PostgreSQL tests require a disposable pgvector database via
`ASSISTANT_POSTGRES_DSN`; they skip visibly only when it is unset, and configured
connection failures fail the suite. Install `psycopg[binary]==3.3.3` in a supported
Linux environment for the full suite.

On the Windows build machine, where no DSN is set, that is **648 passed and 43
skipped** — the 43 being the PostgreSQL-gated tests, which CI runs against a real
`pgvector/pgvector:pg16` service. A skip is printed, never swallowed.

The target is **100% line and branch coverage** across the knowledge and answer
libraries, including crawler, queue and both stores. CLI/UI presentation wrappers
are outside that measurement. CI runs the full suite against both stores and
checks that `missing_lines` and `missing_branches` are zero. Coverage is evidence
of exercised paths, not a guarantee that technical answers are always correct.

[Verification evidence](docs/knowledge-pipeline-verification.md) records the
measured result and the limits of the container/model checks.

## Notes for an assessor

- The corpus is Lime Green's published material only, crawled once by sitemap and cached in the repo. The boundary — what is in, what is out, and why — is in `DECISIONS.md` entry 1 and inventoried in `docs/architecture.md`.
- Every answer names its sources by document name. A figure that cannot be found word-for-word in a cited passage does not print.
- The assistant refuses rather than guesses, and a refusal still carries what the site publishes plus the technical team's published contact line. Refusals are a designed outcome, not a failure.
