# Demo runbook

How to start the assistant, how to tell whether it is actually ready, how to
read back why it answered the way it did, what it costs under concurrency, and
how to recover inside a minute when something fails in front of an audience.

This is the operations document. **What** the system is: [`architecture.md`](architecture.md).
**Why** it is that way: [`DECISIONS.md`](../DECISIONS.md). Nothing here changes
the answer engine; it makes the existing behaviour startable, checkable and
inspectable.

Everything below was run on the build machine on 17 September 2026 and the
output is quoted as produced. Where a claim is not measured, it says so.

---

## 1. Before the interview — the four commands

Run these in order, the day before and again an hour before. The first one is
the whole check; the rest are the demonstration itself.

```powershell
.\scripts\start-demo.ps1 -CheckOnly      # verifies everything, starts nothing
.\scripts\start-demo.ps1                 # the page at http://127.0.0.1:8765/
python -m assistant.trace                # the last few turns, newest first
python -m assistant.trace <id>           # how one answer was produced
```

`-CheckOnly` is the one to trust. It fails on the same conditions the running
server fails on, and it fails *before* an audience is watching.

---

## 2. Prerequisites

| Needs | Why | If missing |
|---|---|---|
| Python 3.11+ | 3.13 on the build machine | `scripts/preflight.py` names the version it found |
| Six pinned packages | `pip install -r requirements.txt` | preflight prints the distribution names, which differ from the import names for three of them |
| Ollama running | embedding and generation | `ollama serve` |
| `qwen3-embedding:0.6b` | 639 MB — retrieval | `ollama pull qwen3-embedding:0.6b` |
| `qwen3.5:4b` | 3.4 GB — composition | `ollama pull qwen3.5:4b` |
| A built index | `data/index/knowledge.db` | `python -m assistant.indexing.index` — about two minutes |

The index is **not** in the repository and is not meant to be: it is tied to the
embedding model on the machine that built it. The *embedding cache* is shipped
(`data/embeddings.db`), so building the index re-embeds nothing — the measured
build was 116.9 s, of which embedding was 0.3 s across 552 cache hits.

A vision model is **not** a prerequisite. Images are handled by policy rather
than by capability ([decision 16](../DECISIONS.md#16-images-handled-by-policy-not-by-capability)),
so readiness ignores the vision model unless `ASSISTANT_VISION_DEMO` is set.

---

## 3. Startup

```powershell
.\scripts\start-demo.ps1 [-Port 8765] [-CheckOnly] [-BuildIndex] [-Rebuild]
                         [-VisionDemo] [-Audience public] [-NoBrowser]
```

Five steps in dependency order, stopping at the first failure with the command
that fixes it:

1. **Interpreter and packages** — `scripts/preflight.py`, standard library only.
   It runs before `assistant` is imported precisely so a missing dependency
   prints a sentence rather than a `ModuleNotFoundError` from inside a package
   the reader has never opened.
2. **Model server** — Ollama reachable, and the configured tags pulled. The tags
   are read out of the application's own configuration, never repeated in the
   script; two copies of a model tag is how a demonstration pulls one model and
   queries another.
3. **Knowledge index** — present, or offered as a build.
4. **Readiness** — `python -m assistant.health`, the table in §4.
5. **Serving** — `python -m assistant.interfaces.ui`.

A failure looks like this, and never like a traceback:

```
2/5  Model server
  FAIL  nothing is answering at http://127.0.0.1:19999

To fix:
  start the model server in another terminal:  ollama serve
  or install it from https://ollama.com/download
  if it runs elsewhere, set OLLAMA_HOST and re-run
```

---

## 4. Health and readiness

**Liveness and readiness are different questions, and the difference is not
pedantry here.** A process that starts is alive. It is *ready* only when the
store has an active snapshot, that snapshot's embedding model matches the
configured one, the models are pulled and there are passages to search — and
every one of those can be false while the process runs perfectly well.

| Surface | Question | Answer |
|---|---|---|
| `GET /health` | is the process running | static `OK`, always 200 |
| `GET /ready` | could it answer a question | JSON, **200 or 503** |
| `GET /ready?format=text` | the same, as the table | text, 200 or 503 |
| `python -m assistant.health` | the same, at a terminal | table, exit 0 or 1 |
| `python -m assistant.health --json` | the same, for a script | JSON |
| `GET /metrics` | Prometheus exposition | `assistant_*` families |

Measured output:

```
READY

application       ready                                  process is serving
database          ready                                  sqlite · data/index/knowledge.db
knowledge index   ready                                  95 documents · 552 passages
active snapshot   snap-ff83367ff65849eeab7650cd9c53d1a3  chunking structure-aware/1.0
embedding model   qwen3-embedding:0.6b                   matches index · 1024 dimensions
generation model  qwen3.5:4b                             pulled
vision model      qwen3.5:4b                             pulled · image demo off
Ollama            reachable                              http://127.0.0.1:11434
```

Not ready names the remedy, not only the fault:

```
NOT READY
...
To fix:
  generation model pulled
      ollama pull qwen3.5:4b
```

**What `/ready` deliberately does not publish.** It is unauthenticated, so the
store's filesystem path and database host are filtered out of the HTTP form —
an orchestrator needs to know *whether* the store is reachable, never where it
is. Credentials are redacted everywhere, including in error text: a DSN is the
one configured value that is a secret, and a readiness report is the
most-copied output in an incident. Both are covered by
[`tests/test_readiness_endpoint.py`](../tests/test_readiness_endpoint.py), the
first as a test on the filter rather than on the current contents of the
dictionary.

**Which faults each surface actually catches.** A fault present *before* start
does not need `/ready`: the engine refuses to construct on an absent or
mismatched index, so the process exits rather than serving. `/ready` exists for
faults that arrive afterwards — Ollama stopped, a model deleted, a variable
exported into a shell that is already serving. That is the window the tests are
written against.

---

## 5. Reading back one answer

Every response carries `X-Correlation-Id`, and that value is the trace id.

```powershell
python -m assistant.trace                       # the last turns, newest first
python -m assistant.trace 9ef580a27537          # one answer, as a tree
python -m assistant.trace --session pyQm_...    # the whole conversation
python -m assistant.trace 9ef580a27537 --all    # every recorded attribute
python -m assistant.trace 9ef580a27537 --json   # for a script or a test
```

Measured, from the benchmark run:

```
correlation  9ef580a27537
session      pyQm_WgdFM1_rkKtF3rLpA
turn         1
source       web  2026-09-17T12:57:23+00:00

--- python -m assistant.trace --session pyQm_WgdFM1_rkKtF3rLpA for the whole conversation

answer  28444ms  refused=False
  part  28416ms  path='compose'  cached=False
    cache_lookup  0ms  hit=False
    retrieval  877ms  product='solo'  top_score=0.5953219532966614  hits=5  documents=3
      embed_question  840ms  gen_ai.request.model='qwen3-embedding:0.6b'
      search  36ms  db.system='sqlite'
    slot_detection  1ms  slots=['property_asked']
    route  1ms  top_score=0.595...  path='compose'  step='8'  reason='several passages bear on the question'
    render  27536ms  path='compose'  refused=False
      generation  27533ms  gen_ai.request.model='qwen3.5:4b'
      checks  1ms  failed=[]
  verification  0ms  route='compose'  products=0
```

That is the whole answer in one screen: which product retrieval was told about,
what the top passage scored, how many documents it spanned, **which router step
fired and why**, whether any of the six checks failed, and where the 28 seconds
went. `--all` adds the rest — `gen_ai.usage.output_tokens=22`, `passages=5`,
`per_document_cap=3`, `checks count=6 failed=[]`, `audiences=['public']`.

### What is in the trace, and what is not

| Asked for | Where |
|---|---|
| correlation, session, turn, span ids | trace header and every span |
| route and router step, with the reason | `route` span |
| resolved intent, objective, missing facts | `query_understanding`, `candidate_discovery` spans |
| retrieval: product, top score, hit and document counts | `retrieval` span |
| candidate assessment | `candidate_discovery` span, when that path runs |
| verification result | `verification` span |
| timings per stage | every span's duration |
| vision observations, confidences, discards | `perception` / `vlm_perception` spans |

**Retrieved chunk ids are not on the span.** The `retrieval` span records
counts and the top score, not the identity of each passage; the cited sources
are on the answer itself and in `answer_log`. Naming that gap rather than
implying the trace answers it.

**Two privacy rules hold with no exception, and are enforced by test rather
than by scrubbing** (`tests/test_spans.py`): no span carries question text,
answer text or passage text — the question appears as a fingerprint
(`question='4bee49ac4c8a'`) and a word count. `answer_log.question` is the
single deliberate retention point and the trace does not duplicate it. **Model
chain-of-thought is never recorded anywhere**, and thinking variants are gated
out of model selection entirely ([decision 7](../DECISIONS.md#7-generation-model-qwen354b), G5).

**A CLI and not an endpoint, deliberately.** A trace is operator evidence, and
the surface serving anonymous callers is the wrong place for it: an endpoint
would need an audience gate, and an audience gate on a debugging aid is a new
access-control surface to get wrong. `/ready` qualifies for HTTP only because
what it publishes is operational state.

---

## 6. Hosted telemetry stays off

`langgraph` → `langchain-core` → `langsmith`, a client for a hosted tracing
service that exports whole runs — which here would mean the customer's
question, the retrieved passages and the conversation state.

It is closed **in code, by assignment rather than `setdefault`**, at two levels:
`assistant/__init__.py` before anything imports it, and `assistant/graph.py` at
the process-global level. `setdefault` was the first version and left a measured
hole — a parent environment carrying `LANGCHAIN_TRACING_V2=true` re-enabled
export, because not overriding is exactly what `setdefault` does.

`tests/test_privacy_tracing.py` launches the application under hostile
environment values and proves tracing stays off, **with a control test proving
those same values would switch it on without this application** — so the suite
cannot pass by guarding nothing. Verified passing in this branch.

No hosted telemetry dependency was added by this work. OTLP export exists as an
opt-in shim (`OTEL_EXPORTER_OTLP_ENDPOINT`, empty by default) and points
wherever the operator points it.

---

## 7. What it costs — measured, 17 September 2026

`python scripts/benchmark.py --base http://127.0.0.1:8765 --levels 1,3,5 --warm`

Distinct questions per level, none repeated, so no level is served from the
answer cache. Stage timings are read back from the spans rather than from a
second stopwatch. Raw data: [`eval/results/benchmark.json`](../eval/results/benchmark.json).

| Level | Wall | Per-answer median | Generation median | Retrieval median / max | Errors |
|---|---|---|---|---|---|
| Policy fast path, 5 at once | — | **0.31 s** | none | none | 0 |
| 1 question | 28.5 s | 28.5 s | 27.5 s | 0.88 s | 0 |
| 3 at once | 196.6 s | 196.6 s | 174.6 s | 3.5 s / 17.2 s | 0 |
| 5 at once | 438.8 s | 220.8 s | 81.0 s | 3.8 s / 26.3 s | 0 |
| Repeat of the level-1 question | — | **0.51 s** | cached | cached | 0 |

**The bottleneck is generation, and it is serialised.** At concurrency 1,
generation is 27.5 s of a 28.5 s answer — 97 per cent. Wall time then scales
with the number of questions rather than staying flat: 28.5 s → 196.6 s →
438.8 s for 1, 3 and 5. One Ollama instance generates one answer at a time, so
concurrency buys latency, not throughput.

**Retrieval degrades under load too, and that is the non-obvious finding.**
Retrieval is 0.88 s alone but its worst case rises to 17.2 s and then 26.3 s —
because embedding the *question* goes through the same Ollama that is busy
generating. The vector search itself never moved: 36 ms, in SQLite. So the
database is not the constraint at this corpus size and the model server is,
twice over.

**Nothing failed.** Error rate was zero at every level; the system queues, it
does not drop. What it does not have is a queue anyone can *see* — no visible
wait, no rate limit, no shedding.

**Two things stay fast under load, and they are the two the degradation design
rests on.** The policy fast path answers in 0.31 s with five in flight, because
it never reaches retrieval or a model. A cached answer returns in 0.51 s. That
is the measured basis for the roadmap serving layer degrading to extract-only
under load: it drops the compose path, never a safety check.

**What this is not.** Five concurrent questions on one laptop against one
Ollama. It identifies the bottleneck; it is not a capacity figure and no claim
about concurrent users follows from it.

---

## 8. Docker

Two stacks exist in this repository and only one of them is current. **Use
`deploy/compose.yaml`.**

| | `deploy/` — current | root `Dockerfile` / `docker-compose.yml` — stale |
|---|---|---|
| Referenced by | CI, README, DECISIONS 19 | nothing |
| User | non-root, uid 10001 | root |
| Images | pinned `pgvector/pgvector:pg16`, `ollama/ollama:0.12.3` | `ankane/pgvector:latest` |
| Index | `index-init` completes before the UI starts | never built |
| Healthcheck | `python -m assistant.health` | `/health`, which was liveness-only |
| Port | 8765 | 8000 |

The stale pair is left in place rather than deleted — removing tracked files is
the repository owner's call — but nothing should be run from it, and
`docs/deployment.md` documents it and is wrong in specifics (see §10).

### Verified on this machine

```powershell
docker compose -f deploy/compose.yaml --env-file deploy/.env.example config --quiet   # exit 0
docker build -f deploy/Dockerfile --build-arg PIP_INDEX_URL="<your index>" -t lime-green-assistant:demo .
```

**The image did not build before this work, and the reason is worth recording.**
`.dockerignore` excluded `tests/` and `*.db` while `deploy/Dockerfile` copies
both — `tests/` so the repository contract can run against the container's own
PostgreSQL, and `data/embeddings.db` so a first build reuses the shipped vectors
instead of re-embedding 552 passages inside a CPU container. The build failed
at `COPY tests/ ./tests/` with `failed to compute cache key: "/tests": not
found`. An exclusion that makes a `COPY` fail is not a smaller image, it is no
image. Both are now re-admitted with `!` rules and the reasoning is in the file.

**On a TLS-intercepting network, pass the proxied index.** A container reaching
`pypi.org` directly fails with `SSLV3_ALERT_HANDSHAKE_FAILURE` while the host
installs perfectly well; the Dockerfile anticipates this and takes
`PIP_INDEX_URL` as a build argument. Find yours with `pip config list`. On an
unproxied network no argument is needed.

### Running the stack

```powershell
cp deploy/.env.example deploy/.env     # then set POSTGRES_PASSWORD
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml exec app python -m assistant.health
```

`POSTGRES_PASSWORD` has no default, so `up` fails on the missing variable rather
than starting a database with a committed password. Inside the network, services
are `db:5432` and `ollama:11434` — never `localhost`, which inside a container is
the container. `model-init` pulls both models once and exits; `index-init` builds
the index and must complete before `app` starts.

**Not claimed:** a full live-model Compose run was not completed on this machine
within this work. What is verified is the image build, the compose validation,
the non-root user and the config. The first `up` pulls about 4 GB of models and
is not a demonstration path — for a demonstration, use §1.

---

## 9. Recovery — when something breaks in the room

Ordered by how often each one actually bites.

**The page is up but every answer is slow.** Expected, not broken: an unseen
question costs tens of seconds to minutes (§7). Ask something already asked —
the cache returns in ~0.5 s — or ask a policy question, which never reaches the
model. Do not restart anything; a restart makes the next answer slower, not
faster, because the model has to be read back off disk.

**Ollama has stopped.** `/ready` goes 503 naming it; `/health` will still say OK.
```powershell
ollama serve                         # in another terminal
ollama ps                            # what is resident
python -m assistant.health           # confirm READY before continuing
```
The server does **not** need restarting — it reconnects on the next question.

**A model is missing or was deleted.**
```powershell
ollama list
ollama pull qwen3.5:4b ; ollama pull qwen3-embedding:0.6b
```

**The server will not start, or answers 500.**
```powershell
# find and stop whatever holds the port
Get-NetTCPConnection -LocalPort 8765 -State Listen | Select-Object OwningProcess
Stop-Process -Id <pid> -Force
.\scripts\start-demo.ps1 -NoBrowser
```

**A stale listener survives `Stop-Process`.** Seen on this machine: the port
stays bound and the old code keeps answering, which presents as a *new endpoint
returning 404 while the source plainly contains it*. Confirm with
`Get-NetTCPConnection -LocalPort 8765`, and if the owning process no longer
exists, start on another port rather than fighting it: `.\scripts\start-demo.ps1 -Port 8791`.

**A code change is not taking effect.** Stale `__pycache__` — the source and the
bytecode can share an mtime under OneDrive, and Python then reuses the bytecode.
Note that `inspect.getsource` reads the `.py` and will show you the new code
while the old code runs, which is what makes this confusing.
```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

**The index is missing, empty or mismatched.** `/ready` names which.
```powershell
python -m assistant.indexing.index              # incremental; unchanged documents are skipped
python -m assistant.indexing.index --rebuild    # full, retaining version history
```
A failed indexing run cannot replace a good release — the previous snapshot
keeps serving.

**The conversation has gone strange — wrong substrate, wrong product carried.**
Start a clean chat. The session cookie is `HttpOnly`, so the page cannot clear
it and the button posts to the server:
```
http://127.0.0.1:8765/new
```
Or open a private window. `python -m assistant.trace --session <id>` shows what
was being carried, which is usually the explanation.

**Nothing is working and there are two minutes left.** Fall back to the CLI —
it uses the same library and needs no server:
```powershell
python -m assistant.interfaces.cli -q "How much water does Solo Onecoat need per bag?" -v
```
and to the recorded transcript, `eval/results/transcript.txt`.

---

## 10. Production seams — named, not claimed

| Seam | Today | What production adds |
|---|---|---|
| **Conversation state** | `InMemorySaver`, single process. **Does not survive a restart and is not shared between processes.** | Durable PostgreSQL checkpointing. Blocked, not deferred: `langgraph-checkpoint-postgres` 3.0.1 needs `langgraph-checkpoint<4` and `langgraph==1.2.11` needs `>=4.1.0`. `checkpointer_for()` raises rather than falling back, because a deployment that believed its conversations were durable and silently lost them would be worse than one that refuses to start. Re-check when the pin lifts. |
| **Model serving** | Ollama, one instance, generation serialised. Adequate for a demonstration and measured in §7. | Not a migration to vLLM on present evidence — the blocker is a single serialised instance, so the first move is a generation queue with a visible wait and per-session rate limiting, then horizontal Ollama or a batching server. Measure before migrating. |
| **Serving under load** | No queue, no rate limit, no shedding. Zero errors at 5 concurrent, but latency scales linearly. | Generation queue, per-session rate limit, extract-only degradation. Coverage degrades; safety does not. |
| **Identity** | The audience set is **asserted**, not authenticated. Over HTTP a request may only narrow what the operator started the server with. | Authenticated session resolving to an audience set. The filter itself is real, in code, before ranking, and tested in both directions. |
| **PostgreSQL** | Passes the repository contract, the ingestion lifecycle, the publication lock and the concurrent-reader tests against a real pgvector in CI. **Never operated**: the demo runs on SQLite and no Postgres instance has answered a question outside a test. | Operate it. Contract-verified is not production-proven. |
| **Vision** | Refused by policy. Detected, declared, handed off. | Perception → observations → profile → existing router. Needs a labelled failure library first; the model is the easy part. |
| **Retrieved chunk ids in traces** | Counts and top score only. | Per-passage identity on the retrieval span. |
| **Answer cache** | Exact-key. 28.5 s → 0.51 s measured. | Template-keyed, as decision 14 designs. Identical on safety, far better on hit rate. |

---

## 11. Known test failures in this branch

Pre-existing, unrelated to this work, and **not fixed here because they sit in
core conversation and UI semantics** that this branch was scoped to leave alone.
Recorded so nobody mistakes them for new:

- **`tests/test_ui_server.py` — 11 failures.** The page became a JavaScript
  client rendering from `/ask`; these tests still assert server-rendered HTML
  (`'Earlier in this conversation' in body`). The page itself works — the
  assertions are stale, not the behaviour.
- **`tests/test_vision_seam.py` — 1 failure.** Same cause.
- **`tests/test_integration_core5.py` — 17 errors.** The fixture opens the
  repository without `thread_safe=True` and reproduces the SQLite cross-thread
  error that the shipping UI already avoids by passing it. A test-side defect.

  **This file also does three things the rest of the suite does not**, and they
  are worth knowing before a demonstration rather than after. `setUpClass` calls
  `SQLiteKnowledgeRepository()` with no path, which is **the real
  `data/index/knowledge.db`** — not a temporary one. It then calls **real
  Ollama**, so single tests take 87 s and 143 s. And because the engine records
  spans, **running it writes test turns into the database the demonstration
  serves from**: they appear in `python -m assistant.trace` with `source
  unknown`, interleaved with real ones.

  None of that is dangerous — spans are append-only and the index is not
  modified — but it does mean **the trace list is noisy after a full test run**.
  Before demonstrating, either ask a fresh question and use its correlation id,
  or read the list and ignore the `unknown` rows. It is also why CI cannot run
  this file: `CLAUDE.md` requires deterministic CI independent of live models,
  and on a machine with no index or no Ollama `setUpClass` raises and every test
  in the class errors — which is exactly the 17 errors above.

- **`tests/test_diagnosis_capture.py` — 5 failures.** Date-dependent: the
  capture writes a case under one day's directory and the test reads it from
  another, so it fails whenever the two disagree
  (`FileNotFoundError: ...\2026-09-16\<uuid>.json`). Nothing to do with this
  work; last touched by "Phase 2 Component 1: Diagnosis Hand-off Logging".

### The reconciliation

| | Baseline (`60c9500`) | After this work |
|---|---|---|
| passed | 1368 | **1388** |
| failed | 17 | 26 |
| errors | 17 | **0** |
| skipped | 85 | 85 |
| **not passing** | **34** | **26** |

The failure count rose and the number of failing tests fell, which needs the
explanation rather than the headline. All 17 baseline *errors* were
`test_integration_core5`: every test in the class errored because `setUpClass`
raised when Ollama was unavailable. With Ollama running, `setUpClass` succeeds,
so those 17 errors became 8 passes and 9 failures. The 17 baseline failures —
5 `test_diagnosis_capture` + 11 `test_ui_server` + 1 `test_vision_seam` — still
fail, identically and by the same names.

`1368 + 12 + 8 = 1388` exactly: the twelve tests this work adds, plus the eight
integration tests that now run instead of erroring. **No new failure, and no
existing failure changed.** `git diff 60c9500..HEAD --name-only` touches none of
the failing files.

This work adds **12** passing tests — ten in
[`tests/test_readiness_endpoint.py`](../tests/test_readiness_endpoint.py) and
two in [`tests/test_trace_reader.py`](../tests/test_trace_reader.py) — and
changes no failure.

Two existing trace-reader tests were edited rather than left alone, and that is
a deliberate behaviour change rather than a test bent to fit: they asserted on
`splitlines()[0]`, and the single-turn view now opens with the correlation,
session and turn ids. They find the tree instead, and their assertions are
otherwise untouched.
