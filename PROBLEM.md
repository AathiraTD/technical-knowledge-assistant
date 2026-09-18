# The problem, and what was and was not built

## The problem

Lime Green manufactures lime mortars, plasters and renders for solid-wall and
heritage buildings. Their technical team answers the same questions repeatedly —
how much water per sack, what goes on lath, whether a product suits a substrate —
while the answers are already published across datasheets, product pages, a
knowledge base and an FAQ. The material is scattered, the documents are not
searchable as a body, and the team is the bottleneck.

The failure mode that matters is not "the assistant doesn't know". It is "the
assistant is confidently wrong about a number". A coverage figure quoted without
its thickness, a temperature limit dropped from a mixing instruction, or a
product recommended for a substrate it is not suitable for produces a failed
wall, and the person who acted on it has a photograph and an invoice. In this
domain, an answer that is 90 per cent right is worse than a refusal, because a
refusal costs a phone call and a wrong figure costs a re-render.

So the system is built around that asymmetry. Every claim is traceable to a
published passage, every number must appear verbatim in a passage the answer
cites, and the assistant refuses — with a hand-off — rather than reaching.

## What this is

A local-LLM technical knowledge assistant over Lime Green's published material.
94 documents, crawled once by sitemap and shipped with the repository. A
question is split by topic, gated against a routing table, matched against slot
vocabularies, and retrieved from a knowledge store filtered to the caller's
audience. A deterministic router — not the model — then picks one of seven
paths. The model runs on one of them, at temperature zero, over passages
delimited as data. Six checks run before anything prints.

Everything runs locally. No data leaves the machine.

## What was built

| | |
|---|---|
| Crawler | Sitemap-driven, robots-aware, rate-limited, content-hashed, with a version ledger |
| Extraction | Two heading detectors for PDFs, DOM-aware for HTML, quality recorded per document |
| Indexer | Structure-aware chunking, caveat tagging, name harvesting, atomic snapshot publish |
| Delta ingestion | Unchanged documents skipped entirely; changed ones superseded with history retained; withdrawn ones deactivated; every run recorded |
| Storage | `KnowledgeRepository` with two adapters behind it — SQLite ships and serves every answer in the transcript; PostgreSQL + pgvector passes the same contract against a real instance |
| Retrieval | Local embeddings, audience filtering in code before ranking — a `WHERE` clause in PostgreSQL, a row filter in Python in SQLite — authority ranking with recency |
| Router | Eleven-topic policy gate, eight slots, eight ordered decisions |
| Answering | Five paths plus two composites, six post-generation checks, hand-off rendering |
| Interfaces | A CLI and a web page, both over one library |
| Evaluation | **9/9 situations, 10/10 probes**, audience filter passing both ways, a threshold sweep, and 648 unit tests |

## What was not built, and why

**Authentication.** The audience set is asserted at the command line, not proved.
The filter itself is real — it is applied to rows inside the repository, before
anything is ranked and long before a prompt is built, and the evaluation
demonstrates a staff-tagged document being invisible to a public caller — but
nothing stops a caller asserting `--audience staff`. Identity is the first thing
a deployment adds, and it slots in above the retrieval call without changing it.
The web page is narrower than the command line: a request there may only narrow
the audience set the operator started the server with, never widen it, because a
query string is not a credential.

**A deployed PostgreSQL instance.** The adapter itself is no longer the gap it
once was: it is written against the same Protocol and the same eight tables,
putting the audience filter, the active-version join, the authority ordering and
the distance operator into one query, and it now executes — the repository
contract, the ingestion lifecycle, the publication lock and the concurrent-reader
tests all run against a real PostgreSQL 16 with pgvector, in CI and in a
container, and the evidence is recorded in
[`docs/knowledge-pipeline-verification.md`](docs/knowledge-pipeline-verification.md).
What is still absent is a deployment. `psycopg` publishes no Windows ARM64 wheel,
so on this build machine those 43 tests **skip** rather than run, the transcript
is produced on SQLite, and nothing has yet served a question from Postgres under
load. The adapter is built and verified against the contract; the operation of it
is not evidenced here.

**Vision.** Users want to photograph a wall and be told what to plaster it with.
The model family chosen has a vision sibling, so this is a capability question
that has already been answered — and it is still refused, by policy. There is no
labelled failure library to ground a reading against, diagnosing a wall is the
technical team's judgement, and a confident wrong visual diagnosis is the worst
output this system could produce. The roadmap for doing it properly is
documented; the current behaviour is to say it cannot see the image, quote what
the sheets do say about the symptom, and hand over.

**The re-crawl trigger.** Ingestion is now a true delta: a second crawl of an
unchanged site reprocesses nothing, a changed document supersedes the version it
replaces and keeps it, and a withdrawn one is deactivated rather than deleted.
What is still absent is the *trigger* — the pipeline runs by hand rather than
from a change hook.

**Queueing and load shedding.** A generation queue with a visible wait,
per-session rate limiting and extract-only degradation under load are designed
and documented, not built. Generation is one at a time per Ollama instance, so
that is the part that would actually bind under real demand.

Two things this used to disclaim have since been built, and the exclusion is
narrowed rather than quietly dropped. The **answer cache** exists — exact-key
rather than the template-keyed form the record designs, audience- and
snapshot-scoped, taking a repeated question from 40.78 s to 0.017 s.
**Concurrent serving** exists too, because the web page raised a database error
on every question until it did: the store is opened for cross-thread use and
serialised behind one reentrant lock, each answer pins a consistent read, and
correlation ids are per-context so two simultaneous questions cannot pick up
each other's trace. Neither was scope creep — the first fell out of measuring
latency, the second out of fixing a demo-breaking bug — but both make the old
sentence understate what ships.

**Safety data sheets.** Deliberately not indexed. They are controlled documents
that must be read whole and current, and quoting them in fragments is the wrong
behaviour, not a missing feature. The exclusion is on file and the assistant can
say why.

## What the evaluation shows

Every expectation in the harness is mechanical: a path taken, a source cited, a
refusal state, or a string that must or must not appear. The transcript is in
`eval/results/transcript.txt` and regenerates with `python -m eval.run`.

| | |
|---|---|
| Situations | 9 of 9 |
| Guardrail probes | 10 of 10 |
| Audience filter | staff material invisible to a public caller, visible to staff |
| Unit tests | 648 passed, 43 skipped, with every outbound socket blocked. The skips are the PostgreSQL-gated tests, which need `ASSISTANT_POSTGRES_DSN`; CI supplies one and runs them |
| Safety-critical coverage | 100% line and branch |

The threshold sweep runs over the six situations that reach retrieval, and
produced the result worth arguing about. The two questions the corpus cannot
answer retrieve **more** confidently than the two straightforward ones it can:

| Situation | Top score | Should |
|---|---|---|
| How much water does Solo need | 0.595 | answer |
| How many bags for 20 square metres | 0.615 | answer |
| The pot life of Duro, published nowhere | 0.717 | refuse |
| The U-value of Solo, published nowhere | 0.739 | refuse |

At 0.35, 0.45 and 0.55 the sweep reports the same thing: six answered, none
wrongly refused, two wrongly admitted. No threshold in the swept range separates
them, and none could: a question about the U-value of Solo is a well-formed
question about a real product, so the Solo datasheet genuinely is its nearest
neighbour. Abstention by distance alone would have admitted both. The relevance
gate refuses both, by observing that the asked-for term appears in no retrieved
passage. That is the measured case for the design, and it is the opposite of
what a similarity score is assumed to do.

## Known weaknesses

- Retrieval was measurably wrong before chunks carried their product name into
  the embedding: a question naming Solo returned a page about aerogel adhesive.
  Fixed, but it is a reminder that a passage which reads clearly to a person can
  be unfindable, and only measurement showed it.
- Caveat tagging is rule-based, so an unusually worded limit is missed.
- The abstention threshold does less work than its name suggests. The sweep
  shows it cannot separate the answerable from the unanswerable in this
  corpus; the relevance gate does that, and the threshold only catches
  genuinely off-topic questions.
- Every document in the corpus is public, so the audience filter is exercised
  against a synthetic fixture rather than real staff material.
- One 34-page guide yields passages whose headings are drawing references, so
  its citations read less well than a datasheet's.

## Running it

```
pip install -r requirements.txt
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
python -m assistant.indexing.index      # builds from the shipped cache; no network
python -m assistant.interfaces.cli        # ask a question
python -m assistant.interfaces.ui         # the same library behind a web page
python -m eval.run             # situations, probes, sweep, audience filter
python -m assistant.infrastructure.health     # is it actually able to answer?

pip install pytest coverage
python -m pytest -q tests/     # 648 tests, no Ollama, no network, no index
python -m coverage run --rcfile=.coveragerc -m pytest -q tests/
python -m coverage report --rcfile=.coveragerc
```
