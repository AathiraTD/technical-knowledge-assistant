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
| Storage | `KnowledgeRepository` with a SQLite adapter; the same schema in PostgreSQL + pgvector |
| Retrieval | Local embeddings, audience filtering in the query, authority ranking with recency |
| Router | Eleven-topic policy gate, eight slots, eight ordered decisions |
| Answering | Five paths plus two composites, six post-generation checks, hand-off rendering |
| Interfaces | A CLI and a web page, both over one library |
| Evaluation | 7 situations, 10 guardrail probes, 16 unit tests, a threshold sweep, an audience-filter test |

## What was not built, and why

**Authentication.** The audience set is asserted at the command line, not proved.
The filter itself is real — it runs as a `WHERE` clause against rows, and the
evaluation demonstrates a staff-tagged document being invisible to a public
caller — but nothing stops a caller asserting `--audience staff`. Identity is
the first thing a deployment adds, and it slots in above the retrieval call
without changing it.

**The PostgreSQL adapter.** The schema is written, in the same eight tables with
the same version semantics, and the boundary exists so that swapping it changes
nothing above. But the submission runs on SQLite, so the Postgres adapter is
designed-for rather than exercised. Parity of design is not parity of testing
and it would be dishonest to present it as more than that.

**Vision.** Users want to photograph a wall and be told what to plaster it with.
The model family chosen has a vision sibling, so this is a capability question
that has already been answered — and it is still refused, by policy. There is no
labelled failure library to ground a reading against, diagnosing a wall is the
technical team's judgement, and a confident wrong visual diagnosis is the worst
output this system could produce. The roadmap for doing it properly is
documented; the current behaviour is to say it cannot see the image, quote what
the sheets do say about the symptom, and hand over.

**The re-crawl hook.** The crawler detects change by content hash and the design
for an event-triggered delta re-index is recorded, but it runs by hand.

**Caching of answers, queueing, and concurrency.** Designed and documented for
production, not built. At one user on one laptop there is nothing to cache.

**Safety data sheets.** Deliberately not indexed. They are controlled documents
that must be read whole and current, and quoting them in fragments is the wrong
behaviour, not a missing feature. The exclusion is on file and the assistant can
say why.

## Known weaknesses

- Retrieval was measurably wrong before chunks carried their product name into
  the embedding: a question naming Solo returned a page about aerogel adhesive.
  Fixed, but it is a reminder that a passage which reads clearly to a person can
  be unfindable, and only measurement showed it.
- Caveat tagging is rule-based, so an unusually worded limit is missed.
- The abstention threshold is one number chosen against a small sweep.
- Every document in the corpus is public, so the audience filter is exercised
  against a synthetic fixture rather than real staff material.
- One 34-page guide yields passages whose headings are drawing references, so
  its citations read less well than a datasheet's.

## Running it

```
pip install -r requirements.txt
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
python -m assistant.index      # builds from the shipped cache; no network
python -m assistant.cli        # ask a question
python -m assistant.ui         # the same library behind a web page
python -m eval.run             # situations, probes, sweep, audience filter
python tests/test_checks.py    # the six checks, no Ollama needed
```
