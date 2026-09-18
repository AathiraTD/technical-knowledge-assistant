# Bounded architecture review — trace model, vision seam, provenance

**Status: design only. No code has moved.** This document defines three things
before anything is implemented: the trace and span model that observability will
carry, the seam by which a photograph reaches the router, and the provenance
member that seam requires. It is deliberately narrow. It is not a redesign of
the answer engine, the session store or the repository boundary, all of which
are built, tested and argued for elsewhere — [`DECISIONS.md`](../DECISIONS.md)
and [`docs/architecture.md`](architecture.md) remain authoritative.

The review was prompted by a proposal to rebuild the system as a "conversational
RAG platform" in six phases. Reading the repository first changed the
conclusion: most of the proposed conversational layer already exists, one
finished component is wired to nothing, and the genuine gaps are narrower and
differently shaped than the proposal assumed. What follows is the evidence for
that, then the definitions.

---

## 1. What the review found

### 1.1 The conversational layer is built, and its exclusions are load-bearing

[`assistant/turn/session.py`](../assistant/turn/session.py) is a bounded, idle-expiring,
thread-safe session store: cookie-named with 128 bits of `secrets` randomness,
least-recently-used eviction, an id honoured only if this store minted it. It
carries exactly three slots — `substrate`, `location`, `exposure` — and
explicitly drops `calculation`, `photograph`, `symptom`, `cause_asked` and
`property_asked`.

That exclusion list is the requirement usually described as "multi-turn context
must be selective, not unlimited memory", and the module's own docstring gives
the failure mode for each dropped slot: a held `calculation` sends a later
lookup down router step 6 where the sum is refused; a held `photograph` appends
"I cannot see photographs" to an answer nobody attached anything to; a held
`cause_asked` pins the session to the diagnosis path permanently. The rule it
states — *carry what the person told us about their wall, never what the last
question happened to be asking for* — is a stronger and more defensible design
than a general `ConversationState` holding everything with a topic-reset
heuristic on top.

**Conclusion: do not replace this.** A richer state object would have to
re-derive these exclusions and would be likelier to get them wrong.

Follow-ups are handled by `Session.pending`: a question that produced an
ask-back is held, and the turn supplying the missing fact re-asks the original.
This is requirement "follow-up questions need first-class support", built at the
engine level rather than as UI message retention.

### 1.2 One finished component is wired to nothing

[`assistant/answering/vision.py`](../assistant/answering/vision.py) is 679 lines with 671 lines of
tests. It constrains the model to a JSON schema whose `attribute` field is an
enum of four slots, gates every observed value against
`config/vocabularies.json` so a model answering `"substrate": "Lime Green Solo"`
produces no slot at all, never reads the free-text fields in the resolver, and
fails closed to an empty `Perception` on any error. It implements decision
16.1's observation contract, its confidence bands and its profile resolver.

```
$ grep -rn "from .vision\|import vision" --include=*.py . | grep -v test
(no output)
```

Nothing outside its own tests imports it. There is no `--image` flag on the CLI
and no upload on the web page. This is the single largest gap between what the
repository contains and what it can demonstrate, and closing it is a wiring
problem, not a build.

### 1.3 Provenance exists, and has one honest hole

[`assistant/answering/answer.py`](../assistant/answering/answer.py) defines a `Provenance` enum —
`STATED`, `CARRIED`, `ASSUMED` — and a `SlotFact` pairing a slot value with
where it came from, each rendering a different sentence. The distinction is
real and was added to fix an observed defect: an answer reporting "external
(location), as you said" when *external* came from a question two turns earlier.

The enum's docstring already names the gap:

> Decision 16.1 wants `explicit / visually_observed / inferred / unknown` when a
> photograph can fill a slot. That is one more member here and one more row in
> `_PHRASE` — `assistant/answering/vision.py` resolves observations to a `carried` dict
> and nothing yet hands one to `Assistant.ask`, so an `OBSERVED` member would be
> a state nothing in this repository can produce and a claim the renderer could
> never make honestly.

This is correct discipline and it becomes a defect the moment the vision seam is
wired. See §4.

### 1.4 Observability emits but does not trace

[`assistant/infrastructure/observability.py`](../assistant/infrastructure/observability.py) has JSON-line
events, a correlation-ID `ContextVar` (correct for the `ThreadingHTTPServer`),
a `timed()` context manager, question fingerprinting instead of question text,
and a library logger that never configures itself. The privacy posture is
already decided and already right.

What is absent: any notion of a span, a parent, or a turn; durations only where
someone remembered to wrap a block; and no persistence — the events go to a
stream and nowhere else. `answer_log` persists route, snapshot, chunk ids and
the failed check, which is the audit trail, but it is one row per answer with no
timings, no scores, no slot provenance and no session.

### 1.5 The UI has the data and the wrong default view

[`assistant/interfaces/ui.py`](../assistant/interfaces/ui.py) already renders prior turns, a `Sources`
list and per-answer diagnostics, and manages the session cookie correctly
(`HttpOnly`, `SameSite=Lax`, `Path=/`). It opens on the diagnostics view. The
work is presentation and an upload control, not new plumbing.

### 1.6 Summary

| Requirement | Status | Work |
|---|---|---|
| Session, follow-ups, selective context | Built | None — preserve |
| Context provenance | Built, one member short | §4 |
| Inline citations + sources + six checks | Built | None |
| Vision perception | Built, unwired | §3 |
| Structured events, correlation id, privacy | Built | None |
| Spans, durations, persisted trace, metrics | Absent | §2 |
| Chat-shaped UI, upload, collapsed diagnostics | Absent | Slice C |
| Conversational evaluation | Absent | Slice D |
| Session persistence across restart | Absent | Non-goal — §6 |

---

## 2. The trace and span model

### 2.1 Why OpenTelemetry's shape and not its SDK

The attribute conventions and the trace/span/parent triple are the valuable part
of OpenTelemetry, and they are free. The SDK is six transitive packages in a
project whose decision 4 rests on five inspectable dependencies, and it buys
nothing until a collector exists. So: adopt the semantics now, keep the
dependency list unchanged, and leave an OTLP exporter as a shim that can be
added later without touching a call site.

Concretely, event fields follow OTel naming where a convention exists —
`gen_ai.request.model`, `gen_ai.usage.output_tokens`, `db.system` — and the
correlation id becomes the trace id rather than being replaced by one.

### 2.2 The identifiers

Four, in a strict containment hierarchy:

```
session_id     the conversation        SessionStore, already exists
  turn_id      one message in          new
    trace_id   one call to ask()       today's correlation_id, renamed in concept
      span_id  one stage within it     new, with parent_span_id
```

`turn_id` and `trace_id` are one-to-one today and will stay so for the
single-turn engine; they are separated because a retried or re-asked pending
question is the same turn and a different trace, and collapsing them would make
the ask-back cycle unreadable in the trace.

The existing `correlation()` context manager already binds per-context and resets
in a `finally`. The span stack extends it with the same primitive and the same
guarantee: a `ContextVar` holding the current span id, pushed and popped by a
context manager, so two concurrent questions on the threading server cannot
adopt each other's parent.

### 2.3 The span tree

One question produces this shape. Spans marked † are conditional.

```
answer                                    trace root, one per ask()
├── split_by_topic
└── part                                  one per topic the question splits into
    ├── cache_lookup
    ├── perception †                      only when images are attached
    │   ├── vision_model_call             one per image
    │   └── resolve                       observations → slots
    ├── slot_detection
    ├── retrieval
    │   ├── embed_question
    │   ├── search                        db.system, audience filter, k
    │   └── targeted_retrieval †          the named-product second pass
    ├── route                             the router decision
    ├── generation †                      Compose only
    ├── checks †                          Compose only; one event per check
    └── render
```

This maps onto call sites that already exist. `answer` and `part` are the
existing `obs.timed("answer")` and the per-part loop in `Assistant._ask`;
`route`, `cache`, `targeted_retrieval` and `missed_evidence` are already emitted
as events and become spans or span attributes. The new spans are `perception`,
`slot_detection`, `embed_question`, `search`, `generation`, `checks` and
`render`.

### 2.4 The span record

```
span_id            12 hex, as new_id()
parent_span_id     empty at the root
trace_id
turn_id
session_id         empty for CLI callers
name               from the tree above
started_at         ISO 8601 with offset, as the JSONFormatter already writes
duration_ms        integer
status             ok | error
attributes         event-specific, flat, JSON-serialisable
```

Every span emits exactly one event on completion, so the existing JSON-line
stream remains the transport and `JSONFormatter` needs only to promote the four
ids alongside the correlation id it already promotes.

### 2.5 What each span carries

Only fields the system can produce honestly. Nothing below is aspirational.

| Span | Attributes |
|---|---|
| `answer` | `question` (fingerprint), `question_words`, `audiences`, `parts`, `paths`, `refused` |
| `part` | `question` (fingerprint), `snapshot_id`, `cached` |
| `cache_lookup` | `hit` |
| `perception` | `images`, `observations`, `confirmed`, `discarded`, `cannot_determine` |
| `vision_model_call` | `gen_ai.request.model`, `error` |
| `resolve` | `slots`, `discarded` |
| `slot_detection` | `slots` (names only) |
| `retrieval` | `k`, `returned`, `top_score`, `audiences`, `per_document_cap` |
| `embed_question` | `gen_ai.request.model`, `dimension` |
| `search` | `db.system` (`sqlite` \| `postgresql`), `candidates`, `returned` |
| `route` | `path`, `step`, `reason`, `top_score`, `slots` |
| `generation` | `gen_ai.request.model`, `gen_ai.usage.output_tokens`, `passages`, `truncated` |
| `checks` | `passed`, `failed` (check numbers and text) |
| `render` | `sources`, `caveats`, `disclosure` |

**The privacy rule is unchanged and extends to spans.** No span carries question
text, answer text, or passage text. Fingerprints, counts, ids and scores only.
The single exception stays where it already is: `answer_log.question`, which is
a deliberate, documented, auditable retention decision and is not duplicated
into the trace.

### 2.6 Persistence

Spans go to a new `turn_traces` table behind the `KnowledgeRepository`
boundary, so both adapters carry it and the contract suite proves parity — the
same discipline `answer_log` already follows, including writing on its own
connection outside the read snapshot.

```
turn_traces
  id              identity
  session_id      TEXT NOT NULL DEFAULT ''
  turn_id         TEXT NOT NULL
  trace_id        TEXT NOT NULL
  span_id         TEXT NOT NULL
  parent_span_id  TEXT NOT NULL DEFAULT ''
  name            TEXT NOT NULL
  started_at      TEXT NOT NULL
  duration_ms     INTEGER NOT NULL
  status          TEXT NOT NULL DEFAULT 'ok'
  attributes      JSON in SQLite, JSONB in PostgreSQL
```

Indexed on `(trace_id)` and `(session_id, turn_id)`, which are the two access
paths: reconstruct one answer, and replay one conversation. Bounded by a
retention window rather than growing without limit, for the same reason the
session store is bounded.

`answer_log` stays as it is. It answers "what did this answer use"; the trace
answers "how did it get there, and how long did each step take". Merging them
would put timings into an audit table and audit semantics into a debugging one.

### 2.7 Metrics

Derived, not separately instrumented: every metric in the operational list is a
query over `turn_traces` and `answer_log`. A `/metrics` endpoint on the existing
standard-library server renders the Prometheus text format by hand — the format
is half a page, and a client library would be a dependency for string
formatting.

Exposed: answer/refusal/hand-off rate, route distribution, retrieval top-score
distribution, check-failure counts by check number, cache hit rate, and latency
percentiles per span name. Nothing that requires a counter the code does not
already produce.

### 2.8 What this gives the UI

"Why this answer?" renders from one trace: route and step, evidence count,
context used with provenance, checks passed, and time per stage. It is a read
of the persisted trace, not a second code path, which is why §2 is sequenced
before the UI slice.

---

## 3. The vision seam

### 3.1 The seam already exists

`Assistant.ask` takes `carried: dict | None`, documented as

> slots held from an earlier turn in the same conversation, **or read off an
> uploaded photograph**. The engine does not decide what is worth carrying and
> does not store it; a session or a vision step owns that, and hands it in.

and `vision.slots_from_images()` returns a `Resolution` whose `slots` field is
documented as

> a plain `dict` of slot to vocabulary value, suitable to pass straight to
> `Assistant.ask(question, carried=...)`. Only `CONFIRMED` attributes appear in
> it.

Both sides were designed against each other. The seam is not being invented
here; it is being connected, and the definition below is about what happens at
the join rather than about its shape.

### 3.2 The flow

```
bytes → observe() → Perception → resolve() → Resolution
                                                │
                              Resolution.slots ─┤→ merged under the session's
                                                │  carried slots, under the
                                                │  question's own slots
                                                ▼
                                    Assistant.ask(question, carried=…)
                                                ▼
                                    router, unchanged
```

### 3.3 The four rules at the join

**Precedence, most specific first.** This turn's question beats the photograph;
the photograph beats an earlier turn. The reasoning: someone who has just
uploaded a photograph of a stone wall and typed "it's brick" is correcting the
image, and the current sentence must win — the engine already merges `carried`
*under* whatever the question says. A photograph attached *now* is newer
evidence than a slot carried from a turn two questions ago, so it wins over the
session. Merge order is therefore session, then vision, then question.

**Only `CONFIRMED` attributes cross.** `Resolution.slots` already enforces this;
the seam must not reach into `attributes` for anything below the floor. A slot
that stays uncued is handled by decision 10's existing ask-back, which is
precisely the "no new answer route" property 16.1 claims.

**The photograph slot still fires.** Reading a substrate off an image is
perception; deciding what is wrong with the wall is not. The `photograph` slot
continues to add the cannot-see line and step 3 continues to send a cause
question to hand-off. Filling slots from an image does not license diagnosis,
and `cause_asked` remains absent from `VISION_SLOTS`.

**Failure reduces coverage, not safety.** `observe()` never raises; an
unreachable Ollama or a malformed response yields an empty slot dict and the
router lands exactly where it would have with no photograph. The seam adds no
`try` of its own, because swallowing at the join would hide the `error` field
the `Perception` carries for the hand-off.

### 3.4 What the session must not do

Vision-derived slots are **not** written back into the session's carried slots.

`CARRIED_SLOTS` means facts the person stated about their building. A slot
inferred from an image is a model's reading, and persisting it would make a
low-confidence visual inference indistinguishable from something the caller said
— on every subsequent turn, invisibly, and after the image has scrolled away.
The photograph is re-perceived for the turn it belongs to and its slots live for
that turn.

The cost is real: a follow-up question about the same wall does not inherit what
the photograph showed, so the person may be asked back for something the image
settled. That is the correct trade at this confidence level, and if it proves
annoying the fix is a fourth provenance state with its own expiry, not a quiet
promotion into `CARRIED_SLOTS`.

### 3.5 Boundary handling

The upload path is an external boundary and CLAUDE.md names the risks. Enforced
at the seam: a byte cap before decode, an allowlist of image media types by
content sniffing rather than by filename, a per-session image count cap, and no
use of the uploaded filename for anything — not storage, not logging, not the
hand-off. Decoding is delegated to whatever already reads bytes; nothing new is
introduced to parse image containers.

Images are not persisted. The failure library of 16.1 needs stored images,
labels and expert corrections, and that is partnership work with a retention
decision attached; storing photographs now because the code path is open would
be taking that decision by accident.

---

## 4. The provenance member

### 4.1 The defect the seam creates

`AnswerEngine._facts` decides between `STATED` and `CARRIED` by re-detecting
slots from the question text: a slot present in `decision.slots` but absent from
the current sentence is reported as `CARRIED`, which renders

> substrate: brick, **as you told me earlier in this conversation**

Hand a vision-derived slot to `ask(carried=…)` and that sentence prints about a
photograph. It is false — the person never said it — and it is the exact class
of error the provenance distinction was introduced to fix. Worse than the
original defect, because it attributes a model's inference to the caller.

So the member is not optional polish. **It ships in the same change as the
seam, or the seam ships a lie.**

### 4.2 The addition

```python
OBSERVED = "observed"
```

with one row in `_PHRASE`:

```python
Provenance.OBSERVED: "{value} ({slot}), from the photograph you sent",
```

`SlotFact.stated` must return `False` for `OBSERVED` — the caller did not state
it — which routes it into `assumptions`, under the heading every surface prints
as "Assumed". That heading is then wrong in the other direction: a reading from
an image is not a guess. The renderer therefore needs a third grouping —
*stated* (inside the answer), *observed* (its own line), *assumed* (under the
existing heading) — which is a change to `_finish` and to each surface's
renderer, and is the only place this addition ripples beyond one row.

### 4.3 How `_facts` learns the difference

It cannot infer `OBSERVED` by re-detecting from the question, because the whole
point is that the value is not in the question. The provenance must travel with
the value rather than be reconstructed from it.

The minimal form: `carried` gains an optional parallel mapping of slot to
origin, threaded through `Decision` to `_facts`, defaulting to `CARRIED` so that
every existing caller behaves exactly as it does today. The larger form — make
`carried` a `dict[str, SlotFact]` — is cleaner and touches the session store,
the engine, the router and the cache key, which is more churn than this slice
should carry. **Recommendation: the minimal form now, with the larger one noted
as the refactor if a fourth origin ever appears.**

The cache key must include the origin mapping, not only the values: the same
question with the same substrate from a photograph and from an earlier turn
produces different printed sentences, so they are different answers and must not
share a key. `_cache_key` already takes `carried`; it needs the origins too.

### 4.4 What stays out

`INFERRED` and `UNKNOWN` from 16.1's four-state list are not added. Nothing in
this repository infers a slot and `UNKNOWN` is the absence of a fact, which is
represented by the slot not being there. Adding either would repeat the mistake
the `OBSERVED` docstring avoided: a state nothing can produce.

---

## 5. Ownership, so nothing collides

| File | Change | Slice |
|---|---|---|
| `assistant/infrastructure/observability.py` | span stack, `span()`, id promotion in `JSONFormatter` | B |
| `assistant/knowledge/repository.py` | `record_spans` / `traces` on the boundary | B |
| `db/schema.sqlite.sql`, `db/schema.postgres.sql` | `turn_traces` | B |
| `assistant/knowledge/store/embedded.py`, `store/postgres.py` | implement both | B |
| `assistant/answering/engine.py` | spans at existing call sites; `carried` origins | A + B |
| `assistant/answering/answer.py` | `OBSERVED`, `_PHRASE` row, `_facts`, `_finish` grouping | A |
| `assistant/answering/vision.py` | **no change** — it is already correct | — |
| `assistant/turn/session.py` | **no change** — §3.4 | — |
| `assistant/interfaces/cli.py` | `--image`, observed grouping in the renderer | A |
| `assistant/interfaces/ui.py` | upload, observed grouping, then the chat rework | A, C |
| `tests/` | contract rows for `turn_traces`; seam and provenance tests | A, B |

`assistant/answering/vision.py` and `assistant/turn/session.py` carrying no change is the
review's main structural finding.

---

## 6. Non-goals, stated so they are decisions rather than omissions

**No `ConversationState` rewrite.** §1.1.

**No session persistence across restart.** It requires a store, a retention
period, a privacy decision and a deletion path, for a prototype whose sessions
expire in thirty minutes anyway. It stays a documented known limitation.

**No semantic or template cache work.** Decision 14 already records the
exact-key form as the weaker built form, with its reasoning.

**No OTLP exporter, no collector, no Grafana stack.** Tier 0 only. The stack
becomes a Compose profile when there is something worth putting a dashboard in
front of, and it consumes what the engine emits rather than defining it.

**No image persistence.** §3.5.

**No change to the six checks, the router, or the audience filter.** Nothing in
this review touches a safety boundary. The vision seam fills slots the router
already understands, and the trace observes rather than participates.

---

## 7. Sequence

1. **A — the vision seam.** `OBSERVED`, `_facts` origins, the merge, `--image`,
   upload. Highest demonstrable value, lowest architectural risk, and it is the
   slice that turns 679 tested lines from dead code into the multimodal story.
2. **B — tier-0 observability.** Spans, `turn_traces` on both adapters,
   `/metrics`. Sequenced second because the UI's "Why this answer?" reads from
   it and conversational evaluation asserts against it.
3. **C — the UI.** Chat layout, inline citations, source cards, diagnostics
   collapsed behind "Why this answer?".
4. **D — conversational evaluation.** Sequences asserting on route, slots,
   provenance and figures — never on prose, since decision 7's G4 note records
   that wording is not reproducible and evidence is.

Each slice ends with its tests, and slices B and D both end with the repository
contract run against a real PostgreSQL, per the definition of done.

## 8. Settled questions

Three questions were left open when this review was written: how long trace rows
live, whether the CLI writes them, and what a CLI caller's session id is. They
are settled here as recommendations with their evidence, because each has a
precedent in the repository that ought to be followed or explicitly departed
from rather than re-argued from first principles.

### 8.1 Retention: both a window and a cap, pruned on write

**The decision.** A fourteen-day window as the policy and a two-hundred-thousand
row cap as the backstop, both enforced inside the adapter's span write, on the
same connection that does the insert.

**How much there actually is.** The span tree in §2.3 produces twelve spans for
an ordinary single-part compose — `answer`, `split_by_topic`, `part`,
`cache_lookup`, `slot_detection`, `retrieval`, `embed_question`, `search`,
`route`, `generation`, `checks`, `render` — and a two-topic question repeats
everything below `part`, so twenty-two. A photograph adds `perception`,
`resolve` and one `vision_model_call` per image. Twelve is the number to plan
against; thirty is the bad day.

The volume is not hypothetical, because the audit table has been accumulating
for the life of the project and can be counted. `data/index/knowledge.db` holds
**196 rows in `answer_log`, 171 of them written on a single day** — a
development day of harness runs and manual questions. Their whole payload is
62,447 characters, about **319 bytes a row**, and that row carries the question
text, which §2.5 forbids a span to carry. A span row is ids, a name, a
timestamp, an integer and a small flat JSON object: the same order, 250 to 400
bytes.

So one heavy day is roughly two thousand trace rows and under a megabyte. That
is the reassuring half. The other half is that `data/index/knowledge.db` is
**4.1 MB and is a committed artefact** — a year of unpruned tracing at that
daily rate is on the order of a quarter of a gigabyte in the file that ships
beside the index, which would make the debugging table an order of magnitude
larger than the knowledge it explains. The bound is not about saving bytes
today; it is about the table having a shape rather than a direction.

**Why both, and not one.** A window alone is the honest expression of the policy
— a trace is for debugging the answer somebody is asking about this week — but
it does not bound a burst, and a load test or a runaway retry loop inside the
window is exactly what the bound exists for. A cap alone bounds the burst but
silently redefines retention as "however long two hundred thousand rows happen
to last", which is a number nobody can reason about when a colleague asks how
far back the traces go. The window is the stated policy; the cap is what makes
it safe, and it should be large enough that in normal operation it never fires.
At two thousand rows a day, two hundred thousand is about three months — the cap
binds only when something is wrong.

**Who prunes, and when.** On write, inside `record_spans`, following
[`assistant/turn/session.py:120,178-190`](../assistant/turn/session.py): the session store
sweeps expired entries from `open()` rather than from a timer, and its docstring
argues the case — a structure with no scheduler of its own prunes where it is
touched. The same argument holds here with one addition, that the span write
already opens its own connection outside the read snapshot the way `log_answer`
does ([`assistant/knowledge/store/embedded.py:700-734`](../assistant/knowledge/store/embedded.py)),
so there is a transaction to hang the delete on without disturbing a reader.

It should be amortised rather than run on every write: a stride — the delete
runs once every few hundred span batches — keeps the cost off the answering
path, since a `DELETE … WHERE started_at < ?` over an indexed column costs
nothing when it matches nothing but is still a write per answer if it runs every
time. `AnswerCache` bounds the same way, by evicting at the point of insertion
against `DEFAULT_MAX_ENTRIES` ([`assistant/cache.py:33,51`](../assistant/cache.py)).

**Why not a schedule.** The repository does have a scheduled entry point — the
refresh trigger — and hanging retention off it was the tempting option. It is
wrong for the same reason a ledger is the wrong thing to diff a crawl against
(decision 18): the thing being pruned and the thing doing the pruning would
drift. A deployment that answers questions daily but re-indexes monthly is the
expected shape, and in that shape a schedule-driven prune runs thirty times too
rarely. A store that is being written to is a store that can prune itself.

**Where it breaks, and the precedent deliberately not followed.** `answer_log`
is unbounded and has no prune — no `DELETE`, no window, no cap in either adapter
— and that is a departure, not an oversight. The two tables hold different
things: `answer_log` is the audit trail, it retains question text, and deleting
from it is a data-retention decision with a privacy argument attached that
nobody has taken. `turn_traces` holds no text, is for debugging, and its rows
stop being useful long before they stop existing. Bounding one and not the other
is the honest split, and it leaves `answer_log`'s own unbounded growth as a
stated open item rather than a solved one. The second precedent — the embedding
cache of decision 17, append-only, 5.6 MB in version control, whose entry
already concedes "the fix is a prune step" — is the counter-example this
recommendation exists to avoid repeating.

Three real costs. A quiet system keeps rows past the window until something
writes again, exactly as an idle `SessionStore` does; the bound is on growth,
not on age. A burst can overshoot the cap between strides. And after fourteen
days an `answer_log` row outlives its trace, so "why this answer?" (§2.8)
degrades to "what did this answer use" — the older, weaker question. Trace
retention therefore sets the horizon of the UI's explanation panel, which is
worth saying out loud before the panel is built.

**Schema implication.** One index, for the prune's own access path:

```sql
CREATE INDEX IF NOT EXISTS turn_traces_started_at ON turn_traces (started_at);
```

### 8.2 The CLI writes traces, with a `source` column

**The decision.** Write from every surface, and record which one, in a `source`
column on `turn_traces`.

**Why the question is not neutral.** The tempting answer — don't write from the
CLI, and the mixing problem disappears — is already refuted by what the audit
table contains. The harness builds an ordinary assistant with logging left on
([`eval/run.py:379`](../eval/run.py), against the `log: bool = True` default at
[`assistant/answering/engine.py:101`](../assistant/answering/engine.py)), so **evaluation answers
have been writing into `answer_log` all along, indistinguishably**. The
consequence is measurable: of 196 logged answers, 75 are refusals — a 38 per
cent refusal rate, over a question set deliberately loaded with the unanswerable
ones, S2 and S7 and the near-miss and far-miss probes. Quote that as the
system's refusal rate and it is simply wrong, and §2.7 proposes to expose
exactly that metric.

So the mixing is not a risk the trace would introduce; it is a defect the trace
gets the chance to fix. Not writing from the CLI would not fix it either, for a
different reason: the CLI is canonical, the transcript runs through it, and the
richest traces available — real route decisions, real check failures and real
latencies on questions nobody had asked before — would be the ones thrown away.
§2.8's "Why this answer?" would then be a UI-only feature explaining answers the
UI produced, which is the surface with the least interesting traffic.

A separate table is rejected on the same ground as merging `answer_log` and the
trace: two tables with identical columns and different names is a `WHERE` clause
wearing a schema, and it doubles the contract suite's rows across both adapters
to express one flag.

**What is gained and lost.** Gained: every metric in §2.7 becomes filterable, so
refusal rate and route distribution can be reported for real traffic and for the
evaluation set separately — and the second is genuinely wanted, because a change
in the harness's route distribution between two builds is a regression signal.
Lost: nothing, provided the metrics queries actually filter. They must, and that
is the one thing this decision depends on a later slice getting right — a
`source` column nobody filters on is worse than no column, because it looks like
the problem was solved.

**Where it breaks.** The column records what the caller claims, not what it is,
and nothing authenticates it — the same limitation the audience set already
carries. A misconfigured harness run tagged `ui` pollutes the numbers exactly as
today's untagged one does.

**Update: the parallel defect in `answer_log` is now fixed**, ahead of §2 rather
than after it, because the contaminated rate was being quoted while the fix
waited. `answer_log` carries the same `source` column with the same `unknown`
default, the same additive migration, and a contract row proving both adapters
round-trip it. The 196 rows written before it existed now read `unknown`, so the
38 per cent is attributable to an unrecorded mix rather than publishable as the
system's refusal rate — uncorrectable, as predicted, but no longer mistakable
for a measurement. §2's `turn_traces` inherits the settled shape rather than
re-arguing it.

**Schema implication.**

```sql
source TEXT NOT NULL DEFAULT 'unknown'   -- cli | ui | eval | test | unknown
```

No `CHECK` constraint on the values. The repository uses check constraints where
they protect a real invariant, and this is not one: an unrecognised source must
be recorded, not rejected, because a rejected insert would be an observability
write failing an answer — the failure
[`assistant/answering/engine.py:326-331`](../assistant/answering/engine.py) already goes out of its
way to prevent for `log_answer`. The `unknown` default is the value that makes
an unset source visible in a query rather than quietly filed under a real
surface.

### 8.3 Session id for CLI callers: empty, with a partial index

**The decision.** Empty, as §2.4 already drafted it, and the composite index
becomes partial so the empty rows do not sit in it.

**Why not a synthetic id.** It would be a claim the engine cannot support. The
CLI's interactive loop holds no session: it never constructs a `SessionStore`,
and every iteration calls `assistant.ask(question, audiences=audiences)` with no
`carried` argument at all ([`assistant/interfaces/cli.py:90-103,108`](../assistant/interfaces/cli.py)).
Consecutive CLI questions are *unrelated by construction* — no slot carries, no
`pending` re-asks — which is the deliberate split recorded at
[`assistant/answering/engine.py:122-126`](../assistant/answering/engine.py): the CLI stays
single-turn and stateless while the web page is neither. A synthetic
per-invocation id would group those unrelated turns under one conversation, so
"replay one conversation" would return a list of questions sharing nothing but a
terminal. That is a worse answer than returning nothing, because it is wrong
rather than empty, and the pattern it would create — inventing a state the code
cannot produce — is the one §4.4 refuses for `INFERRED` and `UNKNOWN`.

Uniformity between the two surfaces is the only thing gained, and it is
uniformity of shape over a real difference in kind. The surfaces differ because
one carries conversation and the other does not; flattening that in the trace
hides the distinction the trace exists to make visible.

**What empty does to the index.** `(session_id, turn_id)` with every CLI row
sharing the empty string gives the index one enormous low-selectivity prefix. A
lookup still resolves, because `turn_id` is unique and so is the pair — but the
empty rows are pure overhead in an index whose only purpose is "replay one
conversation", a query that by definition never asks for them. Make it partial:

```sql
CREATE INDEX IF NOT EXISTS turn_traces_session
    ON turn_traces (session_id, turn_id) WHERE session_id <> '';
```

Both dialects support this, and the repository already relies on partial indexes
in both — `index_snapshots_one_active` is one in
[`db/schema.sqlite.sql:120-121`](../db/schema.sqlite.sql) and in
[`db/schema.postgres.sql:133-134`](../db/schema.postgres.sql), which is also the
precedent for the two files expressing one constraint in slightly different
syntax.

**Where it breaks.** Reconstructing a CLI answer goes through `(trace_id)`
alone, so the CLI loses nothing it could have had; and if a session ever arrives
on the CLI — decision 10's ask-back would be more useful with one — that turn
starts writing a real id and joins the index without a migration. That is the
property making empty the reversible choice and a synthetic id the one that
would have to be undone.
