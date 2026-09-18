# Reliability and Design Evolution

This document traces how the system evolved through measured failures, root-cause investigation, focused fixes, and validation. Each change followed an observed failure that either threatened safety or coverage, a deliberate diagnosis of the root cause, and a scoped fix tailored to that cause alone.

**Core principle**: LLMs provide language intelligence; deterministic software provides control; retrieval provides evidence. **Local mode changes infrastructure, not safety semantics.**

---

## Evolution timeline

| Stage | Problem | Root cause | Change | Outcome |
|-------|---------|-----------|--------|---------|
| **1. Trusted state vs transcript** | Prior assistant answers recycled as context; generated figures contaminated later turns | Transcript text is not trusted state; prior answers should not re-enter as evidence | Preserve user questions only; store trusted slots in LangGraph checkpoint; retrieve from index | History-free citation validation |
| **2. Compound-query evidence binding** | Correct evidence retrieved, but model attached claims to wrong passages | Model-generated association between claim and passage not verified | Introduce property-evidence binding; promote deterministically bound passages for composition | Evidence membership verified before ranking |
| **3. Product registry pollution** | Generic editorial links ("lime plaster") entered product registry; false refusals on common sentences | Broad link harvesting without capitalization filter | Harvest only names starting with capital letters; preserve all genuine product links | Product vocabulary clean; name checks valid |
| **4. Product-scope verification** | Answer cited correctly but contradicted evidence; wrong product claim slipped through | Citation correctness ≠ product-scope correctness; Check 6 alone insufficient | Introduce Check 7: explicit product-scope resolution with real-name enforcement | Wrong-product refusals caught before output |
| **5. Compose evidence membership** | Ultra-only set correct, FAQ-only wrong, FAQ+Ultra produced refusal | Lexical attractor (FAQ heading matching question) preferred over deterministic binding | Scope composition evidence to the named or carried product, and to passages that name it (`scoped_evidence()`) | Membership-level control without breaking retrieval |
| **6. Vision substrate contract and provenance** | Vision observations indistinguishable from user statements; substrate uncalibrated from photograph | VLM observations treated as facts; loss of provenance attribution | Constrain to typed observations; vocabulary-filter values; distinguish OBSERVED from STATED | Observations explicit; inferences traced; confidence gates work |
| **7. Ollama context consistency** | Local generation/vision models requested different context sizes; inference instability | Configuration split; no unified context baseline | Single `NUM_CTX` variable; propagate to all model calls | Local inference stability hardened |
| **8. Conversation state security** | Audience set not in cache key; staff answer could reach public caller | Cache key missing audience dimension | Include audience set in exact-key cache | Audience isolation verified in test |
| **9. Question-carrying transcript** | Prior assistant answers leaked back into model context | Transcript history included generated answers | Carry only user questions in history block | No generated-answer feedback loops |
| **10. Evaluation gold fixtures** | Synthetic test conversations persisted; unclear which traces were real vs fixtures | No explicit fixture delineation | Mark and separate gold evaluation conversation state | Reproducible evaluation; clean audit trail |

---

## 1. Starting architecture

The system was designed as a **governed agentic workflow**, not an unconstrained autonomous agent. The core principle separates concerns:

- **Deterministic boundary**: policy routing, evidence sufficiency, version filtering, audience filtering, candidate assessment, product identity, calculations, citations, refusal
- **LLM role**: structured understanding and grounded composition over retrieved passages only
- **Retrieval**: semantic similarity plus metadata constraints (active version, audience, product, authority)
- **Verification**: eight post-generation checks before any answer leaves the system. Six at the start; check 7 (product scope) arrives in §5 and check 8 (product relationships and unsupported property claims) in §5.1, and `assistant/answering/answer.py` counts eight today — `obs.span("checks", count=8, …)`

Conversation state was added to carry context across turns — product identity, substrate, location, exposure — using LangGraph orchestration with in-memory checkpointing in the demo.

**Design rule**: *Transcript ≠ trusted state.*

---

## 2. Trusted state vs transcript contamination

### Observed failure

In multi-turn conversations, the UI history block included prior assistant-generated answers alongside user questions. When a user asked a follow-up question, the assistant's own previous response was available in the prompt as part of the "history for working out what the question refers to".

The design explicitly warned against this: the history block carried a comment stating "This is NOT evidence." However, generated figures that appeared in the history could be recycled into later answers, producing self-referential citations. When Check 2 validated a figure against passages, the passage might be a prior generated answer that itself was cited to an indexed passage — creating an indirect evidence path that bypassed the safety checks.

### Root cause

Conversation state management conflated two distinct needs:
- **Trusted slots**: product identity, substrate, location, exposure — facts the system needs to carry forward and reason about
- **Human reference context**: prior questions and their answers — needed for pronoun resolution but not for evidence

The system carried both through the same history mechanism and supplied both to the model as if they had equivalent epistemic status.

### Resolution

Separate the concerns:

1. **Trusted slots remain in LangGraph checkpoint state** (`assistant/turn/session.py`), visible to routing, candidate assessment, and evidence binding. These values are authoritative for the system's reasoning.
2. **Transcript history carries only user questions**, not generated answers. The history block is formatted so it cannot be cited (no citation markers) and explicitly states "never take a fact from it."
3. **Model sees both**: the trusted slots are printed as stated assumptions, and the question history is printed as working context for pronoun/reference resolution.

No generated answer reenters as evidence-like material.

### Outcome

- Citation chains cannot loop through prior generated answers
- Each turn's facts must still trace to the indexed corpus
- Check 1 (citation overlap) catches any slip where a sentence draws from history instead of passages
- Test confirms no feedback loops: `tests/test_context_isolation.py` — a poisoned prior answer changes neither the route, nor a detected slot, nor the embedded query, and the transcript still reaches the compose prompt delimited as non-evidence

**Design rule**: *Transcript context ≠ evidence source.*

---

## 3. Compound-query evidence binding

### Observed failure

The system retrieved correct evidence but the model attached claims to the wrong passages. Example: a question asking about both substrate suitability and application thickness retrieved passages covering both topics. The model, instructed to mark each claim with its passage number, sometimes tagged a thickness claim to the substrate passage or vice versa.

The citation marker was present (Check 1 passed), and the words overlapped a passage (citation validation passed), but the *semantic association* between the claim and the passage was incorrect. A figure of 10–30 mm, cited to a passage that discussed solid brick substrates, was not the thickness for that substrate.

### Root cause

Citation checks operate at the lexical level: does this sentence's words overlap the passage it claims to cite? They do not verify that the semantic content matches. "Application thickness" and "substrate type" are distinct concepts, and retrieving both in the same set does not authorize the model to mix them.

The checks assumed that retrieval had already enforced scoping: if all five passages were about Product X's thickness on masonry, the model cannot accidentally make them about Product X's suitability for concrete.

That assumption failed when a single retrieved passage contained multiple properties or when the model synthesized a claim that spanned multiple properties but cited them inconsistently.

### Fix strategy

Introduce deterministic property-evidence binding **before** the model runs:

1. **Extract requested properties** from the question (what is the user asking for?)
2. **For each retrieved passage**: record which properties that passage explicitly covers (via vocabulary matching and heading context)
3. **Create property-evidence bindings**: map property → set of passages that support it
4. **Promote bound passages** for composition: reorder the evidence set so passages supporting the actually-requested property appear first
5. **Supplement, not replace**: other passages remain available (needed for cross-product comparisons), but the bound ones get priority

Implemented in `assistant/answering/answer.py:evidence_binding()` (line 450) and integrated into the composition path — `_binding_guidance()` reads it to tell the model where each half of the question is stated, and `promote_bound()` reorders on the single-property case. It lives beside the checks it feeds rather than in the router, which is where this document first placed it.

### Validation

Measured over three scenarios:

- **Correct evidence, full set**: Model retrieves all passages, including some covering different properties. Result: mixed associations possible.
- **Correct evidence, reordered**: Binding promotes the property-matching passages first. Result: ordering alone insufficient (lexical attractor from FAQ heading matching the question overrides position).
- **Binding + membership**: Combine reordering with selective membership (see stage 5). Result: correct scoping.

The binding mechanism detects when a passage covers the requested property and privileges it, but does not force the model to use it. Stage 5 adds the membership layer.

### Outcome

- Property-evidence bindings are deterministic and verifiable
- The mapping from question property to evidence is explicit
- Follows the design principle: *Retrieved passage ≠ sufficient evidence without property verification.*
- Later fixes (stage 5) build on this foundation

---

## 4. Product registry pollution

### Observed failure

The product name registry included generic phrases like "lime plaster" and "lime mortars", which entered from editorial links in the knowledge base articles. Check 5 (real names only) then rejected answers containing common descriptive phrases because those phrases appeared in the registry, and the system interpreted them as product names.

Example: The answer "Ultra is a general purpose lime plaster suitable for..." was rejected because "lime plaster" appeared in the registry (harvested from a link mid-sentence in an article). Check 5 saw that term, read it as a product, and noted that the recommended product (Ultra) was mentioned but also that an unapproved "lime plaster" product was mentioned in the same sentence, triggering a refusal.

### Root cause

The harvest function extracted product names from all links matching the product URL pattern (`/products/[a-z0-9-]+/[a-z0-9-]+`), regardless of context. Editorial prose in knowledge base articles often writes:

```html
<a href="/products/lime-plaster/solo-onecoat-plaster">lime plaster</a>
```

The link text "lime plaster" was a generic descriptor, not a product name, but the harvester treated all link text equally.

### Fix

**Apply a capitalization filter**: harvest product names only from links whose text starts with a capital letter.

Measured over the cached corpus:
- Before: 63 harvested names
- After: 59 harvested names
- Lost names: "lime plaster", "lime mortars", "here in our gallery", "pre-mixed Natural Lime Mortar"
- Retained: all genuine product names (Lime Green Ultra, Duro, Solo, etc.)

The capitalization filter is applied only to product harvesting, not to colours (which are already title-cased in the published block) and not to other link types. Widening rules beyond their measured defect is how the next defect arrives.

### Outcome

- Product vocabulary contains only genuine product names
- Check 5 (real names only) functions correctly
- All editorial links to products still contribute valid names
- No false refusals from generic prose containing product category terms

---

## 5. Product-scope verification

### Observed failure

An answer was fully cited to correct passages but still contradicted the evidence:

- User asked about product Ultra, carried from a previous turn
- Model answered: "Ultra... is suitable for internal walls as an insulating lime plaster base coat" [citation markers present]
- Evidence set also included Duro's FAQ entry: "How thick can I apply lime basecoat in any one go? In the case of our general purpose Duro lime base coat, 9 to 12 mm..."
- Model had pulled the thickness figure from Duro's FAQ despite Ultra being the resolved product

The model cited correctly (markers present, word overlap valid), but the product scope was wrong. The figure belonged to Duro, not Ultra.

### Root cause

Citation checking operates at the lexical level: do the words in this sentence appear in the cited passage? It does not verify the semantic relationship between the product mentioned in the question/context and the product implicitly claimed by the passage.

The checks assumed that retrieval and ranking had already filtered to one product or that the model would not confuse products when multiple datasheets were in the set. That assumption failed when:

1. A question carried a resolved product from context (Ultra)
2. Retrieval returned passages from multiple products for valid reasons (cross-product compatibility, shared sections)
3. The model preferred a lexically-matching passage (FAQ heading "How thick can I apply") over the position-promoted passage (Ultra's datasheet thickness)

### Fix: Check 7 — Product-scope resolution

Introduce a post-generation check that verifies product identity consistency:

1. **Resolve the intended product** from the question and conversation context
2. **Extract product mentions** from the generated answer (both stated and implicit from evidence)
3. **Check consistency**: every product mentioned must be either (a) the resolved product, (b) a product mentioned in a supporting passage for cross-product reasons, or (c) a real product name that is explicitly named in a passage, not inferred

Passes if:
- Only the resolved product is mentioned, OR
- Any additional products are named because they appear in the cited evidence for comparison/compatibility

Fails if:
- A product is mentioned that does not appear in any cited passage
- The implicit product from evidence-context does not match the resolved product

### Outcome

- Wrong-product answers are caught before printing
- Evidence-scope and product-scope are verified independently
- Combined with binding (stage 3), ensures evidence belongs to the correct product
- Follows the principle: *Citation correctness ≠ product-scope correctness.*

### 5.1 Check 8 — relationship direction and unsupported property claims

Check 7 settled *which product* an answer is about. It said nothing about what
an answer asserts *between* two products, and that is a second way a correctly
cited sentence can be wrong. "Forte is suitable over Ultra" and "Ultra is
suitable over Forte" share every content word, so check 1's overlap test cannot
separate them, and a passage stating one of them is not evidence for the other.

Check 8 (`_semantic_failures`, `assistant/answering/answer.py`) verifies two
things a lexical check cannot:

1. **Relationship direction, polarity and conditions.** Every product-to-product
   relation the generated sentence asserts must appear in a cited passage with
   the same direction, the same polarity, and its conditions preserved — where
   a condition includes a following sentence that qualifies or negates
   (`_restriction_context`). Two named products joined by *over*, *under*,
   *onto*, *compatible*, *finished* or *coated* with no relation the passage
   supports fails as "the product relationship cannot be established".
2. **Property claims that a shared product name does not license.**
   *waterproof*, *watertight*, *structural*, *certified*, *certification* must
   be explicitly asserted of the same product in a cited passage. A sheet
   mentioning a product proves none of them, and these are the claims where an
   unsupported sentence is a liability rather than an inaccuracy.

It also enforces layer order where an answer is asked to establish one.

**Design rule**: *A relation is a claim about two products, and needs its own
evidence.*

---

## 6. Compose evidence membership

### Observed failure

With Ultra carried from context and the question "and what thickness should I apply it at?":

- **Full evidence set** (Ultra datasheet + FAQ + articles): Model answered with Duro's 9–12 mm, Check 7 caught it, refusal
- **Ultra datasheet only**: Model answered correctly with 10–30 mm
- **FAQ removed but other documents kept**: Model answered correctly

The problem was not ranking (Ultra was marked [1]) but membership. The FAQ entry matched the question's heading ("How thick can I apply...") and that lexical match overrode the position ranking.

### Root cause

The evidence set included generic, product-range documents (FAQ, knowledge-base articles) alongside product-specific documents (datasheets). When the model prioritizes a lexically-matching question/answer pair in the FAQ over a positionally-ranked datasheet, membership becomes the control.

Earlier attempts at solution:
- **Reordering/ranking alone**: Insufficient; lexical attraction in prompts overrides position.
- **Deterministic binding (stage 3)**: Necessary but not sufficient for this case; binding shows that the property exists in Ultra's evidence, but does not prevent the model from preferring a generic match.

### Fix: membership filtering — designed by document class, shipped by product identity

The rule as designed was conditional on document class: when the resolved
product's own evidence already covers the requested property (verified via
binding), exclude generic product-range documents — FAQ entries,
knowledge-base articles, system-level guides — from the composition evidence
set, with the coverage condition as the safety property.

**That form never landed, and saying so costs less than letting a reader grep
for it.** Nothing in this repository excludes a passage by document type, and
no filter consults the binding for a coverage condition; there is no
`_filter_compose_evidence`, in `engine.py` or anywhere else. What shipped is
`scoped_evidence()` (`assistant/answering/answer.py:580`), which filters on
**product identity** instead:

- **Kept**: every passage whose own product is one the question names — or the
  product carried in `decision.slots["product"]` — and every passage whose text
  explicitly names one of them. That second clause is what keeps a primer or
  finishing relation stated on another product's sheet available, so
  cross-product reasoning survives.
- **Dropped**: everything else, including the Duro FAQ entry that produced the
  failure above, because it is neither about Ultra nor names it.
- **Fallback**: when the question names no product and none is carried, the hit
  list is returned unchanged. That, not a coverage test, is what stops this
  starving the model.

The observed failure is fixed, and by a narrower mechanism than the designed
one: a document class is a judgement about a source, a product name is a fact
about a passage.

### Measurement

- Standard retrieval is unchanged; this filters what retrieval already returned
- It is **not** Compose-only, as this document once claimed. `scoped_evidence()`
  is applied on Extract (`answer.py:1712`) and in the factual and relationship
  helpers (`1639`, `1685`) as well as on Compose (`1778`). Cited hand-off and
  refusal are unaffected
- An empty scoped set is an explicit refusal — "no evidence for the requested
  product" — not a silent fall back to the unscoped list
- Boundary cases are asserted in `tests/test_acceptance_evidence_boundaries.py`

### Outcome

- Membership control is explicit and narrow
- Does not break cross-product reasoning: a passage naming the product crosses
  the filter whatever document it sits in
- The un-named-product case is the release valve, and it is the one to watch:
  a question that names nothing is scoped by nothing
- Follows principle: *Retrieved evidence ≠ sufficient membership for composition.*

---

## 7. Vision substrate contract and provenance

### Observed failure

Vision observations were indistinguishable from user statements in conversation state. When a user asked "what should I use on brick?" and sent a photograph of brick, the vision model returned `substrate: brick` from the image. The answer printed "brick (substrate), as you told me earlier" — attributing the model's visual reading to the user's actual statement.

The system lost the distinction between:
- **User-stated facts**: "I have a brick wall"
- **Vision-inferred facts**: "I see brickwork in the photograph"

The confusing attribution ("as you told me") laundered a model inference into testimony.

### Root cause

Vision observations were resolved to a `carried` state dict and merged with prior context, so they became indistinguishable from facts the user actually stated. The `Provenance` enum had no member for observations from images; adding one required the distinction to propagate through conversation state, answer rendering, and the checks.

Additionally, the vision model could return values (like `exposure: "visible"`) that were not actual values but artifacts of prompt/inference. Without vocabulary filtering, observations could contradict the schema, or introduce values that check 5 would then reject.

### Fix: Three-layer contract

**Layer 1 — Constraint to schema**: The vision model returns JSON with an enum-constrained `attribute` field. The enum is `VISION_SLOTS` — the ten attributes of the three tiers below — and what it omits is the point: no `product`, no `cause_asked`, and nothing naming structure, compliance or chemistry. A second list, `FORBIDDEN_ATTRIBUTES`, is checked in the decoder and counted into `refused`, so a model that ignores its own schema is visible rather than merely unsuccessful. Only the two tier-1 attributes may reach a router slot.

**Layer 2 — Vocabulary resolution**: `vision.resolve()` compares observed values against `config/vocabularies.json`. Only values the vocabulary defines map to slots. An observation like `substrate: "Lime Green Solo"` is discarded (the term is not in the substrate vocabulary). This prevents the model from inventing substrate values or smuggling product names into slots.

**Layer 3 — Provenance tracking**: The `Provenance` enum distinguishes:
- `STATED`: from the current question
- `CARRIED`: from an earlier turn (still user-stated)
- `OBSERVED`: from a photograph via vision
- `ASSUMED`: system-chosen (per-option answers)

Vision observations are marked `OBSERVED` and the renderer prints "as you can see in the photograph" not "as you told me earlier".

### Measured decision on tiers

Initial design expected vision to fill four slots: substrate, location, exposure, symptom.

After running the real model on real photographs, `vision.py` settles on **three
tiers**. Everything in all three is *reported* to the caller; only the first may
change what the system does.

- **Tier 1 — may fill a router slot** (`ROUTER_SLOTS = ("substrate", "symptom")`).
  `substrate` because a photograph of genuinely exposed masonry does settle it,
  and it is gated further by `_covered_without_exposure`. `symptom` because a
  symptom is a visible condition — deposits, cracking, a blown patch — and that
  is perception. `cause_asked` is absent from every tier: a cause is the
  technical team's judgement (decision 16), and a model able to fill it would
  route a diagnosis question away from the hand-off it must take.
- **Tier 2 — reported, never routed** (`CONTEXT_SLOTS = ("location", "exposure")`).
  These were tier 1 until the first real photograph went through the real model,
  which answered a flat brick elevation with `{"attribute": "exposure", "value":
  "visible", "confidence": 1.0}` twelve times. No such string is a value the
  vocabulary defines, so nothing reached a slot and the safety property held —
  but a photograph does not carry these facts, and a model reporting them is
  producing text rather than evidence. Inside/outside is decided by what is
  usually out of frame; exposure is a fact about a site's weather. Both are read,
  both are shown as context, neither reaches the router, and the person is asked.
- **Tier 3 — directly observable wall condition** (`CONDITION_ATTRIBUTES`:
  `exposed_masonry`, `existing_finish`, `damaged_finish`, `cracks`, `staining`,
  `texture`). What a photograph is actually good for. These never route either;
  they are what the assistant reports when asked what it can reliably identify,
  and two of them — `exposed_masonry`, `existing_finish` — additionally gate the
  substrate claim in tier 1.

### Outcome

- Vision observations are typed, vocabulary-gated, and confidence-bounded
- Provenance is explicit; user claims are distinguished from visual inferences
- Model cannot invent substrate values or mention products
- Failed observations (malformed, out-of-vocabulary) produce no slot, leaving the router unchanged
- Failure reduces coverage (ask-back for substrate if observation fails), not safety
- Follows principle: *VLM hypothesis ≠ fact.*

---

## 8. Ollama context consistency

### Observed failure

Local Ollama inference was unstable: some model calls returned quickly, others hung or timed out, and the behaviour was non-deterministic. Investigation traced calls to the vision model and generation model using different context-size parameters. The vision model was configured with one `NUM_CTX` value; the generation model used another. Under memory pressure, context-size mismatches contributed to instability.

### Root cause

Configuration for Ollama context size was scattered across multiple configuration points:
- `OLLAMA_NUM_CTX` (environment)
- Vision-specific settings
- Generation-specific settings

No single source of truth existed. The application never made unified decisions about available context or propagated one authoritative value to all model calls.

### Fix

Introduce a single `NUM_CTX` variable (`assistant/infrastructure/ollama.py`):

```python
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
```

All model calls — generation, embeddings, vision — use this value. Imported and passed to every Ollama HTTP call. Default is 8192; overridable via environment.

**No model selection changed.** Only configuration consistency.

### Outcome

- All Ollama calls use the same context size
- No configuration drift between components
- Local inference stability improved (not guaranteed, but demonstrably better)
- Follows principle: *Infrastructure consistency matters when it affects safety.*

---

## 9. Conversation state security: audience in cache key

### Observed failure

The answer cache keyed on (normalized question, snapshot id, generation model, chunking version) but **not** the audience set. A staff member asking "how much Solo for MgO board?" would generate an answer and cache it. A public caller asking the same question would get the cached staff answer.

The design principles state that answers are not reusable across audiences, but the cache invalidated that contract.

### Root cause

The cache was built as exact-key with these fields:
- Normalized question
- Snapshot ID
- Generation model
- Chunking version

The audience set was not included, under the assumption that a separate cache instance would exist per audience or that audience filtering would prevent collisions. Neither assumption held in the demo.

### Fix

Add `audience_set` to the cache key. The key is now:

```python
(normalized_question, audience_set, snapshot_id, generation_model, chunking_version)
```

Audience must match exactly for a cache hit to apply.

The key has since grown three more fields — the carried slots, their origins and
the transcript (`Assistant._cache_key`, `assistant/answering/engine.py`) — for
the same class of reason: a substrate read off a photograph and a substrate the
caller stated print different sentences, so they are different answers.

### Outcome

- No audience leakage through the cache
- Staff answers stay isolated
- Tests confirm, in both directions: `tests/test_cache.py::test_a_staff_answer_is_not_served_to_a_public_caller` and `::test_a_public_answer_is_not_served_to_a_staff_caller_either`
- Follows principle: *Audience filtering is not optional; it applies everywhere.*

---

## 10. Question-carrying transcript

### Observed failure

The prompt history block included prior generated answers alongside user questions. This created a feedback loop where the system's own output could be recycled into the next answer. Additionally, if a prior generated answer contained a factual error that had been caught by checks and refused, that refusal text could be included in history, subtly shaping later answers.

### Root cause

The history block was designed to carry context for pronoun resolution ("it", "that wall"). Including full prior answers seemed natural for continuity. However, the history was given to the model without citation markers, making it look like background context rather than evidence — but the model might still treat it as authoritative.

### Fix

Simplify the history block to carry **only user questions**, not assistant answers. The history is explicitly marked as non-evidence:

```
Earlier turns of this conversation, for working out what the question refers
to -- "it", "that wall", "the one you mentioned". This is NOT evidence.
```

Trusted state (substrate, location, exposure, product) is carried through LangGraph checkpoint state and printed separately as "Substrate: brick (as you told me earlier)".

### Outcome

- No generated answers re-enter the evidence base
- Pronoun resolution remains possible
- Check 1 catches any sentence that somehow references history instead of passages (no citation possible)
- Clear separation of epistemic roles

---

## 11. Local and production runtime clarity

The demo runs on:

- **Storage**: SQLite (assessment path)
- **Retrieval**: Local embedding similarity + metadata filtering in code
- **Generation**: Local Ollama, `qwen3.5:4b`
- **Embeddings**: Local Ollama, `qwen3-embedding:0.6b`
- **Vision**: Optional local Ollama `qwen3.5:4b` with vision
- **Conversation state**: two halves, and only one of them is durable. The LangGraph checkpointer is `InMemorySaver` — per-process, lost on restart (decision 20's dependency block). The session's carried slots, pending ask-back and turn history persist to the `sessions` table via `PersistedSessionStore` (`assistant/turn/session_storage.py`), wired into the web surface by `open_persisted_session_store()`, which falls back to the in-memory store if no connection can be opened
- **Tracing**: Structured spans with optional OTLP export; LangSmith tracing disabled in code

Production infrastructure (not implemented in demo, but designed as seams):

- **Storage**: PostgreSQL + pgvector (adapter contract-tested; implementation untested in production)
- **Conversation state**: Durable checkpointer via PostgreSQL (dependency-blocked: `langgraph-checkpoint-postgres` 3.0.1 incompatible with current `langgraph`)
- **Identity/audience**: Authenticated via SSO/account claims (designed seam, not implemented)
- **Scaling**: Separate model-serving tier, generation queue, rate limiting (designed, not built)

The boundary between the two is the `KnowledgeRepository` interface and the configuration: one variable (`ASSISTANT_POSTGRES_DSN`) selects between them. The answer engine, verification, routing, and safety checks do not change.

**This is the engineering contract**: *Local mode changes infrastructure, not safety semantics.*

---

## 12. Remaining known limitations

These are supported by the repository and current implementation:

- **Conversation pronouns**: Some bridging questions ("Why is it better?") need cleaner routing when "it" has been established as a product property rather than the product itself. Current workaround: ask-back.
- **Targeted property retrieval**: When a question asks about a property that is only mentioned in one section of a datasheet, retrieval might return multiple sections. Evidence binding helps but doesn't guarantee tight scoping in all cases.
- **Cross-product relation direction**: A passage like "Forte is suitable over Ultra" is evidence for Ultra-to-Forte compatibility but may not answer Forte-to-Ultra compatibility. Check 8 (§5.1) now verifies direction, polarity and conditions against the cited passage, so this fails closed into a refusal rather than printing; what remains is the coverage cost, not a safety gap.
- **Image inference latency**: minutes, on a processor with no graphics card. The one figure actually measured on the build machine is **191.9 seconds** for a single 512×512 image through `qwen3.5:4b` (recorded at `assistant/answering/vision.py`, beside `VISION_TIMEOUT`); the opt-in live lane records a wider observed range of **150 to 600 seconds** (`tests/test_vision_live.py`), and one pre-fix run returned malformed JSON after 204 s. `VISION_TIMEOUT` defaults to 600 s for that reason. Vision is off unless `ASSISTANT_VISION_DEMO=1`; pre-run or use recorded traces for a demonstration.
- **In-memory graph checkpoint**: the LangGraph checkpointer is not durable and is lost on process restart, because the PostgreSQL checkpointer is dependency-blocked. Session slots and turn history *do* survive a restart through `PersistedSessionStore`, so the two halves of conversation state have different durability — worth knowing before quoting either as the whole.
- **Single Ollama instance**: No model parallelism; generation is the bottleneck. Production queue not implemented.
- **Exact-key cache only**: Template-keyed cache (decision 14) remains roadmap. Current form hits only on word-identical questions.

---

## 13. Design principles learned

From the failures and fixes above:

1. **Transcript ≠ trusted state.** Conversation history is for reference; evidence is from retrieval. Generated answers must not re-enter.
2. **Retrieved passage ≠ sufficient evidence.** Retrieval confidence, lexical matching, and position ranking are not evidence sufficiency. Property coverage, product scope, and membership must be checked.
3. **Citation correctness ≠ answer correctness.** A sentence with markers and word overlap to a passage can still attach the wrong claim to the wrong evidence. Property binding, product resolution, and scope checks are separate.
4. **VLM hypothesis ≠ fact.** Vision observations are typed, vocabulary-gated, and explicitly provisional. Confidence, provenance, and vocabulary membership must be enforced.
5. **Evidence membership controls output.** Retrieval is unchanged; composition membership can be scoped. A passage that is neither about a requested product nor names one does not reach the model — and when no product is requested, nothing is scoped away.
6. **Fail closed before weakening verification.** If a fix would require relaxing a check, the design prefers a refusal (fail closed, coverage reduced) over a compromise check (safety maintained but harder to verify).
7. **Fix the narrowest layer that caused the defect.** Product name harvesting: fix the link-text filter, not the whole extraction. Transcript: carry questions, not answers. Cache key: add audience, not rebuild the cache. Narrow fixes are easier to reason about and validate.

---

## References

- `assistant/answering/answer.py` — the eight checks, `evidence_binding()`, `promote_bound()`, `scoped_evidence()`, `Provenance` and `SlotFact`
- `assistant/answering/router.py` — routing, slot detection and precedence
- `assistant/answering/engine.py` — `Assistant`: topic split, per-part orchestration, cache key, logging
- `assistant/answering/vision.py` — Vision contract, tiers, provenance, and vocabulary filtering
- `assistant/cache.py` — Cache key construction with audience isolation
- `assistant/turn/session.py` — Trusted conversation state and slots
- `tests/test_evidence_binding.py` — Property-evidence binding validation
- `tests/test_checks.py` — Post-generation check coverage
- `tests/test_acceptance_evidence_boundaries.py` — `scoped_evidence()` membership boundaries
- `tests/test_context_isolation.py` — transcript-contamination regressions
- `tests/test_cache.py` — Cache isolation and audience separation
- `tests/test_vision_contract.py` — Vision observation typing and resolution
- `eval/conversations.json` — Evaluation conversations with traced failures/fixes
