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
| **5. Compose evidence membership** | Ultra-only set correct, FAQ-only wrong, FAQ+Ultra produced refusal | Lexical attractor (FAQ heading matching question) preferred over deterministic binding | Exclude generic documents when product-bound evidence covers property | Membership-level control without breaking retrieval |
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
- **Verification**: six post-generation checks before any answer leaves the system

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
- Test confirms no feedback loops: `tests/test_transcript_contamination.py`

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

Implemented in `assistant/answering/router.py:evidence_binding()` and integrated into the composition path.

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

### Fix: Conditional membership filtering

Apply a **deliberate, measured, narrow rule**:

When the resolved product's own evidence already covers the requested property (verified via binding), exclude generic product-range documents from the composition evidence set.

Specifically:
- **Excluded when product evidence is sufficient**: FAQ entries, knowledge-base articles, system-level guides
- **Always retained**: 
  - The resolved product's datasheets and product pages
  - Other products' datasheets (needed for cross-product analysis: "can I use Ultra over Solo?")
  - Evidence that explicitly mentions the resolved product

The **coverage condition** is what makes this safe: filtering only occurs when the product's own evidence answers the question. If no single product's evidence covers the property, no filtering happens and all sources are included.

Implemented in `assistant/answering/engine.py:_filter_compose_evidence()`.

### Measurement

Over the corpus:
- Standard retrieval is unchanged
- Membership filtering applies only to Compose path
- Other paths (Extract, Cite hand-off, Refuse) are unaffected
- Evaluation shows: correct answer rates improved; wrong-product answers eliminated; over-refusal rate unchanged

### Outcome

- Membership control is explicit, narrow, and conditional
- Does not break cross-product reasoning
- Evidence sufficiency is maintained (filtering only when condition is met)
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

**Layer 1 — Constraint to schema**: The vision model returns JSON with an enum-constrained `attribute` field. Attributes are `(substrate, symptom)` only — not `location` or `exposure` (see next layer). No `product` attribute exists for it to fill.

**Layer 2 — Vocabulary resolution**: `vision.resolve()` compares observed values against `config/vocabularies.json`. Only values the vocabulary defines map to slots. An observation like `substrate: "Lime Green Solo"` is discarded (the term is not in the substrate vocabulary). This prevents the model from inventing substrate values or smuggling product names into slots.

**Layer 3 — Provenance tracking**: The `Provenance` enum distinguishes:
- `STATED`: from the current question
- `CARRIED`: from an earlier turn (still user-stated)
- `OBSERVED`: from a photograph via vision
- `ASSUMED`: system-chosen (per-option answers)

Vision observations are marked `OBSERVED` and the renderer prints "as you can see in the photograph" not "as you told me earlier".

### Measured decision on tiers

Initial design expected vision to fill four slots: substrate, location, exposure, symptom.

After running the real model on real photographs:
- `substrate`: Safe to tier 1 (may fill) — photographs of exposed masonry can settle this
- `symptom`: Safe to tier 1 — visible conditions like cracking, staining are perception
- `location`: Moved to tier 2 (reported, never routed) — a flat elevation lacks ground lines, skylights, skirting; confidence that a wall is inside/outside from one photograph is poor
- `exposure`: Moved to tier 2 — confidence in exterior/interior/semi-exposed from a single photo face is unreliable

Tier 2 observations are reported to the user for context but never change routing.

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
NUM_CTX = int(os.getenv('OLLAMA_NUM_CTX', '8192'))
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

### Outcome

- No audience leakage through the cache
- Staff answers stay isolated
- Test confirms: `tests/test_cache.py::test_audience_isolation()`
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
- **Conversation state**: LangGraph `InMemorySaver` (per-process, not durable)
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
- **Cross-product relation direction**: A passage like "Forte is suitable over Ultra" is evidence for Ultra-to-Forte compatibility but may not answer Forte-to-Ultra compatibility. Explicit direction needs verification.
- **Image inference latency**: Local CPU vision inference ranges 156–198 seconds for successful cases; timeouts possible under load. Pre-run or use recorded traces for demo.
- **In-memory conversation state**: Not durable; lost on process restart. LangGraph PostgreSQL checkpointer is dependency-blocked.
- **Single Ollama instance**: No model parallelism; generation is the bottleneck. Production queue not implemented.
- **Exact-key cache only**: Template-keyed cache (decision 14) remains roadmap. Current form hits only on word-identical questions.

---

## 13. Design principles learned

From the failures and fixes above:

1. **Transcript ≠ trusted state.** Conversation history is for reference; evidence is from retrieval. Generated answers must not re-enter.
2. **Retrieved passage ≠ sufficient evidence.** Retrieval confidence, lexical matching, and position ranking are not evidence sufficiency. Property coverage, product scope, and membership must be checked.
3. **Citation correctness ≠ answer correctness.** A sentence with markers and word overlap to a passage can still attach the wrong claim to the wrong evidence. Property binding, product resolution, and scope checks are separate.
4. **VLM hypothesis ≠ fact.** Vision observations are typed, vocabulary-gated, and explicitly provisional. Confidence, provenance, and vocabulary membership must be enforced.
5. **Evidence membership controls output.** Retrieval is unchanged; composition membership can be scoped. When product-bound evidence is sufficient, generic documents are not included in the generation set.
6. **Fail closed before weakening verification.** If a fix would require relaxing a check, the design prefers a refusal (fail closed, coverage reduced) over a compromise check (safety maintained but harder to verify).
7. **Fix the narrowest layer that caused the defect.** Product name harvesting: fix the link-text filter, not the whole extraction. Transcript: carry questions, not answers. Cache key: add audience, not rebuild the cache. Narrow fixes are easier to reason about and validate.

---

## References

- `assistant/answering/answer.py` — Citation checks and post-generation verification
- `assistant/answering/router.py` — Evidence binding, routing, and property verification  
- `assistant/answering/engine.py` — Composition path and membership filtering
- `assistant/answering/vision.py` — Vision contract, provenance, and vocabulary filtering
- `assistant/cache.py` — Cache key construction with audience isolation
- `assistant/turn/session.py` — Trusted conversation state and slots
- `tests/test_evidence_binding.py` — Property-evidence binding validation
- `tests/test_checks.py` — Post-generation check coverage
- `tests/test_cache.py` — Cache isolation and audience separation
- `tests/test_vision_contract.py` — Vision observation typing and resolution
- `eval/conversations.json` — Evaluation conversations with traced failures/fixes
