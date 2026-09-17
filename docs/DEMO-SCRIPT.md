# Demo script — 7 to 10 minutes

A rehearsal script for demonstrating the Technical Knowledge Assistant live.
Nine beats, timed. Every expected result is described by **what kind of answer
it is** — the route taken, whether it cites, whether it refuses — and never by
the exact words, because the same question asked twice produces the same
evidence and different prose (`DECISIONS.md` entry 7, gate G4). Rehearsing
against remembered wording is how a demonstration breaks.

Read [the pre-demo checklist](#pre-demo-checklist) first. The single most
important line in it is the cache warm-up: a composed answer on a question the
machine has not seen costs **35 to 200 seconds** on a processor with no
graphics card, and 0.017 s once it is cached.

---

## Before you start

Two windows, both already open:

1. A browser at `http://127.0.0.1:8765/`
2. A terminal in the repository root, for the trace step

Say the audience out loud once: **this is running entirely on this machine.**
No API key, no hosted model, no network. That is the constraint the whole design
answers to, and it is worth claiming at the start rather than at the end.

---

## 1 · The problem (45 seconds, no clicking)

> Lime Green publish about ninety technical documents — datasheets, guides, an
> FAQ. Their technical team answers the same questions off them all day. The
> obvious thing to build is a chatbot over the PDFs. The obvious thing is also
> the dangerous thing: a plaster recommended for the wrong substrate is a
> building that fails, and a paraphrased mixing ratio is a bag of plaster that
> does not set.
>
> So this is built the other way round. The model does not decide anything. It
> composes sentences over passages that deterministic code has already chosen,
> and six checks run before a single word prints.

Do not open the architecture diagram yet. It lands better at the end, once they
have seen the behaviour it explains.

---

## 2 · The straightforward factual question (60 seconds)

**Type:** `How much water does Solo Onecoat need per bag?`

**Expect:** an answer carrying `[n]` citation markers, a **Sources** disclosure
naming the Solo datasheet with a followable link, and the published figure
reproduced exactly — the sheet says *between 5 and 6 litres per 25 kg sack*.

**Point at the number, then say:**

> That figure is not the model remembering a datasheet. Check 2 compares every
> number in the answer against the passage it cites, and a figure that is not
> found word-for-word does not print. If the model had written "5 to 6 litres
> per bag" — same meaning, different words — this would have refused instead.

**Click Sources open.** The link goes to Lime Green's own PDF.

> Every sentence is traceable. That is the property, and it is enforced after
> generation rather than asked for in a prompt.

⏱ This is the one beat that composes. It must be cache-warm — see the checklist.

---

## 3 · Multi-turn context (90 seconds)

**Turn 1:** `I have an old solid brick wall and want to improve its insulation.
Would Lime Green Ultra be suitable internally, and what thickness can it be
applied at?`

**Turn 2:** `How much would I need for 30 m² at 25 mm?`

**Expect:** the second answer is about Ultra, on that wall, without either being
restated. Open **Why this answer?** on the second reply and point at the slots.

> The second question names no product and no wall. The conversation is a state
> machine — the turn is a LangGraph graph with a checkpoint per session — and
> the substrate, the location and the product carry forward. Correct the wall
> and the correction wins; it overwrites rather than accumulating.

**If you have 20 spare seconds**, add a third turn: `Sorry, it is actually
stone.` and show the substrate change in the diagnostics.

---

## 4 · The calculation, and the part it refuses (45 seconds)

**Type:** `How many bags of Duro do I need for 20 square metres?`

**Expect:** the **extract** path. The published coverage and pack-size passages
print verbatim, with their citations — and the multiplication is **not done**.

> It has given the two numbers the sum needs and declined to do the sum. That is
> deliberate. Coverage depends on suction, on how flat the wall is and on how
> thick it goes on; a confident bag count is a number somebody orders against.
> The arithmetic restriction is in code, not in the prompt.

This is the beat that most reliably surprises a technical panel. Do not rush it.

---

## 5 · The photograph (60 seconds)

**Click `+`**, choose a wall photograph, and note the filename chip that appears
beside the input. **Type:** `What plaster should I use on this wall?`

**Expect:** the attachment named in your own message bubble, and an answer that
does not name a product on the strength of the image.

> A vision model reads the photograph into structured observations — substrate,
> exposure, symptom — each with a confidence and the region it came from. It is
> never allowed to name a product: the attribute field is an enum of four slots
> and there is no product in it. Observations fill slots on the same router;
> below the confidence floor the slot stays uncued and it asks back, exactly as
> it would have without the photograph.
>
> Diagnosing a wall stays a human judgement. That is a liability decision, not a
> capability one — seeing the photograph does not license acting on it.

⏱ Perception costs a model call. Budget 30–60 s, and **have this one cached** if
the room is tight. If you are short of time, cut this beat before cutting 4 or 7.

---

## 6 · Product recommendation, and the ask-back (60 seconds)

**Type:** `Which plaster should I use?`

**Expect:** the **ask back** path — it asks what the wall is built of, and
offers the vocabulary (brick, stone, cob, laths, plasterboard, existing plaster
or render).

**Reply:** `brick`

**Expect:** the original question resumed and answered for a brick wall — not
a fresh answer about the word "brick".

> Substrate is a load-bearing slot. Missing exposure becomes a stated
> assumption; missing substrate stops and asks, because a recommendation on an
> assumed wall is the expensive error. And the reply resumes the parked turn —
> the graph paused mid-question and picked it up where it stopped.

---

## 7 · The question it must refuse (45 seconds)

**Type:** `Does this comply with Part L?`

**Expect:** the **route** path, in about two seconds. No retrieval, no model — a fixed referral with
the technical team's published phone number and office hours.

> Eleven topics never reach retrieval at all: price, stock, delivery, colour
> matching, structural judgement, compliance sign-off, health. This one is a
> certification. Lime Green sign those off; an assistant does not.
>
> And the phone number is not in a prompt. It was harvested from their contact
> page at ingestion, so the assistant cannot invent one.

⚠ **Use that exact phrasing.** "Can you confirm my Warmshell build complies with
Part L?" misses the gate's patterns, reaches retrieval, and takes about 80 seconds
to refuse rather than 2 seconds to route. The outcome is still safe — it declines
and hands over — but it is a long silence in a seven-minute demonstration. This is
known issue 1 in the README, and if a panel asks about guardrail coverage it is a
better answer than a rehearsed one: the gap was found by measurement, reproduced in
`tests/test_policy_gate_phrasing.py`, and left for a deliberate decision rather than
patched the night before.

**If they push on hallucination**, follow with:
`What is the U-value of Solo Onecoat plaster?`

**Expect:** a **refusal** — "not stated in the indexed material" — plus what the
sheet does publish, and the contact line.

> This is the interesting failure. That question retrieves at **0.739**, more
> confidently than the water question at 0.595 — because it is a well-formed
> question about a real product, so the Solo datasheet genuinely is its nearest
> neighbour. No similarity threshold separates those two; we swept it and none
> in range does. What catches it is a separate lexical gate: the property asked
> for has to actually appear in a cited passage, or it refuses.

---

## 8 · Why this answer, and the trace (60 seconds)

**On any answer, open `Why this answer?`** — the router step, the reason, the
top score, the generation time.

**Then, in the terminal**, take the correlation id from the page's network
response, or simply run the last one:

```
python -m assistant.trace <correlation-id>
```

> Every answer carries a correlation id, returned in a header. That id reads
> back the stages of that specific answer out of the log — which route, which
> passages, which checks. A reported problem gets investigated rather than
> reproduced.

Also worth pointing at the footer: the document and passage counts, both model
tags and the abstention threshold, so the index answering is named on screen.

---

## 9 · The architecture, last (60 seconds)

Now open [`docs/architecture.md`](architecture.md) — the answer-engine diagram.

> Grey is deterministic code. Purple is the single step where the model runs.
> Red is where it stops and hands over. The model touches one path out of five,
> and never originates a fact.
>
> Storage sits behind one interface with two adapters — SQLite ships so you can
> run this offline from a clean clone, PostgreSQL with pgvector is the
> deployment path, same schema, same version semantics. Exactly one active
> document version, enforced by the database rather than by application code, so
> a changed datasheet can never put two coverage figures into one answer.

Close on the trade, not on the feature list:

> The honest summary is that this refuses more than a general chatbot would.
> That is the design. A refusal costs a phone call; a confident wrong answer
> about a building costs a rebuild.

---

## Pre-demo checklist

Run through this **in order**, ideally 20 minutes before.

### 1. Services

```
ollama list                     # qwen3.5:4b and qwen3-embedding:0.6b present
curl http://localhost:11434/api/tags
python -m assistant.health      # store, snapshot compatibility, model reachability
```

### 2. Index

```
python -m assistant.index
```

Expect `unchanged 94 · reprocessed 0` in a couple of seconds on a second run. If
it reports 94 new and takes ~25 s, that is a first build and is fine.

### 3. Warm the answer cache — **do not skip this**

An uncached compose is 35–200 s. Ask each of the demo's composing questions once
and throw the answers away:

```
python -m assistant.cli "How much water does Solo Onecoat need per bag?"
python -m assistant.cli "I have an old solid brick wall and want to improve its insulation. Would Lime Green Ultra be suitable internally, and what thickness can it be applied at?"
python -m assistant.cli "How much would I need for 30 m² at 25 mm?"
```

The cache is keyed on the question, the audience set, the snapshot id, the
generation model and the chunking version — so warm it **after** any re-index,
never before.

### 4. Smoke the surface

```
set ASSISTANT_E2E=1
python -m pytest tests/e2e -m smoke -q
```

### 5. Start the server

```
python -m assistant.ui
```

Open the printed address. Check the footer names the index, ask one routed
question (`How much does a bag cost?`) to confirm answers render, then click
**New chat** so the panel starts on a clean page.

### 6. Have ready

- A wall photograph on the desktop, for beat 5
- A second terminal in the repository root, for beat 8
- `docs/architecture.md` open but not in front

---

## If something goes wrong

| Symptom | What it is | What to do |
|---|---|---|
| An answer takes more than a minute | Uncached compose | Say so — the latency number is in `DECISIONS.md` and owning it reads better than waiting. Move to a routed question. |
| A red bubble with a reference id | Ollama unreachable | `ollama serve`, then retry. The id is readable with `python -m assistant.trace`. |
| The page will not start | Index missing or built by another model | `python -m assistant.index`. The engine refuses a mismatched index by design — that refusal is itself worth showing. |
| An answer refuses that you expected to answer | The relevance gate | This is the design. Show the refusal, then show what it still hands over. |
| The photograph is ignored | Attachment not registered | Look for the filename chip beside the input before sending. No chip, no attachment. |

**The fallback that always works:** `eval/results/transcript.txt` is a complete
recorded run — nine situations, every route, every citation, with the snapshot
id and model tags in its header. If the live demonstration cannot proceed, that
file is the evidence, and `DECISIONS.md` records that the demonstration is meant
to run from it if latency demands.
