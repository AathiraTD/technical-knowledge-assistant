# Gold evaluation — Lime Green technical assistant

    Lime Green technical assistant — evaluation transcript
    snapshot       snap-2fc8668266cd440b9e18f3fdbe6298f1
    built          2026-09-16T12:30:41+00:00
    embedding      qwen3-embedding:0.6b (1024d)
    chunking       structure-aware/1.0
    documents      95
    chunks         552
    threshold      0.45
    run at         2026-09-17T15:49:14+00:00

**Scenarios 29/32**  
Correct answers 18/19  
Correct abstentions 5/5  
Correct hand-offs 5/7  
Correct ask-backs 1/1  

Over-refusals (answerable, refused): 6 — GB2, GC2, GM2, GM4, GM5, GM6
Unsupported answers (unanswerable, answered): 0
Multi-turn scenarios: 6, conversation-state failures: 0
Not evaluated (the model server failed): 0

## Failed expectations by stage

| stage | failed expectations |
|---|---|
| generation | 4 |
| verification | 1 |

## Scenarios

| id | category | expects | result | route |
|---|---|---|---|---|
| GA1 | factual_lookup | answer | pass | compose |
| GA2 | factual_lookup | answer | pass | compose |
| GA3 | factual_lookup | answer | pass | compose |
| GB1 | suitability | answer | pass | compose |
| GB2 | suitability | answer | pass | refuse |
| GB3 | compatibility | handoff | pass | refuse |
| GC1 | recommendation | ask_back | pass | ask back |
| GC2 | recommendation | answer | pass | ask back; refuse |
| GD1 | preparation | answer | pass | cited hand-off |
| GD2 | multi_property | answer | pass | compose |
| GE1 | multi_turn_quantity | answer | pass | compose; extract |
| GE2 | quantity | answer | pass | extract |
| GG1 | troubleshooting | handoff | pass | diagnosis |
| GG2 | troubleshooting | handoff | **FAIL** | cited hand-off; refuse |
| GH1 | ambiguous | abstain | pass | refuse |
| GJ1 | correction | answer | pass | compose |
| GJ2 | correction | answer | pass | compose |
| GK1 | unsupported | abstain | pass | refuse |
| GK2 | unsupported | abstain | pass | refuse |
| GK3 | unsupported | handoff | **FAIL** | compose |
| GL1 | policy | handoff | pass | route |
| GL2 | policy | handoff | pass | route |
| GL3 | policy | handoff | pass | route |
| GM1 | false_premise | answer | pass | extract |
| GM2 | false_premise | answer | pass | refuse |
| GM3 | adversarial | abstain | pass | refuse |
| GM4 | adversarial | answer | pass | refuse |
| GM5 | adversarial | answer | pass | refuse |
| GM6 | adversarial | answer | pass | refuse |
| GM7 | cross_product | answer | pass | compose |
| GM8 | cross_product | answer | **FAIL** | compose |
| GM9 | audience | abstain | pass | refuse |

## What failed, and where

### GG2 — Render cracking, with the symptom slot dropped afterwards

*turn 2*: What is the coverage of Solo Onecoat?
  - [generation] refused=True, expected False
  - [verification] nothing that answered carried a citation
  - [generation] answer contains none of ['1.5m2 at 10mm', '3m2 at 5mm', '1.6kg']

### GK3 — A published deferral beats a computed quantity

*turn 1*: How much Solo Onecoat do I need for an MgO board wall?
  - [generation] answer contains none of ['mgo', 'contact us']

### GM8 — Stale conversation state must not answer a new question

*turn 2*: And what coats does Duro go on in?
  - [generation] answer contains none of ['10 to 15mm']

