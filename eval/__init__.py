"""The evaluation harness, and the evidence it produces.

`run.py` drives the transcript situations and the probe suite through the same
CLI path a person uses, so what it measures is what ships rather than a test
double of it. `embedding_choice.py` is the benchmark decision 6 defers to —
the same corpus and the same known-answer questions across candidate embedding
models — and `vision_eval.py` scores the perception stage against the labelled
image fixtures.

Two properties make the output usable as evidence rather than as reassurance.
Every run prints a self-describing header binding it to the snapshot id, the
embedding model and dimension, and the chunking version, so a result cannot be
quoted against an index that did not produce it. And one synthetic staff-tagged
fixture sits in the set purely to be invisible: a public-mode run that retrieves
it is a failed run.

Deliberately offline and deliberately not in CI. The harness needs a real model,
so a green build must mean the guardrails hold rather than that a multi-gigabyte
download succeeded; this is an opt-in local check.
"""
