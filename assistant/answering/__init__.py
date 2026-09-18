"""Deciding what to say, and proving it may be said.

This package holds the two halves of an answer that are deliberately kept
apart. `router.py` decides — a policy gate, slot detection and eight ordered
steps, none of which is a model — and `answer.py` executes: it extracts a
passage verbatim, or composes over passages and then subjects the result to the
post-generation checks. `engine.py` assembles the two into one question in and
one composite reply out. Keeping the decision separate from the execution is
what lets the router be tested without Ollama running at all.

`understanding.py` and `vision.py` are the two places a model is allowed to
*propose* structure — a turn read into a schema, a photograph read into
observations — and neither is trusted. Both hand their output to deterministic
code that decides whether to believe it. `phrasing.py` is the opposite end: it
reorders an answer that has already passed every check, and returns the
original unchanged unless it can prove nothing material moved.

The boundary worth stating is that nothing in this package reaches a database
driver. Evidence arrives through `KnowledgeRepository`, already filtered to the
caller's audience, so the answer path cannot see material it was not allowed to
retrieve — and cannot be talked into it by a prompt.
"""
