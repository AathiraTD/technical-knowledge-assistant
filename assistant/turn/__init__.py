"""The turn as a state machine, and what survives between turns.

`graph.py` holds ordering and nothing else. That restraint is the whole of
decision 20: LangGraph was adopted as a stated exception to decision 4's "no
frameworks", and it earns its place only because the turn genuinely became a
state machine — structured understanding, observations that persist, a
missing-information loop, candidate discovery, evidence sufficiency,
recommendation, verification, correction across turns. The bugs it replaced
were all ordering bugs. No domain logic moved in: the repository, retrieval,
the policy gate, the post-generation checks, the calculations and the audience
filter are called by nodes, not absorbed into them, which is what keeps
decision 4's real claim — that the technical team can inspect every guardrail —
true.

`conversation.py` holds conversation state as typed, provenance-bearing facts,
and `session.py` records what a conversation deliberately forgets. That
distinction is load-bearing rather than tidy: a prior assistant answer must not
re-enter as evidence, and a fact established about one wall must not reach a
question about another.

`session_storage.py` is the persistent backend. Durable checkpointing remains a
documented seam rather than a working adapter — `checkpointer_for()` raises
instead of falling back, because a deployment that believed its conversations
were durable and silently lost them would be worse than one that refuses to
start.
"""
