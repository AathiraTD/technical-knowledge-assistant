"""The two ways a person reaches the assistant.

`cli.py` is canonical, and that is a decision rather than an accident — the
transcript and the evaluation harness both run through it, so a failure in the
web page can never cost the evidence. `ui.py` is a thin page over the same
library, served from the standard library so that a clean clone needs no UI
framework to install (decision 15).

Both are wrappers, and proving that is most of the point. The engine is a
library: a question and an audience set in; an answer, its sources, a status
and diagnostics out. A second surface is therefore a wrapper rather than a
rewrite, which is the property the production channel adapters depend on —
building one demonstrates it instead of asserting it.

One asymmetry is worth naming. Over HTTP the audience set arrives in a query
string, so `ui.py` may only ever *narrow* what the operator started the server
with; see `assistant/knowledge/audience.py` for why that rule exists and what
it is not. At a command line the person holding the shell already holds the
database file, and an asserted audience is defensible there.
"""
