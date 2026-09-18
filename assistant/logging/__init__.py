"""Capturing what the system could not answer, so that one day it can.

One module, and it exists for a reason that is easy to miss: the hand-off is
not only how a diagnosis question ends safely, it is **the data-collection
mechanism**. DECISIONS 16.1 is explicit that the labelled failure library —
photographs paired with the cause an advisor actually diagnosed — is what
would eventually make a vision model accurate on lime render, and that library
does not exist. It can only be built out of the questions the system refuses.
So every diagnosis hand-off writes down the question, the images, what was
retrieved, and why it declined, leaving a slot for the expert's corrected
answer to be added later.

That last slot is the whole discipline. The model's own answer is never ground
truth; a library built by pairing photographs with what the *system* said would
train it on its own mistakes. Only an advisor's correction counts, which is why
captures written by a test run carry `expert_diagnosis: null` and are capture
noise rather than library.

Deliberately not general-purpose logging. Structured events, correlation ids,
spans and metrics all live in `assistant/infrastructure/observability.py`; this
package is evidence capture for a future training set, and the two are kept
apart because they have different retention, different consumers and a very
different privacy weight — a real captured case holds a photograph of somebody's
own building, which is why `data/failure_library/` is in `.gitignore`.

**That rule is not currently doing its job, and it is worth knowing why.** A
`.gitignore` entry does not untrack what is already tracked, and 35 capture
files committed before the rule was added are still in the repository. They are
harmless — no image bytes, `expert_diagnosis: null` throughout, and developer
questions rather than customers' — but the exposure to watch for is the one
that arrives later, when a real case with a real photograph is written into a
directory somebody believes is ignored.
"""
