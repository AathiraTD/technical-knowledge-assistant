"""Browser tests for the one surface the CLI cannot exercise.

The web page has behaviour that exists nowhere else — cookies, session reset,
multipart upload, rendered citations — and decision 15 keeps the CLI canonical
precisely so a UI failure never costs the evidence. That division means the
page needs its own thin proof rather than a share of the evaluation harness's.

Opt-in twice, for two different reasons. `playwright` lives in
`requirements-dev.txt` so the clean-clone path stays at five runtime
dependencies, and `ASSISTANT_E2E` gates the run so a missing browser binary
skips these tests instead of failing them. CI and a fresh clone are unaffected.
"""
