"""The outside world, behind narrow seams.

Everything here is something the answer path needs but must not be coupled to:
the model server (`ollama.py`, two HTTP endpoints and no client library),
structured events and the correlation id that ties one answer's log lines
together (`observability.py`), reading those events back (`trace.py`), deriving
Prometheus text from them rather than adding a second instrumentation path
(`metrics.py`), optional OTLP export (`otel_export.py`), and the readiness
check that a container orchestrator and a person both run (`health.py`).

The seams are narrow on purpose. Liveness and readiness are different
questions, and `health.py` answers the harder one — database reachable, schema
present, repository usable, model available, snapshot compatible — because a
process that is merely running is not a process that can answer.

Storage is deliberately *not* here. It lives behind `KnowledgeRepository` in
`assistant.knowledge`, because the application depends on that boundary rather
than on infrastructure, and an adapter that drifted into this package would be
one import away from the answer engine.
"""
