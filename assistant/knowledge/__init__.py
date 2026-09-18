"""What is known, who may read it, and the boundary in front of both.

`repository.py` defines `KnowledgeRepository`, the single interface the
application depends on for evidence. `model.py` holds the domain types that
cross it. `audience.py` resolves which audience a request is allowed to read
as, and enforces that a request may narrow that set and never widen it.
`store/` holds the two adapters — SQLite for the path that ships and runs
offline, PostgreSQL with pgvector for deployment.

The rule this package exists to enforce is that the answer engine never imports
a database driver. It calls the boundary, infrastructure implements it, and
only composition and bootstrap know which adapter is in play. That is what
makes the assessment path and the deployment path one system rather than two
systems that resemble each other (decision 3).

The audience filter lives down here rather than upstream for a reason worth
saying out loud: filtering happens inside retrieval, against rows, before
anything is ranked or put in front of a model. Restricted evidence is never
fetched and then hidden, because a prompt is not an access control. The two
adapters reach that guarantee by different mechanisms — a `WHERE` clause beside
the distance operator in PostgreSQL, a row filter inside the SQLite adapter
before scoring — and the difference is stated rather than smoothed over,
because only one of them is a `WHERE` clause and it is not the one that ships.
"""
