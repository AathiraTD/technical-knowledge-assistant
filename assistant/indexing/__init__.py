"""Turning published sources into citable, versioned passages.

The pipeline is a handful of stages and each one is a module: `crawl.py`
fetches the site to disk with its HTTP validators, `extract.py` turns those
cached bytes into sections worth citing, `index.py` chunks, tags caveats,
embeds and publishes a snapshot, `embedcache.py` keeps a content-addressed
cache so a rebuild costs seconds rather than a quarter of an hour, and
`staff_knowledge.py` admits expert-authored JSON through the same versioning
and audience rules as anything crawled.

It is a delta pipeline, not a rebuild wearing a schema — decision 18. The
comparison is made against the content hashes of what is *currently being
served*, because a local ledger can drift from the index and the index cannot
drift from itself. An unchanged document is hashed and then left alone: not
re-extracted, not re-chunked, not re-embedded. A changed one supersedes the
version it replaces, in one transaction, with the old version retained and
linked by `supersedes_id`. A withdrawn one is deactivated and never deleted,
because an answer given while it was live still has to be explicable.

`pipeline.py` is the durable local job queue that lets an external scheduler
drive all of this without running a message broker for a single host.
"""
