"""Resolving which audience a request is allowed to read as.

An asserted audience is defensible at a command line, where the person holding
the shell already has the database file. It is not defensible over HTTP, where
the assertion arrives in a query string: the web page read `?a=staff` straight
into the retrieval call, so any visitor could promote themselves to staff and
the audience filter — which is enforced properly in SQL, and tested — could be
stepped around entirely by typing.

The rule here is that a request may only ever **narrow** what the server was
started to allow. It cannot widen it. That keeps the useful behaviour (an
operator running a staff-facing instance can still look at the public view) and
removes the self-service promotion.

This is not authentication and must not be described as one. It moves the
assertion from the caller to the operator, which is where it belongs until
there is an identity to resolve. `CLAUDE.md` is explicit that production
identity determines the audience set later; this is the honest interim.
"""

from __future__ import annotations

from .model import AUDIENCES

DEFAULT = ("public",)


def parse(value: str | None) -> tuple[str, ...]:
    """A comma-separated audience string to a tuple, unknown names dropped.

    Silently dropping an unrecognised name rather than raising, because this
    parses untrusted query strings as well as command-line flags, and an
    unknown audience is a caller mistake rather than a server fault.
    """
    if not value:
        return ()
    wanted = [part.strip().lower() for part in value.split(",")]
    return tuple(dict.fromkeys(a for a in wanted if a in AUDIENCES))


def resolve(requested: str | None, allowed: tuple[str, ...]) -> tuple[str, ...]:
    """What this request may read: the requested set, clipped to the allowed one.

    Falls back to the whole allowed set when nothing usable was asked for, so a
    plain visit to the page behaves as the operator configured rather than
    failing closed into silence.
    """
    permitted = parse(",".join(allowed)) or DEFAULT
    asked = parse(requested)
    narrowed = tuple(a for a in asked if a in permitted)
    return narrowed or permitted
