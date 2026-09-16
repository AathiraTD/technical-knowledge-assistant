"""A web page over the same library, served from the standard library.

`python -m assistant.ui`, then open the address it prints.

No Streamlit and no Flask. Streamlit pulls pyarrow, which publishes no Windows
ARM64 wheel and falls back to a source build needing MSVC — a clean-clone
failure on an assessor's machine, for a page that is a form and a list. The
standard library serves both, and the only dependency this adds is zero.

The page is a view. Every routing decision, check and refusal happens in the
library the CLI calls, so the two interfaces cannot disagree — and the
evaluation harness drives the library rather than either of them.

Structured logging is on by default here, to stderr, which is the opposite of
the CLI's default and for the opposite reason: this surface produces no
transcript to keep clean, and a server nobody can see is not operable. Each
answered request carries its own correlation id, returned in
`X-Correlation-Id` and in the JSON body, so a reported problem can be found in
the log rather than reproduced. A 404 is the exception: it is served by
`send_error` and never reaches the sender that attaches the header, which is
tolerable because a request for a path that does not exist has no answer to
trace.

This is also the only surface with more than one turn. The CLI is stateless by
design and the harness must stay reproducible, so conversation state lives here
and in `assistant/session.py`: a cookie names the session, the session holds the
facts earlier turns established about the caller's building, and those are
handed to `ask(carried=...)`. The engine merges them under whatever the current
question says, so a correction always wins. The session holds slots and never an
audience — the audience set is resolved per request from what this server was
started to allow, and a session that could widen it would be an access-control
bug with a cookie on it.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import webbrowser
from http.cookies import CookieError, SimpleCookie
from collections import OrderedDict
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs, urlparse

from . import observability as obs, ollama, use_utf8
from .answer import Provenance
from .audience import DEFAULT as PUBLIC_ONLY, resolve
from .engine import Assistant
from .repository import IndexMismatch
from .router import Path_
from .session import SessionStore
from .store.factory import open_repository

# Named for what it is and scoped to this server. HttpOnly because no script on
# the page has any use for it, SameSite=Lax because a session that follows a
# cross-site form post is a session somebody else is steering.
SESSION_COOKIE = "tka_session"

# ------------------------------------------------------------ the upload boundary
#
# An upload is the one place on this surface where an anonymous caller hands the
# process arbitrary bytes, and `CLAUDE.md` names the risks by name: oversized
# input, unsafe filenames, malformed containers, denial of service. Every guard
# below is one of those, and each is enforced here rather than deeper in, so the
# perception stage is never reached by something that should not have got in.
#
# What is deliberately *not* here is any attempt to parse an image. Nothing in
# this file decodes a pixel: the bytes are sniffed for a recognised container
# signature and then handed to Ollama, which is the only thing that reads them.
# Introducing an image library to validate an image would add exactly the
# attack surface — a C decoder fed hostile bytes — that the validation is for.

# The whole request body, headers of the parts included. A photograph from a
# phone is a few megabytes and `assistant/vision.py` refuses anything over eight
# on its own; this is the cap that applies *before* a byte is read, which is the
# only cap that helps against a body that never ends.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# How many photographs one request may carry. The guided visual survey of
# decision 16.1 asks for a handful — a wider elevation, an exposed section, the
# ground line — and a number well past that is not a survey.
MAX_IMAGES_PER_REQUEST = 4

# How many one session may send in total. Perception costs minutes of CPU per
# image on this hardware, so an unbounded count is a denial-of-service path
# rather than a generous allowance. Reached, it stops reading images and says
# so; it never stops answering the question.
MAX_IMAGES_PER_SESSION = 12

# How many sessions the budget remembers. Bounded for the same reason the
# session store is: a per-caller counter reachable by an anonymous caller is
# only a counter while it cannot grow without limit.
MAX_TRACKED_SESSIONS = 1024

# Container signatures, by content rather than by name. The uploaded filename is
# never consulted for anything at all — not for the media type, not for storage,
# not for logging, not for the hand-off — because a filename is a string the
# caller chose and "wall.png" is not evidence that anything is a PNG. These are
# the formats a phone or a laptop actually produces, and the list is an
# allowlist: an unrecognised signature is refused rather than passed along to
# see what happens.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff(data: bytes) -> str:
    """The media type these bytes actually are, or an empty string.

    WEBP is the one signature that is not a simple prefix: it is a RIFF
    container with the format written twelve bytes in, so the check has to look
    at both ends of the header rather than at the first four bytes, which every
    RIFF file shares with a WAV.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for signature, media_type in _SIGNATURES:
        if data.startswith(signature):
            return media_type
    return ""


class UploadBudget:
    """How many photographs each session has sent, bounded and thread-safe.

    Deliberately not in `assistant/session.py`. That module carries what the
    person told us about their building and is argued at length for carrying
    nothing else; a rate limit is a property of this HTTP surface, expires on a
    restart, and means nothing to the CLI. Putting it there would widen a store
    whose narrowness is the point.
    """

    def __init__(self, limit: int = MAX_IMAGES_PER_SESSION,
                 tracked: int = MAX_TRACKED_SESSIONS) -> None:
        self.limit = limit
        self.tracked = tracked
        self._counts: "OrderedDict[str, int]" = OrderedDict()
        self._lock = Lock()

    def take(self, session_id: str, wanted: int) -> int:
        """How many of `wanted` this session may still send, charging for them."""
        with self._lock:
            used = self._counts.get(session_id, 0)
            allowed = max(0, min(wanted, self.limit - used))
            self._counts[session_id] = used + allowed
            self._counts.move_to_end(session_id)
            while len(self._counts) > self.tracked:
                self._counts.popitem(last=False)        # least recently used out
            return allowed

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lime Green technical assistant</title>
<style>
  :root {{
    --ink:#1c2321; --muted:#5d6b66; --line:#dfe5e1; --bg:#f7f9f7;
    --card:#ffffff; --accent:#4a7c59; --warn:#8a5a2b; --warnbg:#fdf6ec;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:780px; margin:0 auto; padding:32px 20px 80px; }}
  header h1 {{ font-size:22px; margin:0 0 6px; }}
  header p {{ color:var(--muted); margin:0 0 4px; font-size:14px; }}
  .meta {{ color:var(--muted); font-size:12.5px; margin-top:10px;
           border-top:1px solid var(--line); padding-top:10px; }}
  form {{ display:flex; gap:8px; margin:22px 0 8px; }}
  input[type=text] {{ flex:1; padding:12px 14px; border:1px solid var(--line);
                      border-radius:8px; font-size:16px; background:var(--card); }}
  input[type=text]:focus {{ outline:2px solid var(--accent); outline-offset:-1px; }}
  button {{ padding:12px 20px; border:0; border-radius:8px; background:var(--accent);
            color:#fff; font-size:15px; cursor:pointer; }}
  button:hover {{ background:#3d6749; }}
  .working {{ background:var(--warnbg); border:1px solid var(--warn);
              border-radius:10px; padding:14px 18px; margin-bottom:18px;
              color:var(--warn); }}
  .working p {{ margin:6px 0 0; font-size:13px; }}
  button[disabled] {{ opacity:.6; cursor:progress; }}
  .opts {{ color:var(--muted); font-size:13px; margin-bottom:26px; }}
  .opts label {{ margin-right:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:18px 20px; margin-bottom:16px; }}
  .part {{ font-size:12.5px; text-transform:uppercase; letter-spacing:.06em;
           color:var(--muted); margin-bottom:10px; }}
  .answer {{ white-space:pre-wrap; }}
  .tag {{ display:inline-block; font-size:12px; padding:2px 9px; border-radius:99px;
          background:#eef3ef; color:var(--accent); margin-left:8px; }}
  .tag.refused {{ background:#fdecec; color:#9b3b3b; }}
  h3 {{ font-size:13px; text-transform:uppercase; letter-spacing:.06em;
        color:var(--muted); margin:18px 0 8px; }}
  ol.src {{ margin:0; padding-left:20px; font-size:14px; }}
  ol.src li {{ margin-bottom:7px; }}
  ol.src a {{ color:var(--accent); word-break:break-all; }}
  .caveats {{ background:var(--warnbg); border-left:3px solid var(--warn);
              padding:10px 14px; font-size:14px; border-radius:0 6px 6px 0; }}
  .caveats li {{ margin-bottom:5px; }}
  .diag {{ font:12.5px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;
           color:var(--muted); background:#f4f6f4; border-radius:6px;
           padding:10px 12px; margin-top:14px; white-space:pre-wrap; }}
  .empty {{ color:var(--muted); font-size:14px; }}
  details.passage {{ border:1px solid var(--line); border-radius:8px;
                     padding:10px 14px; margin-top:14px; background:#fbfcfb; }}
  details.passage summary {{ cursor:pointer; font-size:13px; color:var(--muted); }}
  details.passage .quote {{ white-space:pre-wrap; font-size:14px; margin-top:10px;
                            color:var(--ink); }}
  ol.hist {{ margin:0; padding-left:20px; font-size:14px; color:var(--muted); }}
  ol.hist li {{ margin-bottom:9px; }}
  ol.hist b {{ color:var(--ink); font-weight:600; }}
</style>
<div class="wrap">
<header>
  <h1>Lime Green technical assistant</h1>
  <p>Answers only from Lime Green's published material, cites every source by
     document name, and refuses when the material does not answer the question.</p>
  <div class="meta">{meta}</div>
</header>

<form method="get" action="/" onsubmit="working()">
  <input type="text" name="q" value="{q}" placeholder="Ask about a product…" autofocus>
  <button type="submit" id="ask">Ask</button>
</form>
<!-- A second form rather than one that posts everything. The ordinary question
     stays a GET so the answer keeps a shareable address and the back button
     works, which is how every existing link and the evaluation harness reach
     this page; an upload cannot be a GET, so it gets its own. -->
<details class="passage">
  <summary>Ask about a photograph</summary>
  <form method="post" action="/" enctype="multipart/form-data" onsubmit="working()">
    <input type="text" name="q" placeholder="What would you like to know about it?">
    <input type="file" name="image" accept="image/*" multiple>
    <button type="submit">Ask</button>
  </form>
  <p class="empty">The photograph is read for what is visible in it — what the
     wall is built of, inside or outside, exposure, a symptom — and for nothing
     else. It never chooses a product and never decides what has gone wrong:
     that stays a judgement for the technical team. Anything read this way is
     reported as coming from the photograph rather than as something you said,
     and the image is not stored. Expect a wait of minutes: a vision model on a
     processor with no graphics card is slow.</p>
</details>
<div class="working" id="working" hidden>
  <strong>Thinking…</strong> <span id="elapsed">0s</span>
  <p>A question the model has not seen before takes tens of seconds on a
     processor with no graphics card — prompt reading is the cost, not typing
     the answer. Asking the same question again returns immediately.</p>
</div>
<div class="opts">
  <label><input type="checkbox" name="v" form="" onchange="toggle('v',this)"
    {vchecked}> show how it was answered</label>
  <label>audience:
    <select onchange="setAudience(this.value)">
      {audience_options}
    </select>
  </label>
  <span>asserted here, authenticated in production</span>
</div>

{body}
</div>
<script>
function param(k,v){{const u=new URL(location);v?u.searchParams.set(k,v):u.searchParams.delete(k);location=u;}}
function toggle(k,el){{param(k, el.checked?'1':'');}}
function setAudience(v){{param('a',v);}}
// The form is a plain GET, so the browser shows nothing at all until the
// server answers — and on this hardware that is tens of seconds. Silence for
// that long is indistinguishable from a broken button, which is exactly how it
// was first reported. The counter is the point: it says the wait is real work
// rather than a hang, and it degrades to an ordinary form if scripting is off.
function working(){{
  var box = document.getElementById('working');
  var out = document.getElementById('elapsed');
  var btn = document.getElementById('ask');
  if (!box) return;
  box.hidden = false;
  if (btn) {{ btn.disabled = true; btn.textContent = 'Asking…'; }}
  var t0 = Date.now();
  setInterval(function(){{
    out.textContent = Math.round((Date.now() - t0) / 1000) + 's';
  }}, 1000);
}}
</script>
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_history(turns: list[tuple[str, str]]) -> str:
    """The conversation so far, so a follow-up reads as one.

    Without it the page answers "brick" with a wall of text and no sign of the
    question it belongs to, which is the single-turn experience decision 10
    complains about wearing a session cookie.
    """
    if not turns:
        return ""
    rows = "".join(f"<li><b>{_esc(q)}</b><br>{_esc(a)}</li>" for q, a in turns)
    return (f"<div class='card'><div class='part'>Earlier in this conversation"
            f"</div><ol class='hist'>{rows}</ol></div>")


def render_html(reply, verbose: bool) -> str:
    blocks = []
    multi = len(reply.parts) > 1
    for i, (part, answer) in enumerate(reply.parts, 1):
        head = (f'<div class="part">Part {i} — {_esc(part)}</div>' if multi else "")
        tag = ('<span class="tag refused">refused</span>' if answer.refused
               else f'<span class="tag">{_esc(answer.path)}</span>')
        # `body`, not `text`: on a refusal the raw passage is the tail of the
        # text, and leading a public visitor with 600 characters of datasheet is
        # the complaint this disclosure answers. It is shown in full below,
        # folded, so nothing the refusal carries is lost — only demoted.
        out = [f'<div class="card">{head}',
               f'<div class="answer">{_esc(answer.body)}</div>{tag}']

        if answer.disclosure:
            # Open on the diagnostics view, where the reader is auditing rather
            # than asking, and closed for the visitor who only wanted a sentence.
            out.append(f"<details class='passage'{' open' if verbose else ''}>"
                       f"<summary>Show source passage</summary>"
                       f"<div class='quote'>{_esc(answer.disclosure)}</div></details>")

        # Assumed means assumed. A value the caller supplied is reported inside
        # the answer text by `AnswerEngine._finish` as something they said, and
        # deliberately not repeated here under a heading that would call it a
        # guess — that mislabelling is the defect this block used to carry.
        if answer.assumptions:
            out.append("<h3>Assumed</h3><div class='empty'>"
                       + _esc("; ".join(answer.assumptions)) + "</div>")

        if answer.caveats:
            items = "".join(f"<li>{_esc(c)}</li>" for c in answer.caveats)
            out.append(f"<h3>Also published about these products</h3>"
                       f"<ul class='caveats'>{items}</ul>")

        if answer.sources:
            rows = "".join(
                f"<li>{_esc(s['name'])}"
                + (f" — {_esc(s['section'])}" if s["section"] else "")
                + (f" ({_esc(s['date'])})" if s["date"] else "")
                + f"<br><a href='{_esc(s['url'])}' rel='noreferrer'>{_esc(s['url'])}</a></li>"
                for s in answer.sources)
            out.append(f"<h3>Sources</h3><ol class='src'>{rows}</ol>")

        if verbose:
            d = answer.diagnostics
            lines = [f"path      {answer.path}",
                     f"step      {d.get('step', '-')}",
                     f"why       {d.get('reason', d.get('refusal_reason', '-'))}",
                     f"top score {d.get('top_score', 0)}",
                     f"slots     {d.get('slots', {})}"]
            if d.get("generation_seconds"):
                lines.append(f"generated {d['generation_seconds']}s")
            if answer.failed_checks:
                lines.append("checks that failed:")
                lines += [f"  - {c}" for c in answer.failed_checks]
            out.append(f"<div class='diag'>{_esc(chr(10).join(lines))}</div>")

        out.append("</div>")
        blocks.append("".join(out))
    return "".join(blocks)


class Handler(BaseHTTPRequestHandler):
    assistant: Assistant
    meta: str
    # What this server was started to allow. A request may narrow this and can
    # never widen it, so `?a=staff` against a public instance stays public.
    audiences: tuple[str, ...] = PUBLIC_ONLY
    # Conversation state, shared by every request thread. Slots only; see
    # assistant/session.py for what is carried and what is deliberately not.
    sessions: SessionStore = SessionStore()
    # How many photographs each session has sent. Separate from the session
    # store on purpose — see UploadBudget.
    uploads: UploadBudget = UploadBudget()
    correlation_id: str
    session_id: str

    def _session(self) -> str:
        """The session this request belongs to, minting one if it has none.

        The Cookie header is attacker-controlled, and `SimpleCookie` raises on a
        malformed key rather than ignoring it, so a hand-written header of
        "=====" would otherwise answer every request with a 500. An unreadable
        cookie is treated as no cookie, which is the safe reading: it starts a
        new conversation instead of guessing which one was meant.
        """
        try:
            jar = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            jar = SimpleCookie()
        morsel = jar.get(SESSION_COOKIE)
        return self.sessions.open(morsel.value if morsel else "")

    def log_message(self, *args) -> None:      # keep the console for answers
        pass

    def do_GET(self) -> None:
        # One id per request, established before anything can answer and handed
        # back on every response. It is what turns "it refused and I do not know
        # why" into a line in the log: the caller quotes the id, the operator
        # greps for it, and every event the answer produced comes back together.
        self.correlation_id = obs.new_id()
        url = urlparse(self.path)
        if url.path not in ("/", "/ask"):
            self.send_error(404)
            return
        params = parse_qs(url.query)
        self._respond(url.path,
                      (params.get("q", [""])[0] or "").strip(),
                      params.get("v", [""])[0] == "1",
                      params.get("a", [""])[0], [], [])

    def do_POST(self) -> None:
        """The same answer, for a request that carries photographs.

        Everything after the boundary checks is the GET path unchanged, which is
        the property worth having: an upload adds evidence to a question and
        adds no answer route, no second renderer and no second place for the
        audience set to be resolved.
        """
        self.correlation_id = obs.new_id()
        url = urlparse(self.path)
        if url.path not in ("/", "/ask"):
            # Drained first. A request whose body is never read is answered on a
            # connection the client is still writing into, and what the client
            # then sees is a reset rather than the 404 — the status code is
            # correct and unreadable, which is the worst of both. It showed up
            # as an intermittently failing test rather than as a bug report,
            # which is exactly what a race looks like from the outside.
            self._drain()
            self.send_error(404)
            return
        # Established before the body is read, because the per-session image
        # budget is one of the things deciding how much of that body is worth
        # looking at.
        self.session_id = self._session()
        fields, images, notes = self._read_upload()
        if fields is None:
            return                              # already answered with a status
        params = parse_qs(url.query)
        self._respond(url.path, fields.get("q", "").strip(),
                      params.get("v", [""])[0] == "1" or fields.get("v") == "1",
                      params.get("a", [""])[0] or fields.get("a", ""),
                      images, notes, session_open=True)

    def _drain(self) -> bool:
        """Read and discard a declared body, so a refusal can be read back.

        Returns whether it managed to. It refuses to drain a body larger than
        the cap, because reading twelve megabytes in order to say "that is too
        large" would be doing the work the cap exists to avoid — that case
        closes the connection instead, which is the one honest way to stop a
        client mid-upload. A body with no declared length cannot be drained at
        all, for the same reason it cannot be accepted.
        """
        length = self.headers.get("Content-Length", "")
        if not length.isdigit() or int(length) > MAX_UPLOAD_BYTES:
            self.close_connection = True
            return False
        remaining = int(length)
        while remaining > 0:
            block = self.rfile.read(min(remaining, 64 * 1024))
            if not block:
                break
            remaining -= len(block)
        return True

    def _read_upload(self):
        """The posted fields and the photographs in them, or a refusal.

        Returns `(fields, images, notes)`, or `(None, [], [])` when the request
        has already been answered with a status code. `notes` are the things the
        caller should be told about their own upload — a file that is not an
        image, a count over the cap — because silently dropping an attachment
        and answering as though none was sent is the kind of quiet failure this
        codebase refuses everywhere else.

        The order of the guards is the point of the method. The length is
        checked before a byte is read, so an oversized body is refused rather
        than buffered; the declared type is checked before the body is parsed;
        the content is sniffed before anything is treated as an image; and the
        filename is discarded at the only moment it is ever visible.
        """
        declared = self.headers.get("Content-Type", "")
        length = self.headers.get("Content-Length", "")
        if not length.isdigit():
            # No length, or a length that is not a number. Reading until the
            # connection closes is exactly the unbounded read the cap exists to
            # prevent, so this is refused rather than guessed at.
            self._drain()               # cannot, so it closes the connection
            self._send(b"a length is required", "text/plain; charset=utf-8",
                       status=411)
            return None, [], []
        size = int(length)
        if size > MAX_UPLOAD_BYTES:
            obs.event("upload_rejected", reason="too large", bytes=size)
            self._drain()               # declines, and closes the connection
            self._send(b"that is larger than this server accepts",
                       "text/plain; charset=utf-8", status=413)
            return None, [], []
        if not declared.lower().startswith("multipart/form-data"):
            self._drain()
            self._send(b"expected a multipart form",
                       "text/plain; charset=utf-8", status=415)
            return None, [], []

        # Parsed by the standard library's own MIME parser rather than by a
        # hand-written boundary splitter. A multipart body is a MIME body, this
        # is the parser that already ships, and a second implementation of it
        # here would be a new place to get a length or a delimiter wrong.
        raw = self.rfile.read(size)
        message = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + declared.encode("latin-1") + b"\r\n\r\n" + raw)
        if not message.is_multipart():
            self._send(b"expected a multipart form",
                       "text/plain; charset=utf-8", status=415)
            return None, [], []

        fields: dict[str, str] = {}
        candidates: list[bytes] = []
        notes: list[str] = []
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            payload = part.get_payload(decode=True) or b""
            # A part with a filename is an attachment whatever it claims to be,
            # and a part without one is a form field. The filename itself is
            # read for this test and then dropped on the floor: it is never
            # stored, never logged, never echoed and never used to decide a
            # type. A caller-supplied string that reaches a path or a page is
            # the traversal and injection risk `CLAUDE.md` names.
            if part.get_filename() is not None:
                candidates.append(payload)
            elif name:
                fields[name] = payload.decode("utf-8", "replace")

        candidates = [c for c in candidates if c]
        if len(candidates) > MAX_IMAGES_PER_REQUEST:
            notes.append(f"Only the first {MAX_IMAGES_PER_REQUEST} photographs "
                         "were read; the rest were ignored.")
            candidates = candidates[:MAX_IMAGES_PER_REQUEST]

        images: list[bytes] = []
        for data in candidates:
            media_type = sniff(data)
            if not media_type:
                # Named by shape, not by filename, and the filename is not
                # quoted back at the caller either: echoing it would put a
                # string they chose into the page.
                notes.append("One attachment was not a recognised image and was "
                             "not read.")
                obs.event("upload_rejected", reason="unrecognised container",
                          bytes=len(data))
                continue
            images.append(data)

        allowed = self.uploads.take(self.session_id, len(images))
        if allowed < len(images):
            notes.append("This conversation has sent as many photographs as the "
                         "server accepts, so the rest were not read.")
            obs.event("upload_rejected", reason="session cap",
                      dropped=len(images) - allowed)
            images = images[:allowed]
        return fields, images, notes

    def _respond(self, path: str, question: str, verbose: bool, audience: str,
                 images: list, notes: list, session_open: bool = False) -> None:
        audiences = resolve(audience, self.audiences)

        if not session_open:
            self.session_id = self._session()
        carried = self.sessions.carried(self.session_id)
        earlier = self.sessions.turns(self.session_id)
        pending = self.sessions.pending(self.session_id)
        asked = question
        if pending and question:
            # The previous turn ended in an ask-back, so this one may be the
            # answer to it rather than a new question. Detection is the router's
            # own vocabulary — not a second copy of it here — and only the slot
            # step 5 actually asked for counts: "brick" resumes the pending
            # question, "and what colour is it" does not.
            stated = self.assistant.router.slots.detect(question)
            if "substrate" in stated:
                carried = {**carried, **stated}
                asked = pending

        if path == "/ask":
            try:
                reply = (self.assistant.ask(asked, audiences=audiences,
                                            correlation_id=self.correlation_id,
                                            carried=carried, images=images)
                         if question else None)
            except (ollama.OllamaUnavailable, IndexMismatch) as exc:
                # The HTML branch has always handled this; the JSON branch did
                # not, so an unreachable model answered a request with a
                # stack trace and no status code worth acting on.
                self._send(json.dumps({"error": str(exc), "question": question,
                                       "correlation_id": self.correlation_id},
                                      ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8", status=503)
                return
            self._remember(question, reply)
            payload = {
                "question": question,
                "answered": asked,
                "session_slots": self.sessions.carried(self.session_id),
                "images_read": len(images),
                "upload_notes": notes,
                "correlation_id": self.correlation_id,
                "parts": [
                    {"question": q, "path": a.path, "refused": a.refused,
                     "text": a.text, "sources": a.sources, "caveats": a.caveats,
                     "diagnostics": a.diagnostics, "failed_checks": a.failed_checks,
                     # Structured rather than prose, so a channel adapter can
                     # render provenance its own way instead of parsing ours.
                     "facts": [{"slot": f.slot, "value": f.value,
                                "provenance": f.provenance.value}
                               for f in a.facts],
                     "assumptions": a.assumptions,
                     # Its own group, never folded into the assumptions: a
                     # reading from a photograph is not a guess, and the heading
                     # over `assumptions` says it is.
                     "observed": a.observed,
                     "disclosure": a.disclosure}
                    for q, a in (reply.parts if reply else [])
                ],
            }
            body = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
            return

        if question:
            try:
                reply = self.assistant.ask(asked, audiences=audiences,
                                           correlation_id=self.correlation_id,
                                           carried=carried, images=images)
                self._remember(question, reply)
                body_html = render_html(reply, verbose)
            except (ollama.OllamaUnavailable, IndexMismatch) as exc:
                body_html = (f"<div class='card'><div class='answer'>"
                             f"{_esc(str(exc))}</div></div>")
        else:
            body_html = ("<div class='card'><div class='empty'>Ask a question above. "
                         "Try “how much water does Solo need per bag”, or something "
                         "the site does not publish — it will say so rather than "
                         "guess.</div></div>")

        options = "".join(
            f"<option value='{a}'{' selected' if a == audience else ''}>{a}</option>"
            for a in ("public", "trade", "staff"))
        if notes:
            body_html = ("<div class='card'><div class='empty'>"
                         + "".join(_esc(n) + "<br>" for n in notes)
                         + "</div></div>") + body_html
        page = PAGE.format(meta=self.meta, q=_esc(question),
                           body=render_history(earlier) + body_html,
                           vchecked="checked" if verbose else "",
                           audience_options=options)
        self._send(page.encode("utf-8"), "text/html; charset=utf-8")

    def _remember(self, question: str, reply) -> None:
        """Fold what this turn established back into the session.

        The slots taken are the ones the router actually detected, read off the
        answer rather than re-derived, so the page and the answer cannot
        disagree about what was assumed. `pending` is set only by an ask-back
        and cleared by anything else, so a question that was answered never
        resumes later.

        **A slot read off a photograph is dropped here, and that is the rule the
        whole vision seam rests on.** `diagnostics["slots"]` is the router's
        merged view, so once an upload can fill a slot it carries values nobody
        said — and folding those into the session would persist a model's
        uncalibrated reading of an image as though the caller had stated it, on
        every later turn, invisibly, after the image has scrolled out of the
        page. The provenance recorded on the answer is what distinguishes them,
        which is why the exclusion reads the facts rather than the slots: a slot
        the photograph supplied *and* the question stated comes back as
        ``STATED`` and is kept, because the person did say it.

        The cost is real and is the accepted one: a follow-up about the same
        wall does not inherit what the photograph showed, so the caller may be
        asked back for something the image settled. If that proves annoying the
        fix is an expiring fourth state, not a quiet promotion into the session.
        """
        if reply is None:
            return
        slots: dict = {}
        pending = ""
        for _part, answer in reply.parts:
            observed = {fact.slot for fact in answer.facts
                        if fact.provenance is Provenance.OBSERVED}
            slots.update({name: value
                          for name, value in answer.diagnostics.get("slots", {}).items()
                          if name not in observed})
            if answer.path == Path_.ASK_BACK.value:
                pending = reply.question
        summary = "\n\n".join(answer.text for _part, answer in reply.parts)
        self.sessions.remember(self.session_id, question, summary, slots, pending)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Correlation-Id", self.correlation_id)
        # Path and HttpOnly on every response, including the first: the session
        # is established before the answer, so a caller who asks one question
        # and reads the reply already has somewhere to put the next turn.
        self.send_header("Set-Cookie",
                         f"{SESSION_COOKIE}={self.session_id}; Path=/; "
                         f"HttpOnly; SameSite=Lax")
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(prog="assistant.ui")
    parser.add_argument("-p", "--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; 0.0.0.0 inside a container")
    parser.add_argument("--db", default="data/index/knowledge.db")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--allow-audience", default="public",
        help="which audiences this server may read, comma separated. A request "
             "can narrow this and never widen it, so the default refuses to "
             "serve staff material however the URL is written.")
    args = parser.parse_args(argv)
    # On by default here, unlike the CLI: a server produces no transcript to
    # protect, and one that logs nothing cannot be operated. stderr keeps the
    # address and the JSON lines on separate streams.
    obs.configure(sys.stderr)

    # Threaded server, one shared store: see assistant/store/locking.py.
    repo = open_repository(args.db, thread_safe=True)
    try:
        assistant = Assistant(repo)
    except (IndexMismatch, ollama.OllamaUnavailable) as exc:
        # Let the store go before giving up on it. A refused start is followed
        # by a rebuild into the same file, and a connection left open is one
        # the operating system is still holding.
        repo.close()
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    snapshot = repo.snapshot()
    Handler.assistant = assistant
    # One store per server, rather than the class default, so a restarted
    # process never inherits a conversation from the last one.
    Handler.sessions = SessionStore()
    Handler.uploads = UploadBudget()
    Handler.audiences = resolve(args.allow_audience, ("public", "trade", "staff"))
    Handler.meta = (
        f"{snapshot.document_count} documents · {snapshot.chunk_count} passages · "
        f"index built {snapshot.created_at[:10]} · retrieval "
        f"{snapshot.embedding_model} · composition {ollama.GENERATION_MODEL} · "
        f"abstention threshold {assistant.retriever.threshold}"
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    # The port actually bound, not the one that was asked for. They differ when
    # the port is left to the operating system, and the address is both printed
    # and opened in a browser — so reporting the request rather than the bind
    # advertises an address nothing is listening on.
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    address = f"http://{shown}:{server.server_address[1]}/"
    print(f"Lime Green technical assistant on {address}")
    print("JSON for the same question at /ask?q=...   Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
