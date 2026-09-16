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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import observability as obs, ollama, use_utf8
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

<form method="get" action="/">
  <input type="text" name="q" value="{q}" placeholder="Ask about a product…" autofocus>
  <button type="submit">Ask</button>
</form>
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
        out = [f'<div class="card">{head}',
               f'<div class="answer">{_esc(answer.text)}</div>{tag}']

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
        question = (params.get("q", [""])[0] or "").strip()
        verbose = params.get("v", [""])[0] == "1"
        audience = params.get("a", [""])[0]
        audiences = resolve(audience, self.audiences)

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

        if url.path == "/ask":
            try:
                reply = (self.assistant.ask(asked, audiences=audiences,
                                            correlation_id=self.correlation_id,
                                            carried=carried)
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
                "correlation_id": self.correlation_id,
                "parts": [
                    {"question": q, "path": a.path, "refused": a.refused,
                     "text": a.text, "sources": a.sources, "caveats": a.caveats,
                     "diagnostics": a.diagnostics, "failed_checks": a.failed_checks}
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
                                           carried=carried)
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
        """
        if reply is None:
            return
        slots: dict = {}
        pending = ""
        for _part, answer in reply.parts:
            slots.update(answer.diagnostics.get("slots", {}))
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
