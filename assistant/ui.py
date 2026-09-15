"""A web page over the same library, served from the standard library.

`python -m assistant.ui`, then open the address it prints.

No Streamlit and no Flask. Streamlit pulls pyarrow, which publishes no Windows
ARM64 wheel and falls back to a source build needing MSVC — a clean-clone
failure on an assessor's machine, for a page that is a form and a list. The
standard library serves both, and the only dependency this adds is zero.

The page is a view. Every routing decision, check and refusal happens in the
library the CLI calls, so the two interfaces cannot disagree — and the
evaluation harness drives the library rather than either of them.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import ollama, use_utf8
from .engine import Assistant
from .repository import IndexMismatch
from .store import EmbeddedRepository
from .store.factory import open_repository

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

    def log_message(self, *args) -> None:      # keep the console for answers
        pass

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path not in ("/", "/ask"):
            self.send_error(404)
            return
        params = parse_qs(url.query)
        question = (params.get("q", [""])[0] or "").strip()
        verbose = params.get("v", [""])[0] == "1"
        audience = params.get("a", ["public"])[0]
        audiences = tuple(a.strip() for a in audience.split(",") if a.strip())

        if url.path == "/ask":
            reply = self.assistant.ask(question, audiences=audiences) if question else None
            payload = {
                "question": question,
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
                reply = self.assistant.ask(question, audiences=audiences)
                body_html = render_html(reply, verbose)
            except ollama.OllamaUnavailable as exc:
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
        page = PAGE.format(meta=self.meta, q=_esc(question), body=body_html,
                           vchecked="checked" if verbose else "",
                           audience_options=options)
        self._send(page.encode("utf-8"), "text/html; charset=utf-8")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
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
    args = parser.parse_args(argv)

    repo = open_repository(args.db)
    try:
        assistant = Assistant(repo)
    except (IndexMismatch, ollama.OllamaUnavailable) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    snapshot = repo.snapshot()
    Handler.assistant = assistant
    Handler.meta = (
        f"{snapshot.document_count} documents · {snapshot.chunk_count} passages · "
        f"index built {snapshot.created_at[:10]} · retrieval "
        f"{snapshot.embedding_model} · composition {ollama.GENERATION_MODEL} · "
        f"abstention threshold {assistant.retriever.threshold}"
    )

    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    address = f"http://{shown}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
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
