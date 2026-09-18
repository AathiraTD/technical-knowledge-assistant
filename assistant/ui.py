"""A web page over the same library, served from the standard library.

`python -m assistant.ui`, then open the address it prints.

No Streamlit and no Flask. Streamlit pulls pyarrow, which publishes no Windows
ARM64 wheel and falls back to a source build needing MSVC -- a clean-clone
failure on an assessor's machine, for a page that is a form and a list. The
standard library serves both, and the only dependency this adds is zero.

The page is a view. Every routing decision, check and refusal happens in the
library the CLI calls, so the two interfaces cannot disagree -- and the
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
audience -- the audience set is resolved per request from what this server was
started to allow, and a session that could widen it would be an access-control
bug with a cookie on it.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
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

from . import health, metrics, observability as obs, ollama, use_utf8
from .answer import Provenance
from .audience import DEFAULT as PUBLIC_ONLY, resolve
from .engine import Assistant
from .repository import IndexMismatch
from .router import Path_
from .session import SessionStore
from .store.factory import open_repository, open_persisted_session_store

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
# attack surface -- a C decoder fed hostile bytes -- that the validation is for.

# The whole request body, headers of the parts included. A photograph from a
# phone is a few megabytes and `assistant/vision.py` refuses anything over eight
# on its own; this is the cap that applies *before* a byte is read, which is the
# only cap that helps against a body that never ends.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# How many photographs one request may carry. The guided visual survey of
# decision 16.1 asks for a handful -- a wider elevation, an exposed section, the
# ground line -- and a number well past that is not a survey.
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
# never consulted for anything at all -- not for the media type, not for storage,
# not for logging, not for the hand-off -- because a filename is a string the
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
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ink:#f5f5f4; --muted:#a0aaa5; --line:#3a4440; --bg:#1a1f1d;
      --card:#252a28; --accent:#5a8c69; --warn:#b8845a; --warnbg:#3a2f1f;
    }}
  }}
  :root[data-theme="dark"] {{
    --ink:#f5f5f4; --muted:#a0aaa5; --line:#3a4440; --bg:#1a1f1d;
    --card:#252a28; --accent:#5a8c69; --warn:#b8845a; --warnbg:#3a2f1f;
  }}

  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:0; background:var(--bg); color:var(--ink);
         font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         display:flex; flex-direction:column; height:100vh; }}

  header {{
    background:var(--card); border-bottom:1px solid var(--line);
    padding:16px 20px; display:flex; justify-content:space-between;
    align-items:center; gap:16px;
  }}
  header h1 {{ font-size:18px; margin:0; flex:1; }}
  .header-controls {{ display:flex; gap:12px; align-items:center; }}
  .audience-badge {{
    font-size:12px; padding:4px 10px; border-radius:99px;
    background:#eef3ef; color:var(--accent); white-space:nowrap;
  }}
  .new-chat-btn {{
    padding:8px 14px; font-size:14px; border:1px solid var(--line);
    background:var(--card); color:var(--accent); border-radius:6px;
    cursor:pointer; transition:all 0.2s;
  }}
  .new-chat-btn:hover {{ background:var(--line); }}

  .chat-container {{ flex:1; display:flex; flex-direction:column;
                      overflow:hidden; }}
  .chat-messages {{ flex:1; overflow-y:auto; padding:20px;
                    display:flex; flex-direction:column; gap:12px; }}
  .answer-response {{ display:flex; flex-direction:column; gap:12px; }}

  .landing {{ display:flex; flex-direction:column; align-items:center;
             justify-content:center; text-align:center; padding:40px 20px; }}
  .landing h2 {{ font-size:32px; margin:0 0 12px; }}
  .landing p {{ color:var(--muted); margin:0; max-width:400px;
               font-size:16px; }}

  .message {{ display:flex; gap:12px; animation:slideIn 0.3s ease-out; }}
  @keyframes slideIn {{ from {{ opacity:0; transform:translateY(8px); }}
                       to {{ opacity:1; transform:translateY(0); }} }}

  .message.user {{ justify-content:flex-end; }}
  .message.assistant {{ justify-content:flex-start; }}

  .message-bubble {{
    max-width:75%; padding:12px 14px; border-radius:12px;
    word-wrap:break-word;
  }}
  .message.user .message-bubble {{
    background:var(--accent); color:#fff;
  }}
  .message.assistant .message-bubble {{
    background:var(--card); border:1px solid var(--line);
    color:var(--ink);
  }}

  .answer-text {{ margin:0; white-space:pre-wrap; line-height:1.6; }}
  .citation {{ cursor:pointer; color:var(--accent); font-weight:500;
             border-bottom:1px solid var(--accent); padding:0 1px; }}
  .citation:hover {{ text-decoration:underline; }}

  .sources-disclosure {{
    margin-top:12px; border-top:1px solid var(--line); padding-top:8px;
  }}
  .disclosure-btn {{
    background:none; border:none; color:var(--accent); cursor:pointer;
    font-size:13px; padding:0; text-align:left; font-weight:500;
  }}
  .disclosure-btn:hover {{ text-decoration:underline; }}
  .disclosure-btn::before {{
    content:"▶ "; display:inline-block; transition:transform 0.2s;
    margin-right:4px;
  }}
  .disclosure-btn.open::before {{ transform:rotate(90deg); }}

  .source-list {{
    display:none; margin-top:8px; padding:8px;
    background:var(--line); border-radius:6px;
    font-size:13px;
  }}
  .source-list.open {{ display:block; }}
  .source-item {{
    margin-bottom:8px; padding-bottom:8px; border-bottom:1px solid #ddd;
  }}
  .source-item:last-child {{ border-bottom:none; margin-bottom:0; padding-bottom:0; }}
  .source-name {{ font-weight:500; color:var(--ink); }}
  .source-date {{ color:var(--muted); font-size:12px; }}
  .source-link {{ color:var(--accent); word-break:break-all; font-size:12px;
                 display:block; margin-top:4px; }}

  .diagnostics-disclosure {{
    margin-top:10px; border-top:1px solid var(--line); padding-top:8px;
  }}
  .diagnostics-list {{
    display:none; margin-top:8px; padding:8px;
    background:var(--line); border-radius:6px; font-size:12px;
    font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  }}
  .diagnostics-list.open {{ display:block; }}
  .diag-line {{ margin-bottom:4px; }}
  .diag-key {{ color:var(--muted); min-width:120px; display:inline-block; }}

  .tag {{ display:inline-block; font-size:11px; padding:2px 7px;
         border-radius:99px; background:#eef3ef; color:var(--accent);
         margin-left:8px; vertical-align:middle; }}
  .tag.refused {{ background:#fdecec; color:#9b3b3b; }}

  .input-area {{
    background:var(--card); border-top:1px solid var(--line);
    padding:12px 16px; display:flex; gap:8px; flex-wrap:wrap;
    align-items:flex-start;
  }}
  .input-wrapper {{
    flex:1; display:flex; gap:8px; min-height:40px; align-items:flex-end;
  }}
  .input-field {{
    flex:1; padding:10px 12px; border:1px solid var(--line);
    border-radius:6px; font-size:15px; background:var(--bg);
    color:var(--ink); font-family:inherit;
  }}
  .input-field:focus {{ outline:2px solid var(--accent); outline-offset:-1px; }}
  .attach-btn, .send-btn {{
    padding:8px 12px; border:none; border-radius:6px; cursor:pointer;
    font-size:16px; transition:all 0.2s;
  }}
  .attach-btn {{
    background:transparent; color:var(--muted); border:1px solid var(--line);
  }}
  .attach-btn:hover {{ background:var(--line); color:var(--ink); }}
  .send-btn {{
    background:var(--accent); color:#fff; padding:10px 14px;
  }}
  .send-btn:hover {{ background:#3d6749; }}
  .send-btn:disabled {{ opacity:0.5; cursor:not-allowed; }}

  .input-extras {{
    width:100%; font-size:12px; color:var(--muted);
    display:flex; justify-content:space-between; align-items:center;
    padding:0 0 8px;
  }}
  .input-info {{ display:flex; gap:12px; }}
  .upload-notes {{
    color:var(--warn); font-size:11px; margin:0; padding:0;
  }}

  /* The photograph, as the page reports it back. */
  .attached {{
    margin-top:6px; font-size:11px; opacity:.85;
  }}
  ul.perception {{
    list-style:none; margin:8px 0 0; padding:0;
    font-size:13px; line-height:1.9;
  }}
  ul.perception li {{ margin:0; }}
  .pill {{
    display:inline-block; min-width:74px; text-align:center;
    border:1px solid currentColor; border-radius:999px;
    font-size:10px; letter-spacing:.04em; text-transform:uppercase;
    padding:1px 7px; margin-right:8px; vertical-align:1px;
  }}
  .withheld {{ color:var(--muted); font-size:12px; font-style:italic; }}
  .perception-none {{
    margin-top:8px; font-size:13px; color:var(--muted);
  }}
  .cannot {{ margin-top:10px; font-size:12px; color:var(--muted); }}
  .cannot ul {{ margin:4px 0 0; padding-left:18px; }}

  .message.assistant .message-bubble.upload-note {{
    background:var(--warnbg); color:var(--warn);
    border:1px solid var(--warn);
  }}
  .message.assistant .message-bubble.error-bubble {{
    background:var(--warnbg); color:var(--warn);
    border:1px solid var(--warn);
  }}
  .error-id {{
    margin-top:6px; font-size:11px; opacity:0.8;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  }}
  .sent-attachments {{ margin-top:6px; font-size:12px; opacity:0.85; }}
  .attached-file {{ list-style:none; font-size:12px; color:var(--accent); }}
  .attach-btn.has-files {{
    border-color:var(--accent); color:var(--accent); font-weight:700;
  }}

  #file-input {{ display:none; }}

  footer {{
    background:var(--card); border-top:1px solid var(--line);
    padding:12px 16px; font-size:12px; color:var(--muted);
    text-align:center; display:flex; justify-content:space-between;
    align-items:center;
  }}
  .audience-display {{
    font-size:11px; color:var(--muted);
  }}

  .thinking {{
    color:var(--muted); font-size:13px; font-style:italic;
    text-align:center; padding:8px;
  }}
  .thinking::after {{
    content:" ."; animation:dots 1.5s steps(4, end) infinite;
  }}
  @keyframes dots {{
    0%, 20% {{ content:" ."; }}
    40% {{ content:" .."; }}
    60% {{ content:" ..."; }}
    80%, 100% {{ content:" ."; }}
  }}
</style>

<div class="chat-container">
  <header>
    <h1>Lime Green technical assistant</h1>
    <div class="header-controls">
      <div class="audience-badge" id="audience-display">{audience_display}</div>
      <button class="new-chat-btn" onclick="newChat()">New chat</button>
    </div>
  </header>

  <div class="chat-messages" id="chat-messages">
    {initial_content}
  </div>

  <div class="input-area">
    <div style="width:100%;">
      <div class="input-extras">
        <div class="input-info">
          <span>🔒 Lime Green sources only</span>
          <ul class="upload-notes" id="upload-notes"></ul>
        </div>
      </div>
      <div class="input-wrapper">
        <input type="text" class="input-field" id="question-input"
               placeholder="Ask a question…" autocomplete="off">
        <button class="attach-btn" title="Upload image" onclick="triggerFileInput()">+</button>
        <button class="send-btn" id="send-btn" onclick="sendMessage()">→</button>
        <input type="file" id="file-input" name="image" accept="image/*" multiple>
      </div>
    </div>
  </div>
</div>

<footer>
  <div>Responses are grounded in Lime Green published material</div>
  <div class="audience-display" id="index-meta">{meta}</div>
  <div class="audience-display" id="footer-audience">{footer_audience}</div>
</footer>

<script>
let conversationActive = false;

// The operator's routing view, off for a customer. Read from this page's own
// query string, which is where the server reads it from too, so the two
// renderers cannot disagree about whether this is a diagnostic session.
//
// `let` rather than `const` so it can be switched on from a console, and by a
// browser test that has no URL to carry a query string.
let DIAGNOSTICS_VISIBLE =
  new URLSearchParams(window.location.search).get('v') === '1';

function newChat() {{
  // The server ends the session, not this. The cookie is HttpOnly, so a
  // `document.cookie` write here is silently discarded by the browser.
  location.href = '/new';
}}

function triggerFileInput() {{
  document.getElementById('file-input').click();
}}

// What is attached, shown before it is sent.
function showAttached() {{
  const files = Array.from(document.getElementById('file-input').files);
  const list = document.getElementById('upload-notes');
  list.innerHTML = '';
  document.querySelector('.attach-btn').classList.toggle('has-files', files.length > 0);
  files.forEach(file => {{
    const li = document.createElement('li');
    li.className = 'attached-file';
    li.textContent = '📎 ' + file.name;
    list.appendChild(li);
  }});
}}

function clearAttached() {{
  document.getElementById('upload-notes').innerHTML = '';
  document.querySelector('.attach-btn').classList.remove('has-files');
}}

function showThinking(hasImages) {{
  const container = document.getElementById('chat-messages');
  if (container.querySelector('.landing')) {{ container.innerHTML = ''; }}
  const el = document.createElement('div');
  el.className = 'thinking';
  el.id = 'thinking';
  el.textContent = hasImages
    ? 'Reading the photograph and searching Lime Green sources'
    : 'Searching Lime Green sources';
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
}}

function hideThinking() {{
  const el = document.getElementById('thinking');
  if (el) el.remove();
}}

function sendMessage() {{
  if (document.getElementById('send-btn').disabled) return;
  const input = document.getElementById('question-input');
  const question = input.value.trim();
  if (!question) return;

  // Copied, not referenced. `input.files` is a live FileList, and clearing the
  // input below empties it -- so the append loop ran over nothing and every
  // photograph was posted as a body with no image parts in it. The server then
  // read zero images and had no attachment to complain about either, which is
  // why this looked like "the assistant ignored my photo" rather than an error.
  const files = Array.from(document.getElementById('file-input').files);
  const hasImages = files.length > 0;

  // Add user message to chat, naming the attachment. A photograph the person
  // can no longer see -- the file input is cleared on the next line -- is a
  // photograph they cannot tell was sent, and "did it get my picture?" is the
  // first thing anybody asks of an upload.
  addMessage('user', question, files.map(f => f.name));
  input.value = '';
  input.focus();
  document.getElementById('file-input').value = '';
  clearAttached();

  // Send to server
  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  showThinking(hasImages);

  if (hasImages) {{
    // Use FormData for image upload
    const formData = new FormData();
    formData.append('q', question);
    for (let file of files) {{
      formData.append('image', file);
    }}

    fetch('/ask', {{
      method: 'POST',
      body: formData
    }})
    .then(r => r.json())
    .then(data => handleResponse(data))
    .catch(err => showError(err.message))
    .finally(() => {{ hideThinking(); btn.disabled = false; }});
  }} else {{
    // Use GET for text-only
    const q = encodeURIComponent(question);
    fetch('/ask?q=' + q, {{
      headers: {{ 'Accept': 'application/json' }}
    }})
    .then(r => r.json())
    .then(data => handleResponse(data))
    .catch(err => showError(err.message))
    .finally(() => {{ hideThinking(); btn.disabled = false; }});
  }}
}}

function addMessage(role, text, attachments) {{
  const container = document.getElementById('chat-messages');
  if (container.querySelector('.landing')) {{
    container.innerHTML = '';
    conversationActive = true;
  }}
  const msgEl = document.createElement('div');
  msgEl.className = 'message ' + role;
  let inner = escapeHtml(text);
  if (attachments && attachments.length > 0) {{
    inner += '<div class="sent-attachments">📎 ' +
             attachments.map(escapeHtml).join(', ') + '</div>';
  }}
  msgEl.innerHTML = '<div class="message-bubble">' + inner + '</div>';
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
}}

function responseContainer(correlationId) {{
  const container = document.createElement('div');
  container.className = 'answer-response';
  container.dataset.state = 'rendering';
  // Only the server's short random reference, never session or diagnostic data.
  if (typeof correlationId === 'string' && /^[a-f0-9]{{12}}$/.test(correlationId)) {{
    container.dataset.correlationId = correlationId;
  }}
  return container;
}}

function showError(message, correlationId, container) {{
  if (!container) {{
    container = responseContainer(correlationId);
    document.getElementById('chat-messages').appendChild(container);
  }}
  correlationId = container.dataset.correlationId;
  const chat = document.getElementById('chat-messages');
  const landing = chat.querySelector('.landing');
  if (landing) landing.remove();
  const el = document.createElement('div');
  el.className = 'message assistant';
  let inner = '<div class="message-bubble error-bubble">' + escapeHtml(message);
  if (correlationId) {{
    inner += '<div class="error-id">reference ' + escapeHtml(correlationId) + '</div>';
  }}
  el.innerHTML = inner + '</div>';
  container.appendChild(el);
  container.dataset.state = 'error';
  chat.scrollTop = chat.scrollHeight;
}}

// One row per reading, each carrying its own certainty word and, where the
// system declined to act on it, the reason. The value and the hedge are
// rendered together and never separately: a row showing "brick" without
// "likely, but not certain" beside it would turn a reading into a fact, which
// is the whole failure the perception stage is arranged to avoid.
const CERTAINTY_STYLE = {{
  'OBSERVED':         ['#166534', '#f0fdf4', 'seen'],
  'LIKELY':           ['#92400e', '#fffdf5', 'likely'],
  'UNCERTAIN':        ['#9f1239', '#fff8fa', 'uncertain'],
  'CANNOT_DETERMINE': ['#334155', '#f8fafc', 'cannot tell']
}};

function renderPerception(perception, container) {{
  if (!perception) return;
  const el = document.createElement('div');
  el.className = 'message assistant perception-panel';

  // Switched off is a supported state, not an error, and it gets its own
  // sentence rather than an empty "From the photograph" heading -- which would
  // read as "looked and saw nothing" when nothing looked.
  if (perception.enabled === false) {{
    el.innerHTML = '<div class="message-bubble"><strong>Photograph not read' +
                   '</strong><div class="perception-none">' +
                   escapeHtml((perception.summary || []).join(' ')) +
                   '</div></div>';
    container.appendChild(el);
    return;
  }}

  let html = '<div class="message-bubble"><strong>From the photograph</strong>';

  const rows = (perception.observations || []);
  if (rows.length === 0) {{
    html += '<div class="perception-none">Nothing could be read from the ' +
            'image with enough confidence to report.</div>';
  }} else {{
    html += '<ul class="perception">';
    rows.forEach(o => {{
      const style = CERTAINTY_STYLE[o.certainty] || CERTAINTY_STYLE['UNCERTAIN'];
      const shown = (o.certainty === 'CANNOT_DETERMINE' ||
                     o.certainty === 'UNCERTAIN')
        ? escapeHtml(o.attribute.replace(/_/g, ' '))
        : escapeHtml(o.attribute.replace(/_/g, ' ')) + ': <b>' +
          escapeHtml(String(o.value).replace(/_/g, ' ')) + '</b>';
      html += '<li><span class="pill" style="color:' + style[0] +
              ';background:' + style[1] + '">' + style[2] + '</span> ' + shown;
      if (o.withheld) {{
        html += '<span class="withheld"> &mdash; ' +
                escapeHtml(o.withheld) + '</span>';
      }}
      html += '</li>';
    }});
    html += '</ul>';
  }}

  const cannot = perception.cannot_determine_from_image || [];
  if (cannot.length > 0) {{
    html += '<div class="cannot"><strong>Not determinable from the photograph' +
            '</strong><ul>' +
            cannot.slice(0, 6).map(c => '<li>' + escapeHtml(c) + '</li>').join('') +
            '</ul></div>';
  }}
  if (perception.truncated) {{
    html += '<div class="withheld">The model\\u2019s answer was cut short, so ' +
            'what it did say is shown but was not acted on.</div>';
  }}
  html += '</div>';
  el.innerHTML = html;
  container.appendChild(el);
}}

function handleResponse(data) {{
  const chat = document.getElementById('chat-messages');
  const container = responseContainer(data.correlation_id);
  const landing = chat.querySelector('.landing');
  if (landing) landing.remove();
  chat.appendChild(container);
  renderPerception(data.perception, container);
  renderUploadNotes(data.upload_notes, container);
  if (data.error) {{
    showError(data.error, data.correlation_id, container);
    return;
  }}
  if (!data.parts || data.parts.length === 0) {{
    showError('No answer came back for that question.', data.correlation_id, container);
    return;
  }}

  data.parts.forEach((answer, index) => renderAnswer(answer, null, container, index));
  container.dataset.state = 'complete';
  chat.scrollTop = chat.scrollHeight;
}}

function renderUploadNotes(uploadNotes, container) {{
  if (uploadNotes && uploadNotes.length > 0) {{
    uploadNotes.forEach(note => {{
      const noteEl = document.createElement('div');
      noteEl.className = 'message assistant';
      noteEl.innerHTML = '<div class="message-bubble upload-note">' + escapeHtml(note) + '</div>';
      container.appendChild(noteEl);
    }});
  }}
}}

function renderAnswer(answer, uploadNotes, container, index = 0) {{
  const text = answer.text || answer.body || 'No response';
  container = container || document.getElementById('chat-messages');
  renderUploadNotes(uploadNotes, container);
  const msgEl = document.createElement('div');
  msgEl.className = 'message assistant';
  msgEl.dataset.partIndex = String(index);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  // Render text with citations
  const textHtml = renderTextWithCitations(text);
  const p = document.createElement('p');
  p.className = 'answer-text';
  p.innerHTML = textHtml;
  bubble.appendChild(p);

  // Add sources disclosure
  if (answer.sources && answer.sources.length > 0) {{
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources-disclosure';
    const btn = document.createElement('button');
    btn.className = 'disclosure-btn';
    btn.textContent = 'Sources';
    btn.onclick = (e) => {{ e.preventDefault(); toggleSources(btn, sourcesList); }};
    sourcesDiv.appendChild(btn);

    const sourcesList = document.createElement('div');
    sourcesList.className = 'source-list';
    answer.sources.forEach(src => {{
      const item = document.createElement('div');
      item.className = 'source-item';
      item.innerHTML = '<div class="source-name">' + escapeHtml(src.name) +
        (src.section ? ' -- ' + escapeHtml(src.section) : '') +
        (src.date ? ' (' + escapeHtml(src.date) + ')' : '') + '</div>' +
        '<a href="' + escapeHtml(src.url) + '" class="source-link" target="_blank" rel="noreferrer">' +
        escapeHtml(src.url) + '</a>';
      sourcesList.appendChild(item);
    }});
    sourcesDiv.appendChild(sourcesList);
    bubble.appendChild(sourcesDiv);
  }}

  // Add diagnostics disclosure. Gated on the same `?v=1` the server reads,
  // taken from the page's own URL so the client and the server agree without
  // a second switch to keep in step: a customer sees the answer and its
  // sources, an operator who asked for the routing view sees it on every turn
  // of the conversation rather than only the first. The payload still carries
  // `diagnostics`; this decides whether the page draws them.
  if (DIAGNOSTICS_VISIBLE &&
      (answer.path || answer.diagnostics || answer.failed_checks ||
       container.dataset.correlationId)) {{
    const diagDiv = document.createElement('div');
    diagDiv.className = 'diagnostics-disclosure';
    const btn = document.createElement('button');
    btn.className = 'disclosure-btn';
    btn.textContent = 'Why this answer?';
    const diagList = renderDiagnostics(answer.diagnostics || {{}}, answer.failed_checks);
    if (answer.path) {{
      const tag = document.createElement('span');
      tag.className = 'tag' + (answer.refused ? ' refused' : '');
      tag.textContent = answer.path;
      diagList.prepend(tag);
    }}
    if (container.dataset.correlationId) {{
      const reference = document.createElement('div');
      reference.className = 'diag-line';
      reference.textContent = 'reference ' + container.dataset.correlationId;
      diagList.appendChild(reference);
    }}
    btn.onclick = (e) => {{ e.preventDefault(); toggleDiagnostics(btn, diagList); }};
    diagDiv.appendChild(btn);
    diagDiv.appendChild(diagList);
    bubble.appendChild(diagDiv);
  }}

  msgEl.appendChild(bubble);
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
}}

function renderRefusal(answer, uploadNotes) {{
  renderAnswer(answer, uploadNotes);
}}

function renderDiagnostics(diag, failedChecks) {{
  const div = document.createElement('div');
  div.className = 'diagnostics-list';

  function addLine(key, value) {{
    const line = document.createElement('div');
    line.className = 'diag-line';
    line.innerHTML = '<span class="diag-key">' + escapeHtml(key) + '</span> ' +
                     escapeHtml(String(value));
    div.appendChild(line);
  }}

  if (diag.path) addLine('route', diag.path);
  if (diag.step) addLine('step', diag.step);
  if (diag.reason) addLine('why', diag.reason);
  if (diag.refusal_reason) addLine('why', diag.refusal_reason);
  if (typeof diag.top_score === 'number') addLine('top score', diag.top_score.toFixed(3));
  if (diag.evidence_count !== undefined) addLine('evidence', diag.evidence_count);
  if (diag.generation_seconds !== undefined) addLine('generated', diag.generation_seconds + 's');
  if (failedChecks && failedChecks.length > 0) {{
    addLine('checks failed', failedChecks.join(', '));
  }}

  return div;
}}

function renderTextWithCitations(text) {{
  return escapeHtml(text).replace(/\\[(\\d+)\\]/g, '<span class="citation">[$1]</span>');
}}

function toggleSources(btn, list) {{
  btn.classList.toggle('open');
  list.classList.toggle('open');
}}

function toggleDiagnostics(btn, list) {{
  btn.classList.toggle('open');
  list.classList.toggle('open');
}}

function escapeHtml(text) {{
  const map = {{
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }};
  return text.replace(/[&<>"']/g, m => map[m]);
}}

document.getElementById('file-input').addEventListener('change', showAttached);

// Enter key to send
document.getElementById('question-input').addEventListener('keydown', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
}});
</script>
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_landing() -> str:
    """The landing page shown on first load with no messages yet."""
    return (
        '<div class="landing">'
        '<h2>How can I help?</h2>'
        '<p>Ask about Lime Green products, installation, compatibility, or anything '
        'else published in their technical material.</p>'
        '</div>'
    )


def render_upload_notes(notes) -> str:
    """What happened to the caller's attachment, as its own block.

    Escaped like everything else on this page. A note names a filename the
    caller chose, and a filename is caller-controlled text arriving on a
    surface that renders HTML -- the one place an upload can reach the page
    without going anywhere near a model.
    """
    if not notes:
        return ""
    items = "".join(f"<li>{_esc(note)}</li>" for note in notes)
    return ("<div class='message assistant'><div class='message-bubble'>"
            f"<ul class='upload-notes'>{items}</ul></div></div>")


def _public_reference(correlation_id: str) -> str:
    """Accept only the bounded, random identifier minted by observability."""
    return (correlation_id if isinstance(correlation_id, str)
            and re.fullmatch(r"[a-f0-9]{12}", correlation_id) else "")


def render_perception(perception) -> str:
    """Keep visual uncertainty visible, outside the answer and its diagnostics."""
    if not perception:
        return ""
    blocks = ['<div class="message assistant perception-panel"><div class="message-bubble">']
    if perception.get("enabled") is False:
        blocks.append('<strong>Photograph not read</strong><div class="perception-none">')
        blocks.append(_esc(" ".join(perception.get("summary") or [])))
        blocks.append('</div>')
    else:
        blocks.append('<strong>From the photograph</strong>')
        rows = perception.get("observations") or []
        if not rows:
            blocks.append('<div class="perception-none">Nothing could be read from the '
                          'image with enough confidence to report.</div>')
        else:
            certainty_labels = {
                "OBSERVED": "seen", "LIKELY": "likely", "UNCERTAIN": "uncertain",
                "CANNOT_DETERMINE": "cannot tell",
            }
            blocks.append('<ul class="perception">')
            for row in rows:
                certainty = row.get("certainty", "UNCERTAIN")
                label = certainty_labels.get(certainty, "uncertain")
                shown = _esc(str(row.get("attribute", "")).replace("_", " "))
                if certainty in ("OBSERVED", "LIKELY"):
                    shown += ": <b>" + _esc(str(row.get("value", "")).replace("_", " ")) + "</b>"
                blocks.append(f'<li><span class="pill">{label}</span> {shown}')
                if row.get("withheld"):
                    blocks.append('<span class="withheld"> &mdash; '
                                  + _esc(row["withheld"]) + '</span>')
                blocks.append('</li>')
            blocks.append('</ul>')
        cannot = perception.get("cannot_determine_from_image") or []
        if cannot:
            blocks.append('<div class="cannot"><strong>Not determinable from the photograph'
                          '</strong><ul>')
            blocks.extend(f'<li>{_esc(item)}</li>' for item in cannot[:6])
            blocks.append('</ul></div>')
        if perception.get("truncated"):
            blocks.append('<div class="withheld">The model’s answer was cut short, so '
                          'what it did say is shown but was not acted on.</div>')
    blocks.append('</div></div>')
    return "".join(blocks)


def render_html(reply, verbose: bool, correlation_id: str = "") -> str:
    """Render every answer part, with internal labels confined to diagnostics."""
    if not reply.parts:
        return ""
    reference = _public_reference(correlation_id)
    blocks = [f'<div class="answer-response" data-state="complete"'
              f' data-correlation-id="{reference}">']
    blocks.append(render_perception(reply.parts[0][1].diagnostics.get("perception")))
    for index, (_part, answer) in enumerate(reply.parts):
        blocks.append(_render_answer_html(answer, index, reference, verbose))
    blocks.append('</div>')
    return "".join(blocks)


def _render_answer_html(answer, index: int, reference: str,
                        verbose: bool = False) -> str:

    blocks = []

    # Message container
    blocks.append(f'<div class="message assistant" data-part-index="{index}">')
    blocks.append('<div class="message-bubble">')

    # Answer text with citations rendered as spans
    text = answer.text or answer.body
    # Escape HTML, then restore citation markers with spans
    text_html = _esc(text)
    # Replace [n] citations with citation spans (simple loop since we expect few)
    text_html = re.sub(r"\[(\d+)\]", r'<span class="citation">[\1]</span>', text_html)
    blocks.append(f'<p class="answer-text">{text_html}</p>')

    # Sources disclosure section
    if answer.sources:
        blocks.append('<div class="sources-disclosure">')
        blocks.append('<button class="disclosure-btn" onclick="this.nextElementSibling.classList.toggle(\'open\'); this.classList.toggle(\'open\');">Sources</button>')
        blocks.append('<div class="source-list">')
        for src in answer.sources:
            blocks.append('<div class="source-item">')
            blocks.append(f'<div class="source-name">{_esc(src["name"])}')
            if src.get("section"):
                blocks.append(f' -- {_esc(src["section"])}')
            if src.get("date"):
                blocks.append(f' <span class="source-date">({_esc(src["date"])})</span>')
            blocks.append('</div>')
            blocks.append(f'<a href="{_esc(src["url"])}" class="source-link" target="_blank" rel="noreferrer">{_esc(src["url"])}</a>')
            blocks.append('</div>')
        blocks.append('</div>')
        blocks.append('</div>')

    # Diagnostics disclosure section. Off unless this request asked for it
    # with `?v=1`, which is the same switch the CLI's `-v` throws and the same
    # one the server already parses. A customer is shown the answer, its
    # sources and the provenance sentences; the route, the step, the score and
    # the reference are an operator's view of the same turn and are not
    # explanation a person asked for. Nothing is recomputed or discarded --
    # `answer.diagnostics` is untouched, the JSON branch still returns it, and
    # the structured log still records it.
    if verbose and (answer.path or answer.diagnostics
                    or answer.failed_checks or reference):
        blocks.append('<div class="diagnostics-disclosure">')
        blocks.append('<button class="disclosure-btn" onclick="this.nextElementSibling.classList.toggle(\'open\'); this.classList.toggle(\'open\');">Why this answer?</button>')
        blocks.append('<div class="diagnostics-list">')
        if answer.path:
            tag_class = 'tag refused' if answer.refused else 'tag'
            blocks.append(f'<span class="{tag_class}">{_esc(answer.path)}</span>')
        if reference:
            blocks.append(f'<div class="diag-line"><span class="diag-key">reference</span> {reference}</div>')

        d = answer.diagnostics
        if d.get('path'):
            blocks.append(f'<div class="diag-line"><span class="diag-key">route</span> {_esc(d["path"])}</div>')
        if d.get('step'):
            blocks.append(f'<div class="diag-line"><span class="diag-key">step</span> {_esc(str(d["step"]))}</div>')
        if d.get('reason'):
            blocks.append(f'<div class="diag-line"><span class="diag-key">why</span> {_esc(d["reason"])}</div>')
        elif d.get('refusal_reason'):
            blocks.append(f'<div class="diag-line"><span class="diag-key">why</span> {_esc(d["refusal_reason"])}</div>')
        if d.get('top_score') is not None:
            blocks.append(f'<div class="diag-line"><span class="diag-key">top score</span> {d["top_score"]:.3f}</div>')
        if d.get('evidence_count') is not None:
            blocks.append(f'<div class="diag-line"><span class="diag-key">evidence</span> {d["evidence_count"]}</div>')
        if d.get('generation_seconds'):
            blocks.append(f'<div class="diag-line"><span class="diag-key">generated</span> {d["generation_seconds"]}s</div>')
        if answer.failed_checks:
            checks_str = ", ".join(answer.failed_checks)
            blocks.append(f'<div class="diag-line"><span class="diag-key">checks</span> {_esc(checks_str)}</div>')

        blocks.append('</div>')
        blocks.append('</div>')

    blocks.append('</div>')
    blocks.append('</div>')

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
    # store on purpose -- see UploadBudget.
    uploads: UploadBudget = UploadBudget()
    # The store this server was started against, so `/ready` inspects the one
    # actually serving rather than whatever the module default happens to be.
    db_path: str = "data/index/knowledge.db"
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
        # Liveness. Deliberately static and deliberately cheap: it answers
        # "is this process running", nothing more, and a probe that hit the
        # store would fail the process for a fault the process does not have.
        if url.path == "/health":
            self._send_plain(b"OK\n", "text/plain")
            return
        # Readiness, which is a different question and used to have no answer
        # here. `/health` returns OK from a process that has no index, an index
        # built by another embedding model, or no Ollama to reach -- so a
        # container could report healthy while unable to answer anything. This
        # runs the same checks `python -m assistant.health` runs and fails the
        # probe, with 503, when any of them is false.
        if url.path == "/ready":
            self._send_ready(parse_qs(url.query).get("format", [""])[0])
            return
        # One line, delegating immediately. The renderer lives in
        # assistant/metrics.py rather than here because a later slice rewrites
        # this page wholesale and a Prometheus exposition format entangled with
        # the HTML would be rewritten with it. Nothing about the endpoint --
        # its window, its labels, its privacy posture -- is decided in this file.
        if url.path == "/metrics":
            self._send_plain(metrics.render(self.assistant.repo).encode("utf-8"),
                             metrics.CONTENT_TYPE)
            return
        # Ending a conversation is a server action, because the cookie naming it
        # is HttpOnly and a page cannot clear what it is not allowed to read.
        # `newChat()` used to try, with `document.cookie`; the browser ignores
        # that write, so the old cookie went straight back up with the next
        # request and the "new" conversation inherited the substrate, location
        # and product of the one the person believed they had ended.
        if url.path == "/new":
            self.session_id = self.sessions.open("")
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.send_header("X-Correlation-Id", self.correlation_id)
            self.send_header("Set-Cookie",
                             f"{SESSION_COOKIE}={self.session_id}; Path=/; "
                             f"HttpOnly; SameSite=Lax")
            self.end_headers()
            return
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
            # then sees is a reset rather than the 404 -- the status code is
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
        large" would be doing the work the cap exists to avoid -- that case
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
        caller should be told about their own upload -- a file that is not an
        image, a count over the cap -- because silently dropping an attachment
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

    def _build_context(self, session_id: str, limit: int = 2) -> str:
        """The earlier **questions** of this conversation, for the model to see.

        Returns a string like:
        "Earlier in this conversation you asked:
         - What thickness should Lime Green Ultra be applied at?
         - I have a solid brick wall internally. Can I use Ultra?"

        **The answers are deliberately not here, and that is the whole point of
        this function.** It used to emit "You answered: ..." beside each
        question, which put previously generated prose -- figures included --
        into the next prompt. The model then did the obvious thing with it.
        Asked a third question in a live conversation, it wrote "It should be
        applied in a uniform thickness between 10 and 30mm" -- its own answer
        from turn one -- and cited a passage that does not contain that figure.
        Check 2 caught it: "'10' is not in the passage it is cited to". Turn
        four did the same with "It is suitable for masonry backgrounds such as
        brick walls". Both refused, and both were questions the corpus answers.

        `HISTORY_BLOCK` already tells the model this is not evidence and never
        to take a fact from it. It ignored that, which is the ordinary result of
        asking a prompt to be a boundary. Removing the material is a boundary;
        asking nicely is not.

        Nothing is lost that this is for. The job is resolving a reference --
        which Ultra "it" means -- and the earlier *question* carries the subject
        perfectly well. The facts travel separately and always did: the trusted
        slots in the checkpoint carry product, substrate and location, and they
        are what the router and retrieval read. The command line sets no history
        at all and answers the same four turns correctly, which is the clearest
        evidence that prior answers were never load-bearing.

        The cost, stated: a reference that depends on an answer rather than a
        question -- "the one you mentioned" -- is now ambiguous. An ambiguous
        reference becomes an ask-back, and an ask-back is a much better failure
        than a recycled figure under a citation that does not support it.

        Args:
            session_id: the session to get history from
            limit: how many prior turns to include (default 2, keep recent)

        Returns:
            context string (empty if there are fewer than `limit` prior turns)
        """
        turns = self.sessions.turns(session_id)
        if len(turns) < limit:
            # Not enough history to build context
            return ""

        # Questions only. `_answer` is bound and discarded rather than skipped
        # with an underscore-free name, so the shape of what is stored stays
        # visible to whoever reads this next and wonders where the answers went.
        prior = turns[-limit:]
        return "\n".join(["Earlier in this conversation you asked:"]
                         + [f"- {question}" for question, _answer in prior])

    def _respond(self, path: str, question: str, verbose: bool, audience: str,
                 images: list, notes: list, session_open: bool = False) -> None:
        audiences = resolve(audience, self.audiences)

        if not session_open:
            self.session_id = self._session()
        asked = question

        # The ask-back guesswork that used to live here is gone, and its
        # absence is the point of wiring the graph in.
        #
        # It worked like this: the previous turn's question was stored in a
        # `pending` string, and this turn was inspected to guess whether it was
        # an answer to it. The guess got "Can I use Ultra on the same wall?"
        # wrong -- nine words, and the detector found a slot in it -- so a new
        # question was discarded and the earlier one re-answered in its place.
        # Everything about the half-finished turn that was not in the `pending`
        # string was lost, because a string is all there was.
        #
        # `assistant/graph.py` pauses the turn instead of describing it. The
        # conversation is parked in the checkpoint under this session id, and
        # the next message resumes it from the node it stopped in, through
        # retrieval and the evidence gate, to the question originally asked.
        # There is nothing here to keep in step with it.

        if path == "/ask":
            reply = None
            try:
                reply = self._answer(question, audiences, images) if question else None
                # After a resumed ask-back the message typed was "brick" and the
                # question answered was the one from two turns ago. The page
                # reports the latter, which is what the person is reading.
                if reply is not None and reply.parts:
                    asked = reply.parts[0][1].diagnostics.get(
                        "resumed_question") or question
            except (ollama.OllamaUnavailable, IndexMismatch) as exc:
                # The HTML branch has always handled this; the JSON branch did
                # not, so an unreachable model answered a request with a
                # stack trace and no status code worth acting on.
                self._send(json.dumps({"error": str(exc), "question": question,
                                       "correlation_id": self.correlation_id},
                                      ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8", status=503)
                return
            except Exception as exc:
                # Catch any other exception and return JSON error
                self._send(json.dumps({"error": f"Internal error: {str(exc)}",
                                       "question": question,
                                       "correlation_id": self.correlation_id},
                                      ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8", status=500)
                return

            if reply is not None:
                self._remember(question, reply)

            payload = {
                "question": question,
                "answered": asked,
                "session_slots": self.sessions.carried(self.session_id),
                "images_read": len(images),
                "upload_notes": notes,
                # What the photographs actually showed, each reading with its
                # own certainty. Lifted to the top level rather than left in
                # per-part diagnostics because it is a fact about the *request*
                # -- one upload, however many topics the message carried -- and
                # because a page that had to dig it out of diagnostics would be
                # reading an audit field as an interface.
                "perception": (reply.parts[0][1].diagnostics.get("perception")
                               if reply and reply.parts else None),
                "correlation_id": self.correlation_id,
                "parts": [
                    {"question": q, "path": a.path, "refused": a.refused,
                     "text": a.text, "body": a.body, "sources": a.sources,
                     "caveats": a.caveats, "diagnostics": a.diagnostics,
                     "failed_checks": a.failed_checks,
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

        # HTML response for GET / (chat interface)
        audience_display = audiences[0] if audiences else "public"
        if question:
            try:
                reply = self._answer(question, audiences, images)
                self._remember(question, reply)
                initial_content = render_html(reply, verbose, self.correlation_id)
            except (ollama.OllamaUnavailable, IndexMismatch) as exc:
                initial_content = (f"<div class='message assistant'>"
                                   f"<div class='message-bubble'>Error: {_esc(str(exc))}"
                                   f"</div></div>")
        else:
            initial_content = render_landing()

        # What happened to the attachment, on the surface a person is looking
        # at. The JSON branch has carried these all along and this one dropped
        # them when the page became a chat client, so a POST to `/` with a file
        # that was not an image answered the question and said nothing about
        # the file -- which reads as "the assistant ignored my photo" rather
        # than as the refusal it actually was.
        #
        # Above the answer rather than inside it: what happened to an upload is
        # a fact about the request, not about the published material, and
        # putting it in the answer would put an unsourced sentence on a page
        # where every other sentence carries a citation.
        initial_content = render_upload_notes(notes) + initial_content

        page = PAGE.format(
            initial_content=initial_content,
            audience_display=audience_display,
            meta=_esc(self.meta),
            footer_audience=f"audience: {audience_display}")
        self._send(page.encode("utf-8"), "text/html; charset=utf-8")

    def _answer(self, question: str, audiences: tuple, images: list):
        """One turn, through the state machine. The single answering path.

        The session cookie is the graph's `thread_id`, which is what makes the
        two layers agree by construction rather than by being kept in step: the
        conversation the page thinks it is showing and the conversation the
        graph is continuing are the same object, addressed by the same id.

        The transcript is passed as `history` and reaches generation only --
        never the policy gate, the slot detector or the embedder. That
        separation is the whole of `tests/test_context_isolation.py`, and it is
        why `_build_context` is read here and nowhere earlier.
        """
        from .conversation import TurnInput

        turn = TurnInput(
            raw_question=question,
            turn_index=len(self.sessions.turns(self.session_id)) + 1,
            images=tuple(images or ()),
            audiences=tuple(audiences),
            session_id=self.session_id,
            correlation_id=self.correlation_id,
        )
        object.__setattr__(turn, "history", self._build_context(self.session_id))
        reply, _state = self.assistant.ask_turn(turn)
        return reply

    def _remember(self, question: str, reply) -> None:
        """Fold what this turn established back into the session.

        The slots taken are the ones the router actually detected, read off the
        answer rather than re-derived, so the page and the answer cannot
        disagree about what was assumed. `pending` is set only by an ask-back
        and cleared by anything else, so a question that was answered never
        resumes later.

        When `auto_answered` is True this turn answered an ask-back rather than
        asking a new question, and the slots to remember are the carried ones
        **plus what this turn just supplied**. `stated` is that second half, and
        it has to be passed in rather than re-read.

        Re-reading is what the bug was. The branch used to be
        `slots = self.sessions.carried(self.session_id)`, and the value the
        caller had just typed was not in there: it was detected from their
        reply, merged into a *local* dict to answer the pending question, and
        then dropped. So the assistant asked "what is the substrate?", was told
        "brick", answered the original question correctly using brick -- and
        forgot it. The next turn asked for the substrate again. The feature
        looked like it worked, once, and defeated itself on turn three.

        The provenance is `STATED` and not `CARRIED`: the person said it in
        this turn. That distinction reaches the printed sentence, so getting it
        wrong would tell them they had mentioned it "earlier in this
        conversation" about a word they had just typed.

        **A slot read off a photograph is dropped here, and that is the rule the
        whole vision seam rests on.** `diagnostics["slots"]` is the router's
        merged view, so once an upload can fill a slot it carries values nobody
        said -- and folding those into the session would persist a model's
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

        if True:
            # Slots read off the answer, as they always were. The
            # `auto_answered` branch that used to sit here is gone with the
            # guesswork it served: the graph resumes a paused turn itself, so
            # there is no second place deciding what an ask-back established.
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

    # Keys `/ready` will publish over HTTP, and the omission is the point.
    # `store_target` is a filesystem path or a DSN host, and this endpoint is
    # unauthenticated: an orchestrator needs to know *whether* the store is
    # reachable, never where it is. `assistant/trace.py` declines to be an
    # endpoint at all for the same reason, and readiness only qualifies because
    # what remains here is operational state -- counts, tags, booleans.
    READY_FIELDS = ("ready", "checks", "snapshot", "documents", "chunks",
                    "embedding_model", "embedding_dimensions",
                    "chunking_version", "generation_model", "vision", "error")

    def _send_ready(self, form: str = "") -> None:
        """Readiness as JSON, or as the table `python -m assistant.health` prints.

        200 when every check passes and 503 when any does not, because the
        status code is the part an orchestrator reads. The body is for whoever
        then has to fix it.
        """
        report = health.check(self.db_path,
                              os.environ.get("ASSISTANT_POSTGRES_DSN", ""))
        if form == "text":
            body = (health.summary(report) + "\n").encode("utf-8")
            kind = "text/plain; charset=utf-8"
        else:
            public = {k: report[k] for k in self.READY_FIELDS if k in report}
            body = (json.dumps(public, indent=1) + "\n").encode("utf-8")
            kind = "application/json"
        self._send_plain(body, kind, status=200 if report["ready"] else 503)

    def _send_plain(self, body: bytes, content_type: str,
                    status: int = 200) -> None:
        """A response with no session cookie, for a scraper rather than a person.

        `_send` mints and returns a session on every response, which is right
        for a page someone is going to ask a second question on and wrong for
        `/metrics`: a scrape arriving every fifteen seconds would open a fresh
        `SessionStore` entry each time, so an unauthenticated endpoint would
        drive eviction of the conversations of people actually using the page.
        A scraper has no conversation. It gets the document and nothing else --
        no cookie, no correlation id, no state created by having asked.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        assistant = Assistant(repo, source="web")
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
    # process never inherits a conversation from the last one. Use a persisted
    # store so sessions survive server restarts and are shared across instances.
    Handler.sessions = open_persisted_session_store(
        db=args.db,
        dsn=os.environ.get("ASSISTANT_POSTGRES_DSN"),
    )
    Handler.uploads = UploadBudget()
    Handler.db_path = args.db
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
    # and opened in a browser -- so reporting the request rather than the bind
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
