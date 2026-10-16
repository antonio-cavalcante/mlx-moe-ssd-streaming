"""A small OpenAI-compatible HTTP server (/v1/chat/completions, /v1/models) on top of ChatSession.

The API is stateless (the client resends the whole conversation) but the model is not: one ChatSession keeps
the KV cache of the latest conversation. A request that is the same conversation plus one new user message
only feeds the new tokens; any other conversation is rebuilt from scratch (a full prefill).

All MLX work happens in the thread that calls `Engine.run_forever()`; HTTP handler threads only queue jobs
and stream the results back, so requests are served one at a time, in order.
"""
import json
import queue
import secrets
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chat_session import ContextFull

MAX_BODY = 16 << 20
EFFORTS = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}


class ApiError(Exception):
    def __init__(self, status, message, code=None, param=None, kind="invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.code, self.param, self.kind = status, message, code, param, kind

    def body(self):
        return {"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}}


# ---------------------------------------------------------------- request parsing

def _text(content, param):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text") and isinstance(part.get("text"), str):
                out.append(part["text"])
            else:
                raise ApiError(400, "only text content parts are supported", "unsupported_content", param)
        return "".join(out)
    raise ApiError(400, "message content must be a string or a list of text parts", None, param)


def normalize_messages(raw):
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, "'messages' must be a non-empty list", None, "messages")
    out = []
    for i, m in enumerate(raw):
        param = f"messages[{i}]"
        if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant"):
            raise ApiError(400, "supported roles are system, developer, user and assistant "
                                "(tool calling is not supported)", "unsupported_role", param)
        if m.get("tool_calls") or m.get("function_call"):
            raise ApiError(400, "tool calls are not supported", "unsupported_tools", param)
        out.append({"role": m["role"], "content": _text(m.get("content"), param)})
    if out[-1]["role"] != "user":
        raise ApiError(400, "the last message must have role 'user'", None, "messages")
    if any(m["role"] in ("system", "developer") for m in out[1:]):
        raise ApiError(400, "a system/developer message is only supported as the first message", None, "messages")
    return out


def parse_request(body, defaults):
    """Validate a chat.completions body. Returns a dict of what the engine needs."""
    if not isinstance(body, dict):
        raise ApiError(400, "the request body must be a JSON object")
    if body.get("tools") or body.get("functions") or body.get("tool_choice") not in (None, "none"):
        raise ApiError(400, "tools / function calling are not supported", "unsupported_tools", "tools")
    if body.get("n", 1) != 1:
        raise ApiError(400, "only n=1 is supported", None, "n")
    if body.get("logprobs"):
        raise ApiError(400, "logprobs are not supported", None, "logprobs")
    rf = body.get("response_format")
    if rf and (not isinstance(rf, dict) or rf.get("type", "text") != "text"):
        raise ApiError(400, "only response_format type 'text' is supported", None, "response_format")

    def number(name, lo, hi, default):
        v = body.get(name)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ApiError(400, f"'{name}' must be a number between {lo} and {hi}", None, name)
        return v

    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    if mt is None:
        mt = defaults["max_tokens"]
    elif isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
        raise ApiError(400, "'max_tokens' must be a positive integer", None, "max_tokens")
    effort = body.get("reasoning_effort")
    if effort is not None and EFFORTS.get(effort) is None:
        raise ApiError(400, "'reasoning_effort' must be one of low, medium, high, xhigh", None, "reasoning_effort")
    return dict(
        messages=normalize_messages(body.get("messages")),
        max_tokens=mt,
        effort=EFFORTS.get(effort),
        temp=number("temperature", 0, 2, defaults["temp"]),
        top_p=number("top_p", 0, 1, defaults["top_p"]),
        top_k=int(number("top_k", 0, 10**6, defaults["top_k"])),
        stream=bool(body.get("stream")),
        include_usage=bool((body.get("stream_options") or {}).get("include_usage")),
    )


def same_conversation(stored, sent):
    """The stored messages are the same conversation as `sent` (whitespace at the ends is ignored)."""
    return len(stored) == len(sent) and all(
        a["role"] == b["role"] and a["content"].strip() == b["content"].strip() for a, b in zip(stored, sent))


class TextCleaner:
    """Drops the whitespace the model puts before and after a message, without delaying visible text."""

    def __init__(self):
        self.started, self.pending = False, ""

    def feed(self, text):
        text = self.pending + text
        self.pending = ""
        if not self.started:
            text = text.lstrip()
            if not text:
                return ""
            self.started = True
        stripped = text.rstrip()
        self.pending = text[len(stripped):]
        return stripped


# ---------------------------------------------------------------- the engine (owns the model)

class Job:
    def __init__(self, req):
        self.req = req
        self.events = queue.Queue()           # ("chunks", [(channel, text)]) | ("done", Turn) | ("error", ApiError)
        self.cancelled = threading.Event()


class Engine:
    def __init__(self, session, make_sampler, log=lambda s: print(s, file=sys.stderr, flush=True)):
        self.session, self.make_sampler, self.log = session, make_sampler, log
        self.jobs = queue.Queue()
        self.stopping = False

    def submit(self, job):
        self.jobs.put(job)

    def stop(self):
        self.stopping = True
        self.jobs.put(None)

    def run_forever(self):
        while True:
            job = self.jobs.get()
            if job is None or self.stopping:
                return
            if job.cancelled.is_set():
                continue
            try:
                self._run(job)
            except KeyboardInterrupt:
                raise
            except ApiError as e:
                job.events.put(("error", e))
            except ContextFull as e:
                job.events.put(("error", ApiError(400, f"context length exceeded: {e}", "context_length_exceeded",
                                                  "messages")))
            except Exception as e:  # keep serving; the cache may be in doubt, so rebuild it next time
                traceback.print_exc()
                self.session._dirty = True
                job.events.put(("error", ApiError(500, f"{type(e).__name__}: {e}", "server_error",
                                                  kind="server_error")))

    def _run(self, job):
        req, s = job.req, self.session
        history, user = req["messages"][:-1], req["messages"][-1]["content"]
        effort = req["effort"] or s.effort
        if effort != s.effort or not same_conversation(s.messages, history):
            s.restore(history, effort)
        s.sampler = self.make_sampler(req["temp"], req["top_p"], req["top_k"])
        t0 = time.perf_counter()
        turn = s.send(user, on_chunks=lambda c: job.events.put(("chunks", c)), max_tokens=req["max_tokens"],
                      cancel=job.cancelled.is_set)
        self.log(f"chat.completions: ctx {turn.context} (reused {turn.reused}, fed {turn.prompt_tokens}"
                 f"{', rebuilt' if turn.rebuilt else ''}), first token {turn.prompt_s:.1f}s, "
                 f"{turn.generation_tokens} tok @ {turn.generation_tps:.1f} tok/s, {turn.finish}, "
                 f"{time.perf_counter() - t0:.1f}s total")
        job.events.put(("done", turn))
        if turn.finish == "interrupted" and not job.cancelled.is_set():
            raise KeyboardInterrupt           # Ctrl-C at the terminal: stop the server, not just the reply


# ---------------------------------------------------------------- responses

def usage(turn):
    prompt = turn.reused + turn.prompt_tokens
    return {"prompt_tokens": prompt, "completion_tokens": turn.generation_tokens,
            "total_tokens": prompt + turn.generation_tokens,
            "prompt_tokens_details": {"cached_tokens": turn.reused}}


def finish_reason(turn):
    return "length" if turn.finish == "length" else "stop"


def completion(model, cid, created, turn, show_reasoning):
    msg = {"role": "assistant", "content": turn.answer.strip()}
    if show_reasoning:
        msg["reasoning_content"] = turn.reasoning.strip()
    return {"id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason(turn)}],
            "usage": usage(turn)}


def chunk(model, cid, created, delta, finish=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


# ---------------------------------------------------------------- HTTP

def make_handler(engine, model_name, defaults, show_reasoning=True, api_key=None, cors_origin=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mlx-moe-ssd-streaming"

        def log_message(self, fmt, *args):
            pass

        # -- helpers
        def _headers(self, status, ctype, length=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            if length is not None:
                self.send_header("Content-Length", str(length))
            else:
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.close_connection = True
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Vary", "Origin")
            self.end_headers()

        def _json(self, status, obj):
            data = json.dumps(obj).encode()
            self._headers(status, "application/json", len(data))
            self.wfile.write(data)

        def _authorized(self):
            return api_key is None or self.headers.get("Authorization") == f"Bearer {api_key}"

        def _fail(self, e):
            self.close_connection = True      # the request body may be unread
            self._json(e.status, e.body())

        # -- routes
        def do_OPTIONS(self):
            self.send_response(204)
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Vary", "Origin")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                return self._json(200, {"status": "ok"})
            if not self._authorized():
                return self._fail(ApiError(401, "invalid API key", "invalid_api_key", kind="authentication_error"))
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "created": 0, "owned_by": "local"}]})
            self._fail(ApiError(404, f"unknown path {self.path}", "not_found", kind="not_found_error"))

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if not self._authorized():
                    raise ApiError(401, "invalid API key", "invalid_api_key", kind="authentication_error")
                if path != "/v1/chat/completions":
                    raise ApiError(404, f"unknown path {self.path}", "not_found", kind="not_found_error")
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY:
                    raise ApiError(413 if n > MAX_BODY else 400, "missing or oversized request body")
                try:
                    body = json.loads(self.rfile.read(n))
                except ValueError:
                    raise ApiError(400, "the request body is not valid JSON")
                req = parse_request(body, defaults)
            except ApiError as e:
                return self._fail(e)
            job = Job(req)
            engine.submit(job)
            cid, created = "chatcmpl-" + secrets.token_hex(12), int(time.time())
            try:
                if req["stream"]:
                    self._stream(job, cid, created, req)
                else:
                    self._blocking(job, cid, created)
            except (BrokenPipeError, ConnectionResetError):
                job.cancelled.set()           # the client went away: stop generating

        def _blocking(self, job, cid, created):
            while True:
                kind, payload = job.events.get()
                if kind == "error":
                    return self._fail(payload)
                if kind == "done":
                    return self._json(200, completion(model_name, cid, created, payload, show_reasoning))

        def _sse(self, obj):
            self.wfile.write(b"data: " + (b"[DONE]" if obj is None else json.dumps(obj).encode()) + b"\n\n")
            self.wfile.flush()

        def _stream(self, job, cid, created, req):
            cleaners = {"answer": TextCleaner(), "reasoning": TextCleaner()}
            started = False

            def start():
                nonlocal started
                if not started:
                    self._headers(200, "text/event-stream")
                    self._sse(chunk(model_name, cid, created, {"role": "assistant", "content": ""}))
                    started = True

            while True:
                kind, payload = job.events.get()
                if kind == "error":
                    if not started:
                        return self._fail(payload)
                    self._sse({"error": payload.body()["error"]})
                    return self._sse(None)
                if kind == "chunks":
                    for channel, text in payload:
                        text = cleaners[channel].feed(text)
                        if not text or (channel == "reasoning" and not show_reasoning):
                            continue
                        start()
                        self._sse(chunk(model_name, cid, created,
                                        {"content": text} if channel == "answer" else {"reasoning_content": text}))
                else:  # done
                    start()
                    self._sse(chunk(model_name, cid, created, {}, finish_reason(payload)))
                    if req["include_usage"]:
                        self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                                   "model": model_name, "choices": [], "usage": usage(payload)})
                    return self._sse(None)

    return Handler


def serve(engine, host, port, model_name, defaults, **kw):
    """Start the HTTP server in a background thread; the caller then runs `engine.run_forever()`."""
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, model_name, defaults, **kw))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    return httpd
