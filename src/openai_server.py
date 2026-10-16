"""A small OpenAI-compatible HTTP server (/v1/chat/completions, /v1/models) on top of ChatSession.

The API is stateless (the client resends the whole conversation) but the model is not: one ChatSession keeps
the KV cache of the latest conversation. A request that is the same conversation plus one new user message
only feeds the new tokens; any other conversation is rebuilt from scratch (a full prefill).

All MLX work happens in the thread that calls `Engine.run_forever()`; HTTP handler threads only queue jobs
and stream the results back, so requests are served one at a time, in order. Several ChatSessions (one KV
cache each) can share the model, so a few clients that take turns each keep their own cache.

Meant for a trusted network (a home or office LAN), not the open Internet: use --api-key (or the key that is
generated when the server listens beyond this machine) and, if the network is not yours, TLS.
"""
import hmac
import ipaddress
import json
import os
import queue
import secrets
import select
import socket
import ssl
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chat_session import ContextFull

MAX_BODY = 16 << 20
BYTES_PER_TOKEN_MAX = 256       # no token is longer than this, so a longer prompt cannot fit the context
EFFORTS = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}


class ApiError(Exception):
    def __init__(self, status, message, code=None, param=None, kind="invalid_request_error", headers=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param, self.kind = status, message, code, param, kind
        self.headers = headers or {}

    def body(self):
        return {"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}}


# ---------------------------------------------------------------- request parsing

def _check_utf8(text, param):
    try:
        text.encode("utf-8")              # e.g. a lone "\ud800" escape in the JSON would fail later, in the engine
    except UnicodeEncodeError:
        raise ApiError(400, "message text must be valid Unicode", "invalid_text", param) from None
    return text


def _text(content, param):
    if content is None:
        return ""
    if isinstance(content, str):
        return _check_utf8(content, param)
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text") and isinstance(part.get("text"), str):
                out.append(part["text"])
            else:
                raise ApiError(400, "only text content parts are supported", "unsupported_content", param)
        return _check_utf8("".join(out), param)
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
    """Validate a chat.completions body. Returns a dict of what the engine needs.

    `defaults` has max_tokens, temp, top_p, top_k and optionally max_tokens_cap (the most a client may ask
    for) and max_chars (longer conversations cannot fit the context; refusing them early keeps the engine,
    which tokenizes on the one thread that also generates, from being tied up by a huge body)."""
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

    def flag(obj, name, param):
        v = obj.get(name)
        if v is not None and not isinstance(v, bool):
            raise ApiError(400, f"'{name}' must be true or false", None, param)
        return bool(v)

    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    if mt is None:
        mt = defaults["max_tokens"]
    elif isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
        raise ApiError(400, "'max_tokens' must be a positive integer", None, "max_tokens")
    mt = min(mt, defaults.get("max_tokens_cap", mt))
    effort = body.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or EFFORTS.get(effort) is None):
        raise ApiError(400, "'reasoning_effort' must be one of low, medium, high, xhigh", None, "reasoning_effort")
    opts = body.get("stream_options")
    if opts is not None and not isinstance(opts, dict):
        raise ApiError(400, "'stream_options' must be an object", None, "stream_options")
    messages = normalize_messages(body.get("messages"))
    if sum(len(m["content"]) for m in messages) > defaults.get("max_chars", float("inf")):
        raise ApiError(400, "the conversation is longer than the context window", "context_length_exceeded",
                       "messages")
    return dict(
        messages=messages,
        max_tokens=mt,
        effort=EFFORTS.get(effort),
        temp=number("temperature", 0, 2, defaults["temp"]),
        top_p=number("top_p", 0, 1, defaults["top_p"]),
        top_k=int(number("top_k", 0, 10**6, defaults["top_k"])),
        stream=flag(body, "stream", "stream"),
        include_usage=flag(opts or {}, "include_usage", "stream_options.include_usage"),
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
    def __init__(self, req, client="-"):
        self.req, self.client = req, client
        self.session = None                   # set by the engine
        self.events = queue.Queue()           # ("chunks", [(channel, text)]) | ("done", Turn) | ("error", ApiError)
        self.cancelled = threading.Event()


class Engine:
    """Runs the queued jobs one at a time on the thread that calls `run_forever()`.

    `sessions` are ChatSessions that share one model, one KV cache each. A request goes to the session that
    already holds its conversation; otherwise the least recently used session is rebuilt for it."""

    def __init__(self, sessions, make_sampler, log=lambda s: print(s, file=sys.stderr, flush=True), max_queue=8,
                 probe=None):
        self.sessions = list(sessions) if isinstance(sessions, (list, tuple)) else [sessions]
        self.session = self.sessions[0]
        self.default_effort = self.sessions[0].effort     # a request without reasoning_effort gets this
        self.make_sampler, self.log, self.max_queue = make_sampler, log, max_queue
        self.probe, self.extra = probe, {}      # probe() -> dict for /health, run on the engine thread after each job
        self.jobs = queue.Queue()
        self.stopping = False
        self.outstanding = 0                  # queued + running
        self.running = False
        self._lock = threading.Lock()
        self._used = {id(s): 0 for s in self.sessions}
        self._tick = 0

    def submit(self, job):
        with self._lock:
            if self.stopping:
                raise ApiError(503, "the server is shutting down", "shutting_down", kind="server_error")
            if self.outstanding >= self.max_queue:
                raise ApiError(429, f"the server is busy: {self.outstanding} requests are already queued; "
                                    "try again shortly", "server_busy", kind="rate_limit_error",
                               headers={"Retry-After": "10"})
            self.outstanding += 1
        self.jobs.put(job)

    def stop(self):
        with self._lock:
            self.stopping = True
        self.jobs.put(None)

    def status(self):
        return {"busy": self.running, "queued": max(0, self.outstanding - int(self.running)),
                "sessions": len(self.sessions), **self.extra}

    def run_forever(self):
        try:
            while True:
                job = self.jobs.get()
                if job is None or self.stopping:
                    if job is not None:
                        self._reject(job)
                    return
                if job.cancelled.is_set():
                    self._finished()
                    continue
                self.running = True
                try:
                    self._run(job)
                except KeyboardInterrupt:
                    raise
                except ApiError as e:
                    job.events.put(("error", e))
                except ContextFull as e:
                    job.events.put(("error", ApiError(400, f"context length exceeded: {e}",
                                                      "context_length_exceeded", "messages")))
                except Exception as e:  # keep serving; the cache may be in doubt, so rebuild it next time
                    traceback.print_exc()
                    if job.session is not None:
                        job.session._dirty = True
                    job.events.put(("error", ApiError(500, f"{type(e).__name__}: {e}", "server_error",
                                                      kind="server_error")))
                finally:
                    self.running = False
                    self._finished()
                    self._probe()
        finally:
            with self._lock:
                self.stopping = True
            while True:                       # nobody is going to run what is still queued
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    self._reject(job)

    def _probe(self):
        if self.probe:
            try:
                self.extra = self.probe()
            except Exception:  # noqa: BLE001  (a broken probe must not stop the server)
                traceback.print_exc()
                self.probe = None

    def _finished(self):
        with self._lock:
            self.outstanding -= 1

    def _reject(self, job):
        self._finished()
        job.events.put(("error", ApiError(503, "the server is shutting down", "shutting_down",
                                          kind="server_error")))

    def pick_session(self, history, effort):
        """(session, restore_needed): the session that already holds this conversation, else the one to rebuild."""
        for s in self.sessions:
            if s.effort == effort and same_conversation(s.messages, history):
                return s, False
        fresh = [s for s in self.sessions if not s.messages]
        return min(fresh or self.sessions, key=lambda s: self._used[id(s)]), True

    def _run(self, job):
        req = job.req
        history, user = req["messages"][:-1], req["messages"][-1]["content"]
        effort = req["effort"] or self.default_effort
        s, restore = self.pick_session(history, effort)
        job.session = s
        if restore:
            s.restore(history, effort)
        self._tick += 1
        self._used[id(s)] = self._tick
        s.sampler = self.make_sampler(req["temp"], req["top_p"], req["top_k"])
        t0 = time.perf_counter()
        turn = s.send(user, on_chunks=lambda c: job.events.put(("chunks", c)), max_tokens=req["max_tokens"],
                      cancel=job.cancelled.is_set)
        self.log(f"chat.completions for {job.client}: session {self.sessions.index(s) + 1}/{len(self.sessions)}, "
                 f"ctx {turn.context} (reused {turn.reused}, fed {turn.prompt_tokens}"
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


# ---------------------------------------------------------------- network helpers

def is_loopback(host):
    """True for addresses that only this machine can reach (localhost, 127.x, ::1)."""
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host_header_name(value):
    """The host part of a Host header ('example:8000', '[::1]:8000')."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[1:].split("]")[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def lan_addresses():
    """This machine's non-loopback IPv4 addresses (best effort; nothing is sent over the network)."""
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 9))       # UDP connect only picks the outgoing interface
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        found += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    out = []
    for a in found:
        ip = ipaddress.ip_address(a)
        if not (ip.is_loopback or ip.is_unspecified or ip.is_link_local) and a not in out:
            out.append(a)
    return out


def server_urls(host, port, scheme="http"):
    """URLs a client can use, given the address the server listens on."""
    if is_loopback(host):
        names = [host]
    elif _is_ip(host) and ipaddress.ip_address(host.strip("[]")).is_unspecified:
        names = lan_addresses()
        name = socket.gethostname()
        if name and "." in name:                       # macOS: <name>.local
            names.insert(0, name)
        names.append("127.0.0.1")
    else:
        names = [host]
    return [f"{scheme}://{('[' + n + ']') if ':' in n else n}:{port}/v1" for n in names]


def _is_ip(host):
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def resolve_api_key(host, api_key=None, key_file=None, env_key=None, no_auth=False):
    """(key or None, note for the startup banner).

    Order: --api-key, the key file (created with a fresh random key if it does not exist), the environment.
    A server that listens beyond this machine and was given none generates a key for this run, so it is never
    open by accident; `no_auth` opts out."""
    if no_auth:
        if api_key or key_file:
            raise ValueError("--no-auth cannot be combined with an API key")
        return None, "NO AUTHENTICATION: anyone who can reach this port can use the model"
    if api_key:
        return api_key, ""
    if key_file:
        path = Path(key_file).expanduser()
        try:
            key = path.read_text().strip()
            if key:
                return key, f"API key read from {path}"
        except FileNotFoundError:
            pass
        key = secrets.token_urlsafe(24)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)     # reached only if missing or empty
        os.fchmod(fd, 0o600)                                                 # owner-only, even for an old empty file
        with os.fdopen(fd, "w") as f:
            f.write(key + "\n")
        return key, f"new API key {key} saved to {path}"
    if env_key:
        return env_key, "API key from the environment"
    if not is_loopback(host):
        key = secrets.token_urlsafe(24)
        return key, f"generated API key for this run: {key}  (use --api-key-file to keep one)"
    return None, ""


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap on open connections, IPv6, optional TLS and one-line error logs."""

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, handler, max_connections=64, tls=None, log=lambda s: None):
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self._conn_slots = threading.BoundedSemaphore(max_connections)
        self._log = log
        super().__init__(address, handler)
        if tls:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(*tls)
            # the handshake happens in the handler thread (with its timeout), not in the accept loop
            self.socket = ctx.wrap_socket(self.socket, server_side=True, do_handshake_on_connect=False)

    def server_bind(self):
        if self.address_family == socket.AF_INET6 and self.server_address[0] in ("::", ""):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)    # IPv4 clients too
        if self.allow_reuse_address:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)                  # (HTTPServer.server_bind does a slow DNS lookup)
        self.server_address = self.socket.getsockname()
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]

    def process_request(self, request, client_address):
        if not self._conn_slots.acquire(blocking=False):
            try:
                request.settimeout(2)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nConnection: close\r\n"
                                b"Content-Length: 0\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._conn_slots.release()

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, OSError):                           # dropped connections, failed TLS handshakes
            self._log(f"{client_address[0]}: connection error: {type(exc).__name__}: {exc}")
        else:
            traceback.print_exc()


# ---------------------------------------------------------------- HTTP

def make_handler(engine, model_name, defaults, show_reasoning=True, api_key=None, cors_origin=None,
                 heartbeat=10.0, io_timeout=60.0, check_host=False, log=lambda s: None):
    key = api_key.encode() if api_key else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mlx-moe-ssd-streaming"
        timeout = io_timeout                  # a client that stalls (reading or writing) does not hold a thread forever

        def version_string(self):
            return self.server_version        # no Python version in the Server header

        def log_request(self, code="-", size="-"):
            path = getattr(self, "path", "")                  # (not set when the request line itself was bad)
            if path.split("?")[0] != "/health":               # health checks would drown everything else
                log(f"{self.client_address[0]} {getattr(self, 'command', None) or '-'} {path[:200] or '-'} -> {code}")

        def send_error(self, code, message=None, explain=None):
            """Protocol-level errors (bad request line, unknown method, ...) as JSON with a proper status line;
            the standard library answers some of them with an HTML page and no status line at all."""
            self.request_version = "HTTP/1.1" if self.request_version == "HTTP/0.9" else self.request_version
            try:
                phrase = HTTPStatus(code).phrase
            except ValueError:
                phrase = "error"
            try:
                self._fail(ApiError(code, message or phrase, "http_error"))
            except OSError:
                self.close_connection = True

        def log_error(self, fmt, *args):
            log(f"{self.client_address[0]}: " + fmt % args)

        # -- helpers
        def _cors(self, preflight=False):
            if not cors_origin:
                return
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Vary", "Origin")
            if preflight:
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Max-Age", "600")
                if self.headers.get("Access-Control-Request-Private-Network"):   # browsers asking about a LAN host
                    self.send_header("Access-Control-Allow-Private-Network", "true")

        def _headers(self, status, ctype, length=None, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            if length is not None:
                self.send_header("Content-Length", str(length))
                if self.close_connection:
                    self.send_header("Connection", "close")        # tell the client; do not just hang up
            else:
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")        # reverse proxies: do not buffer the stream
                self.send_header("Connection", "close")
                self.close_connection = True
            self._cors()
            self.end_headers()

        def _json(self, status, obj, extra=None):
            data = json.dumps(obj).encode()
            self._headers(status, "application/json", len(data), extra)
            self.wfile.write(data)

        def _authorized(self):
            if key is None:
                return True
            sent = (self.headers.get("Authorization") or "").encode("utf-8", "replace")
            return hmac.compare_digest(sent, b"Bearer " + key)

        def _fail(self, e):
            self.close_connection = True      # the request body may be unread
            self._json(e.status, e.body(), e.headers)

        def _guard(self, fn):
            """Run a route; whatever goes wrong becomes a response (or a closed connection), never a stack trace."""
            try:
                fn()
            except OSError:                   # the client went away or stalled
                self.close_connection = True
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.close_connection = True
                try:
                    self._fail(ApiError(500, f"{type(e).__name__}: {e}", "server_error", kind="server_error"))
                except OSError:
                    pass

        def _check_access(self):
            if check_host and not is_loopback(host_header_name(self.headers.get("Host"))):
                # a web page can reach a local server through a DNS name it controls (DNS rebinding)
                raise ApiError(403, "unexpected Host header (this server only answers to localhost)", "bad_host",
                               kind="permission_error")
            if not self._authorized():
                raise ApiError(401, "invalid API key", "invalid_api_key", kind="authentication_error",
                               headers={"WWW-Authenticate": "Bearer"})

        def _client_gone(self):
            """The client hung up while we wait (a read of 0 bytes means EOF)."""
            try:
                if select.select([self.connection], [], [], 0)[0]:
                    if self.connection.recv(1) == b"":
                        return True
                    self.close_connection = True          # it sent more than we expected; don't reuse the connection
            except (OSError, ValueError):
                return True
            return False

        # -- routes
        def do_OPTIONS(self):
            self.send_response(204)
            self._cors(preflight=True)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            self._guard(self._get)

        def do_POST(self):
            self._guard(self._post)

        def _get(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/health":
                    return self._json(200, {"status": "ok", **engine.status()})
                self._check_access()
                if path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [
                        {"id": model_name, "object": "model", "created": 0, "owned_by": "local"}]})
                raise ApiError(404, f"unknown path {self.path}", "not_found", kind="not_found_error")
            except ApiError as e:
                self._fail(e)

        def _read_body(self):
            if self.headers.get("Transfer-Encoding"):
                raise ApiError(411, "send a Content-Length (chunked request bodies are not supported)")
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":                  # also stops a web page posting here without a preflight
                raise ApiError(415, "Content-Type must be application/json")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError(400, "invalid Content-Length") from None
            if n <= 0:
                raise ApiError(400, "missing request body")
            if n > MAX_BODY:
                raise ApiError(413, f"request body larger than {MAX_BODY >> 20} MB")
            try:
                data = self.rfile.read(n)
            except TimeoutError:
                raise ApiError(408, "timed out waiting for the request body") from None
            if len(data) < n:
                raise ApiError(400, "the request body is shorter than its Content-Length")
            try:
                return json.loads(data)
            except (ValueError, RecursionError):
                raise ApiError(400, "the request body is not valid JSON") from None

        def _post(self):
            path = self.path.split("?")[0].rstrip("/")
            try:
                self._check_access()
                if path != "/v1/chat/completions":
                    raise ApiError(404, f"unknown path {self.path}", "not_found", kind="not_found_error")
                req = parse_request(self._read_body(), defaults)
                job = Job(req, self.client_address[0])
                engine.submit(job)
            except ApiError as e:
                return self._fail(e)
            cid, created = "chatcmpl-" + secrets.token_hex(12), int(time.time())
            try:
                if req["stream"]:
                    self._stream(job, cid, created, req)
                else:
                    self._blocking(job, cid, created)
            except BaseException:
                job.cancelled.set()           # the client went away or stalled (or we failed): stop generating
                raise

        def _next(self, job, idle=lambda: None):
            """The job's next event. While it is silent (queued, or prefilling), watch the client."""
            while True:
                try:
                    return job.events.get(timeout=heartbeat)
                except queue.Empty:
                    if self._client_gone():
                        raise ConnectionAbortedError("client hung up while waiting")
                    idle()

        def _blocking(self, job, cid, created):
            while True:
                kind, payload = self._next(job)
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

            def keep_alive():                 # SSE comment: keeps proxies and client timeouts quiet while we wait
                start()
                self.wfile.write(b": waiting\n\n")
                self.wfile.flush()

            while True:
                kind, payload = self._next(job, keep_alive)
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


def serve(engine, host, port, model_name, defaults, max_connections=64, tls=None, **kw):
    """Start the HTTP server in a background thread; the caller then runs `engine.run_forever()`.

    kw goes to make_handler: show_reasoning, api_key, cors_origin, heartbeat, io_timeout, log.
    Without an API key a loopback server also checks the Host header (DNS rebinding)."""
    kw.setdefault("check_host", is_loopback(host) and not kw.get("api_key"))
    httpd = Server((host, port), make_handler(engine, model_name, defaults, **kw), max_connections, tls,
                   log=kw.get("log", lambda s: None))
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    return httpd
