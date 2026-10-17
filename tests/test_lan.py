"""Server on a network: access control, limits, queueing, several conversations, hostile input.

Run: .venv/bin/python -I tests/test_lan.py   (no real model needed)
"""
import contextlib
import http.client
import io
import json
import os
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import chat_session as cs  # noqa: E402
import expert_stream as es  # noqa: E402
import openai_server as srv  # noqa: E402
import serve  # noqa: E402
import test_chat as tc  # noqa: E402
import test_server as ts  # noqa: E402
import test_tiny as tt  # noqa: E402

check, U, A = ts.check, ts.U, ts.A
DEFAULTS = dict(ts.DEFAULTS)
SECS = 20          # nothing in these tests should take longer than this


# ---------------------------------------------------------------- harness

def scenario(test_fn, sessions=1, max_queue=8, host="127.0.0.1", defaults=None, effort="medium", probe=None,
             **serve_kw):
    """A server on a FakeSession pool; the engine runs here (it owns the model), `test_fn` in a thread."""
    fakes = [ts.FakeSession(effort) for _ in range(sessions)]
    logs = []
    engine = srv.Engine(fakes, lambda t, p, k: None, log=logs.append, max_queue=max_queue, probe=probe)
    serve_kw.setdefault("log", logs.append)
    httpd = srv.serve(engine, host, 0, "test-model", {**DEFAULTS, **(defaults or {})}, **serve_kw)
    port = httpd.server_address[1]
    err = []

    def client():
        try:
            test_fn(port, fakes, engine, logs)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            err.append(e)
        finally:
            for f in fakes:
                if f.gate is not None:
                    f.gate.set()                  # never leave the engine blocked
            engine.stop()

    t = threading.Thread(target=client)
    t.start()
    engine.run_forever()
    t.join()
    httpd.shutdown()
    httpd.server_close()
    if err:
        ts.FAILS.append(f"exception: {err[0]!r}")


def request(port, method, path, body=None, headers=None, raw=None, host="127.0.0.1"):
    """(status, {lower-case header: value}, text)"""
    c = http.client.HTTPConnection(host, port, timeout=SECS)
    h = dict(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    if data is not None:
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    text = r.read().decode(errors="replace")
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, text)
    c.close()
    return out


def chat_body(*msgs, **kw):
    return {"messages": list(msgs), **kw}


def raw_exchange(port, data, read_s=3.0, host="127.0.0.1", family=socket.AF_INET, sock=None):
    """Send raw bytes, return everything the server answers until it closes (or read_s)."""
    s = sock or socket.socket(family, socket.SOCK_STREAM)
    if sock is None:
        s.settimeout(read_s)
        s.connect((host, port))
    if data:
        s.sendall(data)
    out = b""
    try:
        while True:
            b = s.recv(65536)
            if not b:
                break
            out += b
    except (socket.timeout, TimeoutError, ConnectionResetError):
        pass
    s.close()
    return out


def status_of(resp):
    return int(resp.split(b" ", 2)[1]) if resp.startswith(b"HTTP/") else 0


def post_head(path="/v1/chat/completions", length=0, ctype="application/json", extra=b""):
    return (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: {ctype}\r\nContent-Length: {length}\r\n"
            .encode() + extra + b"\r\n")


def async_call(fn, *a, **kw):
    box = {}

    def run():
        try:
            box["r"] = fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=run)
    t.start()
    return t, box


def wait_for(cond, secs=SECS):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- pure functions

def pure_tests():
    print("addresses and hosts")
    check("loopback names", all(srv.is_loopback(h) for h in ["localhost", "127.0.0.1", "127.9.9.9", "::1", "[::1]",
                                                            "LOCALHOST", "app.localhost"]))
    check("network addresses are not loopback", not any(srv.is_loopback(h) for h in
                                                       ["0.0.0.0", "::", "192.168.1.5", "example.com", "10.0.0.1"]))
    check("Host header parsing", [srv.host_header_name(h) for h in
                                  ["localhost:8000", "127.0.0.1", "[::1]:8000", "[::1]", "Evil.COM:80", "", None]]
          == ["localhost", "127.0.0.1", "::1", "::1", "evil.com", "", ""])
    check("loopback server: one URL", srv.server_urls("127.0.0.1", 8000) == ["http://127.0.0.1:8000/v1"])
    urls = srv.server_urls("0.0.0.0", 8000)
    check("0.0.0.0: lists reachable URLs, not 0.0.0.0", urls and all("0.0.0.0" not in u for u in urls)
          and urls[-1] == "http://127.0.0.1:8000/v1", urls)
    check("https and IPv6 URLs", srv.server_urls("::1", 9, "https") == ["https://[::1]:9/v1"])
    check("a specific address is used as given", srv.server_urls("192.168.1.5", 8000) == ["http://192.168.1.5:8000/v1"])

    print("API key resolution")
    tmp = Path(tempfile.mkdtemp(prefix="keys_"))
    try:
        check("loopback, nothing given: no key", srv.resolve_api_key("127.0.0.1")[0] is None)
        k, note = srv.resolve_api_key("0.0.0.0")
        check("network address, nothing given: a key is generated and shown", k and len(k) >= 24 and k in note, note)
        check("two generated keys differ", srv.resolve_api_key("0.0.0.0")[0] != k)
        check("--no-auth: no key, with a warning", srv.resolve_api_key("0.0.0.0", no_auth=True)[0] is None
              and "NO AUTHENTICATION" in srv.resolve_api_key("0.0.0.0", no_auth=True)[1])
        check("--no-auth with a key is an error",
              _raises(lambda: srv.resolve_api_key("0.0.0.0", api_key="k", no_auth=True), ValueError))
        check("explicit key wins", srv.resolve_api_key("0.0.0.0", "mine", tmp / "f", "env")[0] == "mine")
        f = tmp / "key"
        k1, note1 = srv.resolve_api_key("0.0.0.0", key_file=f)
        k2, note2 = srv.resolve_api_key("0.0.0.0", key_file=f)
        check("key file: created once, then reused", k1 == k2 and f.read_text().strip() == k1 and k1 in note1
              and k1 not in note2, (note1, note2))
        check("key file is owner-only", stat.S_IMODE(f.stat().st_mode) == 0o600, oct(f.stat().st_mode))
        f.write_text("")
        f.chmod(0o644)
        k3, _ = srv.resolve_api_key("0.0.0.0", key_file=f)
        check("an empty key file gets a key and tight permissions", k3 and f.read_text().strip() == k3
              and stat.S_IMODE(f.stat().st_mode) == 0o600)
        check("environment key used when no other", srv.resolve_api_key("0.0.0.0", env_key="fromenv")[0] == "fromenv")
        check("key file in a missing directory is an OSError",
              _raises(lambda: srv.resolve_api_key("0.0.0.0", key_file=tmp / "no" / "dir" / "k"), OSError))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("serve.py arguments")
    def args(*a):
        with contextlib.redirect_stderr(io.StringIO()):
            return serve.parse_args(list(a))
    d = args()
    check("defaults: this machine only, no key, 2 conversations", d.host == "127.0.0.1" and d.key is None
          and d.sessions == 2 and d.max_queue == 8)
    l = args("--lan")
    check("--lan: all interfaces and a key", l.host == "0.0.0.0" and l.key)
    check("--lan --no-auth: no key", args("--lan", "--no-auth").key is None)
    check("--host 192.168.1.5 also gets a key", args("--host", "192.168.1.5").key)
    for bad in (["--tls-cert", "c.pem"], ["--sessions", "0"], ["--max-queue", "0"], ["--no-auth", "--api-key", "k"]):
        try:
            args(*bad)
            check(f"{bad} is refused", False)
        except SystemExit as e:
            check(f"{' '.join(bad)} is refused", e.code == 2)

    print("request parsing")
    p = lambda body, **d: srv.parse_request(body, {**DEFAULTS, **d})  # noqa: E731
    bad = lambda body, code=None, **d: _api_error(lambda: p(body, **d), code)  # noqa: E731
    check("a lone surrogate is a 400, not a crash later", bad(chat_body(U("\ud800")), "invalid_text"))
    check("same inside a text part", bad(chat_body({"role": "user", "content": [{"type": "text", "text": "\udfff"}]}),
                                          "invalid_text"))
    check("stream must be a boolean", bad(chat_body(U("x"), stream="yes")))
    check("include_usage must be a boolean", bad(chat_body(U("x"), stream=True, stream_options={"include_usage": 1})))
    check("stream_options must be an object", bad(chat_body(U("x"), stream_options=[1])))
    check("reasoning_effort must be a string", bad(chat_body(U("x"), reasoning_effort=["low"])))
    check("max_tokens: bool, float, zero, negative are refused",
          all(bad(chat_body(U("x"), max_tokens=v)) for v in (True, 1.5, 0, -3, "9")))
    check("temperature NaN / inf are refused", bad(chat_body(U("x"), temperature=float("nan")))
          and bad(chat_body(U("x"), temperature=float("inf"))))
    check("max_tokens is capped", p(chat_body(U("x"), max_tokens=10**9), max_tokens_cap=50)["max_tokens"] == 50)
    check("default max_tokens applies", p(chat_body(U("x")))["max_tokens"] == DEFAULTS["max_tokens"])
    check("max_completion_tokens is accepted", p(chat_body(U("x"), max_completion_tokens=7))["max_tokens"] == 7)
    check("an over-long conversation is refused before tokenizing",
          bad(chat_body(U("x" * 101)), "context_length_exceeded", max_chars=100))
    check("... but exactly at the limit is fine", p(chat_body(U("x" * 100)), max_chars=100)["messages"])
    check("content null is an empty string", p(chat_body({"role": "user", "content": None}))["messages"][-1]["content"] == "")
    check("a non-dict message is refused", bad({"messages": ["hi"]}))
    check("a non-dict body is refused", bad([1, 2]))
    check("assistant tool_calls are refused", bad(chat_body({"role": "assistant", "content": "x", "tool_calls": [1]}, U("y"))))
    check("same_conversation ignores outer whitespace only",
          srv.same_conversation([A(" hi \n")], [A("hi")]) and not srv.same_conversation([A("hi there")], [A("hi")])
          and not srv.same_conversation([A("a")], [U("a")]) and not srv.same_conversation([A("a")], []))


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def _api_error(fn, code=None):
    try:
        fn()
    except srv.ApiError as e:
        return e.status == 400 and (code is None or e.code == code)
    return False


# ---------------------------------------------------------------- the engine: several conversations

def pool_tests():
    print("several conversations (session pool)")

    def body(port, fakes, engine, logs):
        a, b = fakes
        c = lambda *m, **kw: request(port, "POST", "/v1/chat/completions", chat_body(*m, **kw))  # noqa: E731
        r = lambda x: json.loads(x[2])  # noqa: E731
        a.echo = b.echo = True
        r1 = r(c(U("alpha")))
        r2 = r(c(U("beta")))
        check("two new conversations take the two (fresh) sessions, nothing to rebuild", len(a.sends) == 1
              and len(b.sends) == 1 and a.restores == [] and b.restores == [], (a.sends, b.sends, b.restores))
        r3 = r(c(U("alpha"), A(r1["choices"][0]["message"]["content"]), U("alpha 2")))
        r4 = r(c(U("beta"), A(r2["choices"][0]["message"]["content"]), U("beta 2")))
        check("each follow-up finds its own session, no rebuild", len(a.sends) == 2 and len(b.sends) == 2
              and a.restores == [] and b.restores == [], (a.restores, b.restores))
        check("and reports its cache", r3["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
              and r4["usage"]["prompt_tokens_details"]["cached_tokens"] > 0)
        c(U("gamma"))
        check("a third conversation takes over the least recently used session (alpha's)",
              len(a.restores) == 1 and a.restores[0][0] == [] and len(a.sends) == 3, (a.restores, b.restores))
        r5 = c(U("beta"), A(r2["choices"][0]["message"]["content"]), U("beta 2"), A(r4["choices"][0]["message"]["content"]),
               U("beta 3"))
        check("beta is still cached after that", len(b.sends) == 3 and b.restores == [] and r5[0] == 200)

        print("reasoning effort")
        a.restores.clear(); b.restores.clear()
        c(U("hi"), reasoning_effort="high")
        check("a request's effort is passed to the session", any(rs[1] == "high" for f in fakes for rs in f.restores))
        for f in fakes:
            f.restores.clear()
        c(U("another conversation"))
        check("a request without an effort gets the server default, not the previous request's",
              all(rs[1] == "medium" for f in fakes for rs in f.restores) and any(f.restores for f in fakes),
              [f.restores for f in fakes])
        check("health reports the pool", json.loads(request(port, "GET", "/health")[2])["sessions"] == 2)
        check("health includes what the probe reported after the last job",
              json.loads(request(port, "GET", "/health")[2]).get("probed") == 6)
        engine.probe = lambda: 1 / 0
        sys.stderr = io.StringIO()                    # the engine prints the traceback of the failure we cause
        c(U("break the probe"))
        sys.stderr = sys.__stderr__
        check("a broken probe does not stop the server", c(U("still serving"))[0] == 200 and engine.probe is None)

    scenario(body, sessions=2, probe=lambda: {"probed": 6})

    def concurrent(port, fakes, engine, logs):
        for f in fakes:
            f.echo = True
        results, threads = {}, []
        for i in range(6):
            def run(i=i):
                results[i] = request(port, "POST", "/v1/chat/completions", chat_body(U(f"msg{i}")))
            t = threading.Thread(target=run)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(SECS)
        ok = all(results.get(i, (0,))[0] == 200 and json.loads(results[i][2])["choices"][0]["message"]["content"]
                 == f"echo:msg{i}" for i in range(6))
        check("6 simultaneous clients each get their own answer", ok, results)
        check("all of them were served, one at a time", sum(len(f.sends) for f in fakes) == 6)

    scenario(concurrent, sessions=2)


# ---------------------------------------------------------------- queueing and backpressure

def queue_tests():
    print("queue, busy server, cancellation while waiting")

    def body(port, fakes, engine, logs):
        f = fakes[0]
        f.gate = threading.Event()
        t1, b1 = async_call(request, port, "POST", "/v1/chat/completions", chat_body(U("first")))
        check("the first request starts", f.started.wait(SECS))
        t2, b2 = async_call(request, port, "POST", "/v1/chat/completions", chat_body(U("second")))
        check("the second waits", wait_for(lambda: engine.status()["queued"] == 1))
        st, _, txt = request(port, "GET", "/health")
        h = json.loads(txt)
        check("health shows busy + 1 queued", st == 200 and h["busy"] is True and h["queued"] == 1, h)
        st, hd, txt = request(port, "POST", "/v1/chat/completions", chat_body(U("third")))
        err = json.loads(txt)["error"]
        check("a full queue answers 429 with Retry-After", st == 429 and hd.get("retry-after") == "10"
              and err["code"] == "server_busy", (st, hd, txt))
        f.gate.set()
        t1.join(SECS)
        t2.join(SECS)
        check("both accepted requests complete", b1["r"][0] == 200 and b2["r"][0] == 200, (b1, b2))
        check("in order", [s[0] for s in f.sends] == ["first", "second"], f.sends)
        check("afterwards the server is idle and takes new work",
              request(port, "POST", "/v1/chat/completions", chat_body(U("fourth")))[0] == 200
              and json.loads(request(port, "GET", "/health")[2])["queued"] == 0)

    scenario(body, max_queue=2)

    def vanishing(port, fakes, engine, logs):
        f = fakes[0]
        f.gate = threading.Event()
        t1, b1 = async_call(request, port, "POST", "/v1/chat/completions", chat_body(U("running")))
        check("(setup) first request is running", f.started.wait(SECS))
        payload = json.dumps(chat_body(U("ghost"))).encode()
        s = socket.create_connection(("127.0.0.1", port), timeout=SECS)
        s.sendall(post_head(length=len(payload)) + payload)
        check("(setup) the ghost is queued", wait_for(lambda: engine.status()["queued"] == 1))
        s.close()                                     # the client gives up while it waits in line
        time.sleep(0.6)                               # > 2 heartbeats
        f.gate.set()
        t1.join(SECS)
        check("(setup) the running request finished", b1["r"][0] == 200)
        request(port, "POST", "/v1/chat/completions", chat_body(U("after")))
        check("a queued request whose client left is never run", [s[0] for s in f.sends] == ["running", "after"],
              f.sends)
        check("and the queue is empty again", engine.status() == {"busy": False, "queued": 0, "sessions": 1})

    scenario(vanishing, heartbeat=0.2)

    def streaming_wait(port, fakes, engine, logs):
        f = fakes[0]
        f.gate = threading.Event()
        t1, b1 = async_call(request, port, "POST", "/v1/chat/completions", chat_body(U("running")))
        check("(setup) first request is running", f.started.wait(SECS))
        payload = json.dumps(chat_body(U("streamed"), stream=True)).encode()
        s = socket.create_connection(("127.0.0.1", port), timeout=SECS)
        s.sendall(post_head(length=len(payload)) + payload)
        buf = b""
        end = time.time() + 5
        while b": waiting" not in buf and time.time() < end:
            buf += s.recv(4096)
        check("a queued streaming client gets headers and keep-alives while it waits",
              buf.startswith(b"HTTP/1.1 200") and b"text/event-stream" in buf and b": waiting" in buf, buf[:200])
        f.gate.set()
        s.settimeout(SECS)
        while not buf.endswith(b"data: [DONE]\n\n"):
            more = s.recv(4096)
            if not more:
                break
            buf += more
        s.close()
        events = [json.loads(l[6:]) for l in buf.decode().split("\n") if l.startswith("data: {")]
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
        check("and then the answer, ending with [DONE]", buf.endswith(b"data: [DONE]\n\n") and text.strip() == "Hello world!",
              (text, buf[-100:]))
        t1.join(SECS)

    scenario(streaming_wait, heartbeat=0.2)

    def shutdown(port, fakes, engine, logs):
        f = fakes[0]
        f.gate = threading.Event()
        t1, b1 = async_call(request, port, "POST", "/v1/chat/completions", chat_body(U("running")))
        check("(setup) running", f.started.wait(SECS))
        t2, b2 = async_call(request, port, "POST", "/v1/chat/completions", chat_body(U("waiting")))
        check("(setup) queued", wait_for(lambda: engine.status()["queued"] == 1))
        engine.stop()
        st, _, txt = request(port, "POST", "/v1/chat/completions", chat_body(U("late")))
        check("a request after stop() is refused with 503", st == 503 and json.loads(txt)["error"]["code"] == "shutting_down",
              (st, txt))
        f.gate.set()
        t1.join(SECS)
        t2.join(SECS)
        check("the running request finishes", b1["r"][0] == 200)
        check("the queued one is told the server is shutting down, not left hanging", b2["r"][0] == 503
              and [s[0] for s in f.sends] == ["running"], (b2, f.sends))

    scenario(shutdown)


# ---------------------------------------------------------------- hostile or broken clients

def protocol_tests():
    print("malformed and hostile requests")

    def body(port, fakes, engine, logs):
        f = fakes[0]
        good = json.dumps(chat_body(U("hi"))).encode()
        st = lambda **kw: request(port, "POST", "/v1/chat/completions", **kw)[0]  # noqa: E731

        check("Content-Type text/plain is refused (a web page could send it without a preflight)",
              st(raw=good, headers={"Content-Type": "text/plain"}) == 415 and f.sends == [])
        check("no Content-Type is refused", status_of(raw_exchange(port, b"POST /v1/chat/completions HTTP/1.1\r\n"
              b"Host: 127.0.0.1\r\nContent-Length: " + str(len(good)).encode() + b"\r\n\r\n" + good)) == 415)
        check("application/json; charset=utf-8 is fine", st(raw=good, headers={"Content-Type": "application/json; charset=utf-8"}) == 200)
        check("chunked bodies get 411", status_of(raw_exchange(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
              b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")) == 411)
        for cl in ("abc", "-5", "0", "1e3", ""):
            r = raw_exchange(port, post_head(length=cl))
            check(f"Content-Length {cl!r} -> 400", status_of(r) == 400, r[:80])
        r = raw_exchange(port, post_head(length=10**9))
        check("a huge Content-Length -> 413 without waiting for the body", status_of(r) == 413, r[:80])
        check("invalid UTF-8 -> 400", st(raw=b"\xff\xfe{\x00") == 400)
        check("JSON nested 200000 deep -> 400, not a crash", st(raw=b"[" * 200000) == 400)
        check("a JSON number that is not an object -> 400", st(raw=b"42") == 400)
        check("a lone-surrogate escape -> 400", st(raw=b'{"messages":[{"role":"user","content":"\\ud800"}]}') == 400)
        check("(none of them reached the model)", len(f.sends) == 1, f.sends)
        r = raw_exchange(port, b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        check("an over-long request line -> 414", status_of(r) == 414, r[:60])
        r = raw_exchange(port, b"GARBAGE / NOT-HTTP\r\n\r\n")
        check("garbage -> 400 with a JSON error", status_of(r) == 400 and b'"error"' in r, r[:120])
        check("an unsupported HTTP version -> 505", status_of(raw_exchange(port, b"GET / HTTP/9.9\r\nHost: 127.0.0.1\r\n\r\n")) == 505)
        check("unknown method -> 501", status_of(raw_exchange(port, b"DELETE /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")) == 501)
        check("POST to a GET route -> 404", request(port, "POST", "/v1/models", {})[0] == 404)
        check("GET to the POST route -> 404", request(port, "GET", "/v1/chat/completions")[0] == 404)
        check("query strings and a trailing slash are tolerated", request(port, "GET", "/v1/models?x=1")[0] == 200
              and request(port, "GET", "/v1/models/")[0] == 200)

        # the body of a refused request is never read, so the connection must not be reused for the next request
        r = raw_exchange(port, post_head("/nope", 5) + b"abcde" + b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        check("after an error the server closes the connection (no request smuggling from the unread body)",
              r.count(b"HTTP/1.1") == 1 and b"Connection: close" in r, r[:300])

        c = http.client.HTTPConnection("127.0.0.1", port, timeout=SECS)
        for _ in range(3):
            c.request("POST", "/v1/chat/completions", good, {"Content-Type": "application/json"})
            r = c.getresponse()
            r.read()
        check("keep-alive: 3 requests on one connection", r.status == 200)
        c.close()
        check("the server is fine after all that", request(port, "POST", "/v1/chat/completions", chat_body(U("ok")))[0] == 200)

    scenario(body)

    def slow_body(port, fakes, engine, logs):
        s = socket.create_connection(("127.0.0.1", port), timeout=SECS)
        s.sendall(post_head(length=100) + b'{"messages"')
        t0 = time.time()
        r = raw_exchange(port, b"", sock=s, read_s=SECS)
        check("a body that stops arriving times out (408) instead of holding a thread",
              status_of(r) == 408 and time.time() - t0 < 5, (r[:60], time.time() - t0))
        s = socket.create_connection(("127.0.0.1", port), timeout=SECS)
        s.sendall(post_head(length=100) + b'{"messages"')
        s.shutdown(socket.SHUT_WR)
        r = raw_exchange(port, b"", sock=s, read_s=SECS)
        check("a body cut short by the client -> 400", status_of(r) == 400, r[:80])
        s = socket.create_connection(("127.0.0.1", port), timeout=SECS)
        t0 = time.time()
        s.settimeout(5)
        check("an idle connection is closed by the server", s.recv(10) == b"" and time.time() - t0 < 4)
        s.close()

    scenario(slow_body, io_timeout=0.4)

    def conn_cap(port, fakes, engine, logs):
        idle = [socket.create_connection(("127.0.0.1", port), timeout=SECS) for _ in range(2)]
        time.sleep(0.2)
        r = raw_exchange(port, b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        check("beyond the connection limit: 503 + Retry-After", status_of(r) == 503 and b"Retry-After" in r, r[:100])
        idle[0].close()
        check("a freed connection can be used again", wait_for(
            lambda: status_of(raw_exchange(port, b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")) == 200))
        idle[1].close()

    scenario(conn_cap, max_connections=2)

    def broken(port, fakes, engine, logs):
        f = fakes[0]
        f.fail = True
        sys.stderr = io.StringIO()                    # the engine prints the traceback of the failure we cause
        st, _, txt = request(port, "POST", "/v1/chat/completions", chat_body(U("x")))
        err = json.loads(txt)["error"]
        check("an internal error is a JSON 500", st == 500 and err["type"] == "server_error" and "boom" in err["message"], txt)
        check("the session is marked for a rebuild", f._dirty is True)
        st, _, txt = request(port, "POST", "/v1/chat/completions", chat_body(U("x"), stream=True))
        check("also when streaming", st == 500)
        f.fail = False
        sys.stderr = sys.__stderr__
        check("the server keeps serving", request(port, "POST", "/v1/chat/completions", chat_body(U("y")))[0] == 200)

    scenario(broken)


def access_tests():
    print("access control")

    def local(port, fakes, engine, logs):
        h = lambda host: request(port, "GET", "/v1/models", headers={"Host": host})[0]  # noqa: E731
        check("localhost / 127.0.0.1 / [::1] Host headers are accepted",
              h(f"localhost:{port}") == 200 and h(f"127.0.0.1:{port}") == 200 and h(f"[::1]:{port}") == 200)
        st, _, txt = request(port, "GET", "/v1/models", headers={"Host": "rebind.example.com"})
        check("any other Host is refused on a key-less loopback server (DNS rebinding)", st == 403
              and json.loads(txt)["error"]["code"] == "bad_host", txt)
        check("POSTs too", request(port, "POST", "/v1/chat/completions", chat_body(U("x")),
                                   headers={"Host": "rebind.example.com"})[0] == 403 and fakes[0].sends == [])
        check("/health stays open", request(port, "GET", "/health", headers={"Host": "rebind.example.com"})[0] == 200)

    scenario(local)

    def keyed(port, fakes, engine, logs):
        ok = {"Authorization": "Bearer s3cret"}
        hdr = lambda v: {"Authorization": v} if v else {}  # noqa: E731
        for name, v in [("no key", None), ("wrong key", "Bearer nope"), ("wrong scheme", "Basic s3cret"),
                        ("key without scheme", "s3cret"), ("longer key", "Bearer s3cret-and-more"),
                        ("non-ASCII", "Bearer s3crét"), ("lower-case scheme", "bearer s3cret")]:
            st, hd, _ = request(port, "GET", "/v1/models", headers=hdr(v))
            check(f"{name} -> 401 + WWW-Authenticate", st == 401 and hd.get("www-authenticate") == "Bearer")
        check("POST without a key -> 401, nothing run", request(port, "POST", "/v1/chat/completions", chat_body(U("x")))[0] == 401
              and fakes[0].sends == [])
        check("the Server header does not reveal the Python version", "python" not in
              request(port, "GET", "/v1/models", headers=ok)[1].get("server", "").lower())
        check("the right key works", request(port, "GET", "/v1/models", headers=ok)[0] == 200
              and request(port, "POST", "/v1/chat/completions", chat_body(U("x")), headers=ok)[0] == 200)
        check("with a key, the Host header is not restricted (LAN names)",
              request(port, "GET", "/v1/models", headers={**ok, "Host": "mac.local:8000"})[0] == 200)
        check("the key never appears in the log", not any("s3cret" in line for line in logs), logs)

    scenario(keyed, api_key="s3cret")

    def open_lan(port, fakes, engine, logs):
        check("a server on 0.0.0.0 answers on a LAN-style Host header",
              request(port, "GET", "/v1/models", headers={"Host": "192.168.1.5:8000"})[0] == 200)

    scenario(open_lan, host="0.0.0.0", api_key=None)

    print("CORS")

    def cors(port, fakes, engine, logs):
        pre = request(port, "OPTIONS", "/v1/chat/completions", headers={
            "Origin": "http://app.lan", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Private-Network": "true", "Access-Control-Request-Headers": "authorization"})
        check("preflight answered without a key", pre[0] == 204 and pre[1].get("access-control-allow-origin") == "http://app.lan")
        check("preflight allows private-network access (browser -> LAN host)", pre[1].get("access-control-allow-private-network") == "true")
        check("preflight lists Authorization", "Authorization" in pre[1].get("access-control-allow-headers", ""))
        bad = request(port, "GET", "/v1/models")
        check("an error response carries the CORS header too, so the page can read it",
              bad[0] == 401 and bad[1].get("access-control-allow-origin") == "http://app.lan")

    scenario(cors, api_key="k", cors_origin="http://app.lan")

    def no_cors(port, fakes, engine, logs):
        r = request(port, "OPTIONS", "/v1/chat/completions", headers={"Origin": "http://evil"})
        check("CORS is off by default", r[0] == 204 and "access-control-allow-origin" not in r[1])
        r = request(port, "POST", "/v1/chat/completions", chat_body(U("x")), headers={"Origin": "http://evil"})
        check("... for responses too", "access-control-allow-origin" not in r[1])

    scenario(no_cors)


def log_tests():
    print("request log")

    def body(port, fakes, engine, logs):
        request(port, "GET", "/health")
        request(port, "POST", "/v1/chat/completions", chat_body(U("x")))
        request(port, "GET", "/nope")
        check("one line per request with client, route and status", any("127.0.0.1 POST /v1/chat/completions -> 200" in l for l in logs)
              and any("GET /nope -> 404" in l for l in logs), logs)
        check("health checks are not logged", not any("/health" in l for l in logs))
        check("the engine's line names the client and session", any(l.startswith("chat.completions for 127.0.0.1: session 1/1") for l in logs))

    scenario(body)


# ---------------------------------------------------------------- IPv6, TLS

def transport_tests():
    print("IPv6")
    for host, probe in (("::1", "::1"), ("::", "::1")):
        fakes_logs = []
        try:
            engine = srv.Engine([ts.FakeSession()], lambda t, p, k: None, log=fakes_logs.append)
            httpd = srv.serve(engine, host, 0, "test-model", DEFAULTS, api_key=None if host == "::1" else "k")
        except OSError as e:
            print(f"  SKIP  cannot listen on {host}: {e}")
            continue
        port = httpd.server_address[1]
        out = {}

        def client(port=port, out=out, engine=engine, probe=probe, host=host):
            try:
                hdr = f"GET /v1/models HTTP/1.1\r\nHost: [::1]:{port}\r\n" + ("Authorization: Bearer k\r\n" if host == "::" else "")
                out["v6"] = status_of(raw_exchange(port, (hdr + "Connection: close\r\n\r\n").encode(), host=probe,
                                                   family=socket.AF_INET6))
                if host == "::":                              # dual stack: IPv4 clients reach it too
                    out["v4"] = status_of(raw_exchange(port, (hdr + "Connection: close\r\n\r\n").encode()))
            finally:
                engine.stop()

        t = threading.Thread(target=client)
        t.start()
        engine.run_forever()
        t.join()
        httpd.shutdown()
        httpd.server_close()
        check(f"listening on {host}: an IPv6 client is served", out.get("v6") == 200, out)
        if host == "::":
            check("listening on '::': an IPv4 client is served too", out.get("v4") == 200, out)

    print("TLS")
    openssl = shutil.which("openssl")
    tmp = Path(tempfile.mkdtemp(prefix="tls_"))
    try:
        if not openssl:
            print("  SKIP  no openssl")
            return
        done = subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(tmp / "k.pem"),
                               "-out", str(tmp / "c.pem"), "-days", "1", "-subj", "/CN=localhost"],
                              capture_output=True)
        if done.returncode:
            print("  SKIP  openssl could not make a certificate:", done.stderr.decode()[:100])
            return

        def body(port, fakes, engine, logs):
            ctx = ssl.create_default_context()
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
            stalled = socket.create_connection(("127.0.0.1", port), timeout=SECS)      # connects, never handshakes
            t0 = time.time()
            c = http.client.HTTPSConnection("127.0.0.1", port, timeout=SECS, context=ctx)
            c.request("POST", "/v1/chat/completions", json.dumps(chat_body(U("over tls"))), {
                "Content-Type": "application/json", "Authorization": "Bearer k"})
            r = c.getresponse()
            data = json.loads(r.read())
            c.close()
            check("an HTTPS request works", r.status == 200 and data["choices"][0]["message"]["content"] == "Hello world!")
            check("a client that never finishes the TLS handshake does not block others", time.time() - t0 < 5)
            stalled.close()
            r = raw_exchange(port, b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n", read_s=3)
            check("plain HTTP to the TLS port gets no answer", status_of(r) == 0, r[:60])
            check("the server still serves TLS afterwards", _https_status(port, ctx) == 200)
            check("the failed handshake was logged in one line", any("connection error" in l for l in logs), logs)

        scenario(body, api_key="k", tls=(str(tmp / "c.pem"), str(tmp / "k.pem")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _https_status(port, ctx):
    c = http.client.HTTPSConnection("127.0.0.1", port, timeout=SECS, context=ctx)
    c.request("GET", "/health")
    s = c.getresponse().status
    c.close()
    return s


# ---------------------------------------------------------------- the real ChatSession, two conversations

def e2e_pool_test():
    print("two conversations on the tiny checkpoint (real ChatSession x2)")
    tmp = Path(tempfile.mkdtemp(prefix="lan_ckpt_"))
    try:
        ref = tt.make_checkpoint(tmp, mx.float32, fixed_routing=True)
        model, _ = es.build_streaming_model(tmp, slots_per_layer=6, prefetch_k=4, verbose=False)
        tok = tc.make_tokenizer()
        sessions = [cs.ChatSession(model, tok, max_context=500, small_step=4, small_below=30) for _ in range(2)]
        engine = srv.Engine(sessions, lambda t, p, k: make_sampler(temp=t, top_p=p, top_k=k), log=lambda s: None)
        httpd = srv.serve(engine, "127.0.0.1", 0, "test-model", DEFAULTS)
        port = httpd.server_address[1]

        def client():
            try:
                chat = lambda *m: json.loads(request(port, "POST", "/v1/chat/completions",  # noqa: E731
                                                     chat_body(*m, max_tokens=5, temperature=0))[2])
                a1 = chat(U("first conversation"))
                b1 = chat(U("second conversation"))
                ca, cb = sessions[0].context, sessions[1].context
                a2 = chat(U("first conversation"), A(a1["choices"][0]["message"]["content"]), U("more"))
                b2 = chat(U("second conversation"), A(b1["choices"][0]["message"]["content"]), U("more"))
                check("conversation 1 kept its cache while 2 was in progress",
                      a2["usage"]["prompt_tokens_details"]["cached_tokens"] == ca, a2["usage"])
                check("conversation 2 too", b2["usage"]["prompt_tokens_details"]["cached_tokens"] == cb, b2["usage"])
                check("they live in different sessions", sessions[0].messages[0]["content"] == "first conversation"
                      and sessions[1].messages[0]["content"] == "second conversation")
                check("each session's cache is consistent", all({c.offset for c in x.cache} == {x.context} for x in sessions))
                # Interleaving must change nothing: the same conversation on a session of its own gives the same tokens.
                # (Not a logits comparison against the stock model: a random tiny MoE has near-tied routers, so
                # even the stock model differs by ~20% between one-shot and incremental on some inputs.)
                for i, (first, s) in enumerate([("first conversation", sessions[0]), ("second conversation", sessions[1])]):
                    solo = cs.ChatSession(model, tok, max_context=500, small_step=4, small_below=30,
                                          sampler=make_sampler(temp=0))
                    solo.send(first, max_tokens=5)
                    solo.send("more", max_tokens=5)
                    check(f"conversation {i + 1} has the same tokens as when it runs alone", s.history == solo.history,
                          (len(s.history), len(solo.history)))
            finally:
                engine.stop()

        t = threading.Thread(target=client)
        t.start()
        engine.run_forever()
        t.join()
        httpd.shutdown()
        httpd.server_close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    pure_tests()
    pool_tests()
    queue_tests()
    protocol_tests()
    access_tests()
    log_tests()
    transport_tests()
    e2e_pool_test()
    print("ALL PASS" if not ts.FAILS else f"FAILURES: {ts.FAILS}")
    sys.exit(1 if ts.FAILS else 0)


if __name__ == "__main__":
    main()
