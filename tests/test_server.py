"""OpenAI-compatible server: HTTP layer with a scripted session, then end to end on the tiny checkpoint.

Run: .venv/bin/python -I tests/test_server.py   (no real model needed)
"""
import http.client
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import chat_session as cs  # noqa: E402
import expert_stream as es  # noqa: E402
import muse_spark_1975 as ms  # noqa: E402
import openai_server as srv  # noqa: E402
import test_chat as tc  # noqa: E402
import test_tiny as tt  # noqa: E402

DEFAULTS = dict(max_tokens=64, temp=0.0, top_p=1.0, top_k=0)
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


class FakeSession:
    """Stands in for ChatSession: scripted reply, records how the server drove it.

    gate: an Event; while it is not set, send() blocks (polling cancel) - a job that takes as long as the test wants.
    echo: the answer is "echo:<user text>". fail: send() raises RuntimeError."""

    def __init__(self, effort="medium"):
        self.messages, self.effort, self.sampler = [], effort, None
        self.restores, self.sends, self.cancel_seen, self.slow, self.raise_ctx = [], [], False, False, False
        self.gate, self.echo, self.fail, self.started = None, False, False, threading.Event()
        self._dirty = False

    def restore(self, messages, effort=None):
        self.restores.append(([dict(m) for m in messages], effort))
        self.messages, self.effort = [dict(m) for m in messages], effort or self.effort

    def send(self, text, on_chunks=None, max_tokens=0, cancel=None):
        self.sends.append((text, max_tokens))
        self.started.set()
        if self.fail:
            raise RuntimeError("boom")
        if self.raise_ctx:
            raise cs.ContextFull("too long")
        reused = sum(len(m["content"]) for m in self.messages)
        interrupted = False
        answer = f"echo:{text}" if self.echo else " Hello world! "
        if self.gate is not None:
            while not self.gate.wait(0.01):
                if cancel():
                    self.cancel_seen = interrupted = True
                    break
            self.started.clear()
        if interrupted:
            pass
        elif self.slow:
            for _ in range(300):
                on_chunks([("answer", "x ")])
                time.sleep(0.01)
                if cancel():
                    self.cancel_seen = interrupted = True
                    break
        else:
            chunks = [("reasoning", "  let me"), ("reasoning", " think "), ("answer", " " + answer[:3]), ("answer", answer[3:])]
            for ch in chunks:
                on_chunks([ch])
        self.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        return cs.Turn(answer, "  let me think ", 5, 9, 0.1, 4.0,
                       "interrupted" if interrupted else "stop", 100, False, reused)


class Client:
    def __init__(self, port, key=None):
        self.port, self.key = port, key

    def call(self, method, path, body=None, raw=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = dict(headers or {})
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if data is not None:
            h["Content-Type"] = "application/json"
        c.request(method, path, data, h)
        r = c.getresponse()
        text = r.read().decode()
        c.close()
        return r.status, text

    def chat(self, messages, **kw):
        status, text = self.call("POST", "/v1/chat/completions", {"messages": messages, **kw})
        return status, json.loads(text)

    def stream(self, messages, **kw):
        status, text = self.call("POST", "/v1/chat/completions", {"messages": messages, "stream": True, **kw})
        events = [l[6:] for l in text.split("\n\n") if l.startswith("data: ")]
        return status, events


U = lambda t: {"role": "user", "content": t}      # noqa: E731
A = lambda t: {"role": "assistant", "content": t}  # noqa: E731


def scripted_tests(port, fake):
    c = Client(port)
    print("endpoints")
    st, txt = c.call("GET", "/health")
    check("health", st == 200 and json.loads(txt) == {"status": "ok", "busy": False, "queued": 0, "sessions": 1}, txt)
    st, txt = c.call("GET", "/v1/models")
    check("models", st == 200 and json.loads(txt)["data"][0]["id"] == "test-model")
    check("unknown path -> 404", c.call("GET", "/nope")[0] == 404)

    print("non-streaming + conversation matching")
    st, r = c.chat([U("hi")], max_tokens=33)
    ch = r["choices"][0]
    check("200 + cleaned content", st == 200 and ch["message"]["content"] == "Hello world!", r)
    check("reasoning_content cleaned", ch["message"]["reasoning_content"] == "let me think")
    check("finish_reason stop, usage", ch["finish_reason"] == "stop" and r["usage"]["completion_tokens"] == 9
          and r["usage"]["prompt_tokens"] == 5 + 0 and r["object"] == "chat.completion")
    check("max_tokens passed through", fake.sends[-1] == ("hi", 33))
    check("empty session + empty history -> nothing to restore", fake.restores == [])
    n_restores = len(fake.restores)
    st, r = c.chat([U("hi"), A("Hello world!"), U("and then?")])
    check("same conversation (stripped) -> cache reused, no restore", len(fake.restores) == n_restores, fake.restores)
    check("cached_tokens reported", r["usage"]["prompt_tokens_details"]["cached_tokens"] > 0)
    st, r = c.chat([U("different"), A("x"), U("q")])
    check("different conversation -> restore", len(fake.restores) == n_restores + 1
          and fake.restores[-1][0] == [U("different"), A("x")])
    st, r = c.chat([U("different"), A("x"), U("q")], reasoning_effort="high")
    check("reasoning_effort change -> restore with new effort", fake.restores[-1][1] == "high")
    st, r = c.chat([{"role": "system", "content": "be brief"}, U("q")])
    check("system message accepted", st == 200 and fake.restores[-1][0] == [{"role": "system", "content": "be brief"}])
    st, r = c.chat([{"role": "user", "content": [{"type": "text", "text": "multi"}, {"type": "text", "text": "part"}]}])
    check("text content parts accepted", st == 200 and fake.sends[-1][0] == "multipart")

    print("streaming")
    st, ev = c.stream([U("stream me")], stream_options={"include_usage": True})
    objs = [json.loads(e) for e in ev if e != "[DONE]"]
    check("ends with [DONE]", ev[-1] == "[DONE]")
    check("first chunk is the role", objs[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""})
    content = "".join(o["choices"][0]["delta"].get("content", "") for o in objs if o["choices"])
    reasoning = "".join(o["choices"][0]["delta"].get("reasoning_content", "") for o in objs if o["choices"])
    check("streamed content == 'Hello world!'", content == "Hello world!", content)
    check("streamed reasoning == 'let me think'", reasoning == "let me think", reasoning)
    fin = [o for o in objs if o["choices"] and o["choices"][0]["finish_reason"]]
    check("one finish chunk 'stop'", len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "stop")
    check("usage chunk (include_usage)", objs[-1]["choices"] == [] and objs[-1]["usage"]["completion_tokens"] == 9)
    check("chunk ids are consistent", len({o["id"] for o in objs}) == 1)

    print("errors")
    st, txt = c.call("POST", "/v1/chat/completions", raw=b"{nope")
    check("invalid JSON -> 400", st == 400)
    for name, body in [("tools", {"messages": [U("x")], "tools": [{"type": "function"}]}),
                       ("n=2", {"messages": [U("x")], "n": 2}),
                       ("no messages", {}),
                       ("last is assistant", {"messages": [U("x"), A("y")]}),
                       ("image part", {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
                       ("tool role", {"messages": [{"role": "tool", "content": "x"}, U("x")]}),
                       ("late system", {"messages": [U("a"), A("b"), {"role": "system", "content": "s"}, U("c")]}),
                       ("bad temperature", {"messages": [U("x")], "temperature": 9}),
                       ("bad effort", {"messages": [U("x")], "reasoning_effort": "max"}),
                       ("json response_format", {"messages": [U("x")], "response_format": {"type": "json_object"}})]:
        st, txt = c.call("POST", "/v1/chat/completions", body)
        err = json.loads(txt).get("error", {})
        check(f"{name} -> 400 with an error body", st == 400 and err.get("message"), (st, txt))
    fake.raise_ctx = True
    st, r = c.chat([U("long")])
    check("context too long -> 400 context_length_exceeded", st == 400 and r["error"]["code"] == "context_length_exceeded")
    fake.raise_ctx = False
    check("server still fine after errors", c.chat([U("again")])[0] == 200)

    print("client disconnect")
    fake.slow, fake.cancel_seen = True, False
    payload = json.dumps({"messages": [U("slow")], "stream": True}).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    s.recv(200)                      # the first bytes of the stream arrived
    s.shutdown(socket.SHUT_RDWR)     # hang up mid-stream
    s.close()
    deadline = time.time() + 10
    while not fake.cancel_seen and time.time() < deadline:
        time.sleep(0.05)
    check("generation cancelled when the client disconnects", fake.cancel_seen)
    fake.slow = False
    check("server serves the next request", c.chat([U("after")])[0] == 200)


def auth_tests(port):
    print("api key")
    anon, good = Client(port), Client(port, "sekret")
    check("health needs no key", anon.call("GET", "/health")[0] == 200)
    check("models without key -> 401", anon.call("GET", "/v1/models")[0] == 401)
    check("chat without key -> 401", anon.chat([U("x")])[0] == 401)
    check("chat with key -> 200", good.chat([U("x")])[0] == 200)


def cleaner_tests():
    print("TextCleaner")
    tcl, out = srv.TextCleaner(), ""
    for piece in ["  ", " Hel", "lo ", " wor", "ld", " \n"]:
        out += tcl.feed(piece)
    check("strips ends, keeps inner spaces", out == "Hello  world", repr(out))
    tcl = srv.TextCleaner()
    check("text is not delayed", tcl.feed(" ab") == "ab" and tcl.feed("c") == "c")


def run_server(engine, test_fn):
    """Engine on this thread (it owns MLX); HTTP in a thread; `test_fn(port)` in a client thread."""
    httpd = srv.serve(engine, "127.0.0.1", 0, "test-model", DEFAULTS)
    port = httpd.server_address[1]
    err = []

    def client():
        try:
            test_fn(port)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            err.append(e)
        finally:
            engine.stop()

    t = threading.Thread(target=client)
    t.start()
    engine.run_forever()
    t.join()
    httpd.shutdown()
    if err:
        FAILS.append(f"exception: {err[0]!r}")


def main():
    cleaner_tests()
    fake = FakeSession()
    engine = srv.Engine(fake, lambda t, p, k: None, log=lambda s: None)
    run_server(engine, lambda port: scripted_tests(port, fake))

    # api key: a separate server instance
    engine2 = srv.Engine(FakeSession(), lambda t, p, k: None, log=lambda s: None)
    httpd = srv.serve(engine2, "127.0.0.1", 0, "test-model", DEFAULTS, api_key="sekret")
    port = httpd.server_address[1]
    t = threading.Thread(target=lambda: (auth_tests(port), engine2.stop()))
    t.start()
    engine2.run_forever()
    t.join()
    httpd.shutdown()

    print("end to end on the tiny checkpoint (real ChatSession)")
    tmp = Path(tempfile.mkdtemp(prefix="srv_ckpt_"))
    try:
        ref = tt.make_checkpoint(tmp, mx.float32)
        model, _ = es.build_streaming_model(tmp, ms, slots_per_layer=6, prefetch_k=4, verbose=False)
        sess = cs.ChatSession(model, tc.make_tokenizer(), max_context=600, small_step=4, small_below=30)
        engine3 = srv.Engine(sess, lambda t, p, k: make_sampler(temp=t, top_p=p, top_k=k), log=lambda s: None)

        def e2e(port):
            c = Client(port)
            st, r1 = c.chat([U("hello there")], max_tokens=6, temperature=0)
            check("first request 200, nothing cached", st == 200 and r1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, r1)
            ctx1 = sess.context
            reply = r1["choices"][0]["message"]["content"]
            st, r2 = c.chat([U("hello there"), A(reply), U("and more")], max_tokens=6, temperature=0)
            check("follow-up reuses the KV cache", st == 200 and r2["usage"]["prompt_tokens_details"]["cached_tokens"] == ctx1, r2)
            st, ev = c.stream([U("hello there"), A(reply), U("and more"), A(r2["choices"][0]["message"]["content"]),
                               U("third")], max_tokens=6)
            check("streamed follow-up works", st == 200 and ev[-1] == "[DONE]")
            st, r4 = c.chat([U("an unrelated question")], max_tokens=4)
            check("unrelated conversation is rebuilt (nothing cached)", st == 200
                  and r4["usage"]["prompt_tokens_details"]["cached_tokens"] == 0)
            st, r5 = c.chat([U("x" * 700)], max_tokens=4)
            check("over-long prompt -> 400 context_length_exceeded", st == 400
                  and r5["error"]["code"] == "context_length_exceeded", r5)
            check("server alive afterwards", c.chat([U("ok")], max_tokens=3)[0] == 200)

        run_server(engine3, e2e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
