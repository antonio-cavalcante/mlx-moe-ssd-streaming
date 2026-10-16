"""OpenAI-compatible server for Muse-Spark-Open-816B-A42B-3bit, streaming experts from SSD.

  .venv/bin/python serve.py                       # this machine only: http://127.0.0.1:8000/v1
  .venv/bin/python serve.py --lan                 # the local network (an API key is generated for you)
  .venv/bin/python serve.py --lan --api-key-file ~/.llm-server-key   # same key every run
  curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

Point any OpenAI client at the printed URL. Requests are answered one at a time; extra ones wait their turn.
"""
import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
import openai_server as srv  # noqa: E402
from chat_session import ChatSession  # noqa: E402

KV_BYTES_PER_TOKEN = 64 << 10      # measured: 16 full-attention layers x K and V


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_model_args(ap, ROOT)
    cc.add_sampling_args(ap, ROOT / "models/Muse-Spark-Open-816B-A42B-3bit", greedy_default=False)
    net = ap.add_argument_group("network")
    net.add_argument("--host", default="127.0.0.1",
                     help="listen address; 127.0.0.1 = this machine only, 0.0.0.0 = every interface")
    net.add_argument("--lan", action="store_true", help="shorthand for --host 0.0.0.0")
    net.add_argument("--port", type=int, default=8000)
    net.add_argument("--api-key", default=None,
                     help="require 'Authorization: Bearer KEY' (visible in `ps`; prefer --api-key-file)")
    net.add_argument("--api-key-file", default=None,
                     help="read the key from this file; created with a random key if it does not exist")
    net.add_argument("--no-auth", action="store_true", help="allow a network server without an API key")
    net.add_argument("--tls-cert", default=None, help="serve https with this certificate (PEM)")
    net.add_argument("--tls-key", default=None, help="private key for --tls-cert (PEM)")
    net.add_argument("--cors-origin", default=None, help="send CORS headers allowing this origin (off by default)")
    lim = ap.add_argument_group("limits")
    lim.add_argument("--max-queue", type=int, default=8,
                     help="requests waiting or running at once; more get 429 with Retry-After")
    lim.add_argument("--max-connections", type=int, default=64, help="open connections; more get 503")
    lim.add_argument("--client-timeout", type=float, default=60.0,
                     help="seconds a client may stall while sending or receiving")
    lim.add_argument("--heartbeat", type=float, default=10.0,
                     help="seconds of silence before a streaming client gets a keep-alive")
    lim.add_argument("--sessions", type=int, default=2,
                     help="conversations whose KV cache is kept (about 64 KB per token each); a request for "
                          "a conversation that is not kept is prefilled again (~25 s)")
    lim.add_argument("--max-tokens", type=int, default=4096,
                     help="default reply length when the request has no max_tokens (reasoning counts too)")
    lim.add_argument("--max-tokens-cap", type=int, default=8192, help="the most a request may ask for")
    lim.add_argument("--max-context", type=int, default=16384, help="most tokens in a conversation")
    ap.add_argument("--small-step", type=int, default=8)
    ap.add_argument("--small-below", type=int, default=48)
    args = ap.parse_args(argv)
    if args.lan:
        args.host = "0.0.0.0"
    if bool(args.tls_cert) != bool(args.tls_key):
        ap.error("--tls-cert and --tls-key go together")
    if args.sessions < 1 or args.max_queue < 1 or args.max_connections < 1:
        ap.error("--sessions, --max-queue and --max-connections must be at least 1")
    try:
        args.key, args.key_note = srv.resolve_api_key(args.host, args.api_key, args.api_key_file,
                                                      os.environ.get("LLM_SERVER_API_KEY"), args.no_auth)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    return args


def main():
    args = parse_args()
    model, cache, tok = cc.load_all(args, ROOT)
    sessions = [ChatSession(model, tok, reasoning_effort=args.reasoning_effort, max_context=args.max_context,
                            small_step=args.small_step, small_below=args.small_below)
                for _ in range(args.sessions)]
    def memory():
        return {"memory_gib": {"active": round(mx.get_active_memory() / 2**30, 1),
                               "peak": round(mx.get_peak_memory() / 2**30, 1)},
                "contexts": [s.context for s in sessions]}

    engine = srv.Engine(sessions, lambda t, p, k: make_sampler(temp=t, top_p=p, top_k=k), max_queue=args.max_queue,
                        probe=memory)
    defaults = dict(max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p, top_k=args.top_k,
                    max_tokens_cap=args.max_tokens_cap, max_chars=args.max_context * srv.BYTES_PER_TOKEN_MAX)
    name = Path(args.model).name.lower()
    tls = (args.tls_cert, args.tls_key) if args.tls_cert else None
    try:
        httpd = srv.serve(engine, args.host, args.port, name, defaults, show_reasoning=not args.hide_reasoning,
                          api_key=args.key, cors_origin=args.cors_origin, heartbeat=args.heartbeat,
                          io_timeout=args.client_timeout, max_connections=args.max_connections, tls=tls,
                          log=lambda s: print(s, file=sys.stderr, flush=True))
    except OSError as e:
        sys.exit(f"cannot listen on {args.host}:{args.port}: {e}")
    kv_gib = args.sessions * args.max_context * KV_BYTES_PER_TOKEN / 2**30
    print(f"Serving {name} (Built with Muse Spark Open) - reasoning effort {args.reasoning_effort}, "
          f"{args.sessions} conversation(s) of up to {args.max_context} tokens (KV cache up to ~{kv_gib:.1f} GB).",
          file=sys.stderr)
    for url in srv.server_urls(args.host, httpd.server_address[1], "https" if tls else "http"):
        print(f"  {url}", file=sys.stderr)
    if args.key_note:
        print(f"  {args.key_note}", file=sys.stderr)
    if args.key and not tls and not srv.is_loopback(args.host):
        print("  note: without TLS the API key and the conversations cross the network in clear text",
              file=sys.stderr)
    print("Ctrl-C to stop.", file=sys.stderr, flush=True)
    try:
        engine.run_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        engine.stop()
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
