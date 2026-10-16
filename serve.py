"""OpenAI-compatible server for Muse-Spark-Open-816B-A42B-3bit, streaming experts from SSD.

  .venv/bin/python serve.py [--port 8000]
  curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

Point any OpenAI client at http://127.0.0.1:8000/v1 (any API key, unless --api-key is set).
"""
import argparse
import sys
from pathlib import Path

from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
import openai_server as srv  # noqa: E402
from chat_session import ChatSession  # noqa: E402

ap = argparse.ArgumentParser()
cc.add_model_args(ap, ROOT)
cc.add_sampling_args(ap, ROOT / "models/Muse-Spark-Open-816B-A42B-3bit", greedy_default=False)
ap.add_argument("--host", default="127.0.0.1", help="listen address (the default only accepts local connections)")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--api-key", default=None, help="require 'Authorization: Bearer KEY'")
ap.add_argument("--cors-origin", default=None, help="send CORS headers allowing this origin (off by default)")
ap.add_argument("--max-tokens", type=int, default=4096,
                help="default reply length when the request has no max_tokens (reasoning counts too)")
ap.add_argument("--max-context", type=int, default=16384, help="most tokens in a conversation")
ap.add_argument("--small-step", type=int, default=8)
ap.add_argument("--small-below", type=int, default=48)
args = ap.parse_args()

model, cache, tok = cc.load_all(args, ROOT)
session = ChatSession(model, tok, reasoning_effort=args.reasoning_effort, max_context=args.max_context,
                      small_step=args.small_step, small_below=args.small_below)
engine = srv.Engine(session, lambda t, p, k: make_sampler(temp=t, top_p=p, top_k=k))
defaults = dict(max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p, top_k=args.top_k)
name = Path(args.model).name.lower()
httpd = srv.serve(engine, args.host, args.port, name, defaults, show_reasoning=not args.hide_reasoning,
                  api_key=args.api_key, cors_origin=args.cors_origin)
print(f"Serving {name} (Built with Muse Spark Open) on http://{args.host}:{args.port}/v1 - "
      f"reasoning effort {args.reasoning_effort}, up to {args.max_context} tokens of context. Ctrl-C to stop.",
      file=sys.stderr, flush=True)
try:
    engine.run_forever()
except KeyboardInterrupt:
    print("\nstopping", file=sys.stderr)
finally:
    httpd.shutdown()
