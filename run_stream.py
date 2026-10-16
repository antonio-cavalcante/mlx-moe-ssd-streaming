"""Generate with Muse-Spark-Open-816B-A42B-3bit, streaming experts from SSD.

  .venv/bin/python run_stream.py --prompt "Explain mixture of experts in 3 sentences." --max-tokens 64
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import expert_stream as es  # noqa: E402
from turns import TurnParser  # noqa: E402
import muse_spark_1975 as ms  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.tokenizer_utils import load as load_tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models/Muse-Spark-Open-816B-A42B-3bit"))
ap.add_argument("--prompt", default="Explain mixture-of-experts models in three sentences.")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--cache-gb", type=float, default=None,
                help="expert slab size; default = GPU recommended working set minus resident minus reserve")
ap.add_argument("--reserve-gb", type=float, default=10.0, help="headroom for KV cache/activations")
ap.add_argument("--group-max", type=int, default=48, help="max experts stacked at once during prefill")
ap.add_argument("--threads", type=int, default=24, help="parallel SSD reads")
ap.add_argument("--prefetch-k", type=int, default=4,
                help="speculatively read the top-k predicted experts of the next layer (0 = off)")
ap.add_argument("--pf-threads", type=int, default=16, help="parallel SSD reads for prefetch")
ap.add_argument("--layer-weights", default=str(ROOT / "meta/layer_weights.json"),
                help="JSON with per-layer relative slab sizes ('equal' = same slots for every layer)")
ap.add_argument("--page-cache", action="store_true", help="let the OS page cache buffer expert reads")
ap.add_argument("--raw", action="store_true", help="skip the chat template")
ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default=None,
                help="how long the model thinks before answering (default: the chat template's, xhigh)")
ap.add_argument("--hide-reasoning", action="store_true", help="do not show the model's reasoning")
ap.add_argument("--show-markup", action="store_true", help="print the raw token stream incl. <|...|> markers")
args = ap.parse_args()

layer_weights = None
if args.layer_weights != "equal" and Path(args.layer_weights).exists():
    layer_weights = json.loads(Path(args.layer_weights).read_text())["weights"]

mx.set_cache_limit(4 << 30)  # keep MLX's free-buffer pool small; the expert LRU is our cache
model, cache = es.build_streaming_model(
    args.model, ms, cache_gb=args.cache_gb, group_max=args.group_max,
    nocache=not args.page_cache, threads=args.threads, reserve_gb=args.reserve_gb,
    prefetch_k=args.prefetch_k, prefetch_threads=args.pf_threads, layer_weights=layer_weights)
# The tokenizer only knows <|end_of_text|>; the model ends a turn with <|eot|>. Both are in config.json.
cfg = json.loads((Path(args.model) / "config.json").read_text())
eos = cfg.get("eos_token_id") or cfg.get("text_config", {}).get("eos_token_id")
tok = load_tokenizer(Path(args.model), eos_token_ids=[eos] if isinstance(eos, int) else eos)

prompt = args.prompt
if not args.raw and getattr(tok, "chat_template", None):
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                     add_generation_prompt=True, **extra)

# Answer -> stdout; reasoning and statistics -> stderr (dimmed on a terminal), so `> file` keeps just the answer.
parser = None if (args.raw or args.show_markup or not getattr(tok, "chat_template", None)) else TurnParser()
DIM, OFF = ("\033[2m", "\033[0m") if sys.stderr.isatty() else ("", "")
shown = None  # channel currently being written


def show(chunks):
    global shown
    for channel, text in chunks:
        if channel == "reasoning" and args.hide_reasoning:
            continue
        if channel != shown:
            if shown == "reasoning":
                sys.stderr.write(OFF + "\n\n")
            if channel == "reasoning":
                sys.stderr.write(DIM)
            shown = channel
        out = sys.stderr if channel == "reasoning" else sys.stdout
        out.write(text)
        out.flush()


last, after_prefill = None, None
for resp in stream_generate(model, tok, prompt, max_tokens=args.max_tokens):
    if after_prefill is None:  # first token is out: prefill is done
        after_prefill = dict(cache.stats)
    if parser:
        show(parser.feed(resp.text))
    else:
        print(resp.text, end="", flush=True)
    last = resp
if parser:
    show(parser.flush())
    if shown == "reasoning":
        sys.stderr.write(OFF + "\n")
print()
if last:
    print(f"\nprompt: {last.prompt_tokens} tok @ {last.prompt_tps:.2f} tok/s | "
          f"generation: {last.generation_tokens} tok @ {last.generation_tps:.2f} tok/s | "
          f"peak mem {last.peak_memory:.1f} GB", file=sys.stderr)
    d = {k: cache.stats[k] - after_prefill[k] for k in after_prefill}
    n = d["hits"] + d["misses"]
    print(f"decode only: hit rate {d['hits'] / max(n, 1):.1%}, read {d['bytes_read'] / es.GB:.1f} GB "
          f"in {d['read_s']:.1f}s ({d['bytes_read'] / es.GB / max(d['read_s'], 1e-9):.1f} GB/s while reading), "
          f"gpu-sync {d['sync_s']:.1f}s, stack {d['stack_s']:.1f}s, "
          f"{d['misses'] / max(last.generation_tokens - 1, 1) / 62:.1f} expert misses/layer/token", file=sys.stderr)
print("overall:", cache.summary(), file=sys.stderr)
