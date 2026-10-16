"""Arguments, model loading and output helpers shared by run_stream.py and chat.py."""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import load as load_tokenizer

import expert_stream as es
import muse_spark_1975 as ms


def add_model_args(ap, root):
    ap.add_argument("--model", default=str(root / "models/Muse-Spark-Open-816B-A42B-3bit"))
    ap.add_argument("--cache-gb", type=float, default=None,
                    help="expert slab size; default = GPU recommended working set minus resident minus reserve")
    ap.add_argument("--reserve-gb", type=float, default=10.0, help="headroom for KV cache/activations")
    ap.add_argument("--group-max", type=int, default=48, help="max experts stacked at once during prefill")
    ap.add_argument("--threads", type=int, default=24, help="parallel SSD reads")
    ap.add_argument("--prefetch-k", type=int, default=4,
                    help="speculatively read the top-k predicted experts of the next layer (0 = off)")
    ap.add_argument("--pf-threads", type=int, default=16, help="parallel SSD reads for prefetch")
    ap.add_argument("--layer-weights", default=str(root / "meta/layer_weights.json"),
                    help="JSON with per-layer relative slab sizes ('equal' = same slots for every layer)")
    ap.add_argument("--page-cache", action="store_true", help="let the OS page cache buffer expert reads")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default="medium",
                    help="how long the model thinks before answering (the chat template's own default is xhigh)")
    ap.add_argument("--hide-reasoning", action="store_true", help="do not show the model's reasoning")


def generation_defaults(model_dir):
    """The model's recommended sampling settings (generation_config.json)."""
    try:
        return json.loads((Path(model_dir) / "generation_config.json").read_text())
    except OSError:
        return {}


def add_sampling_args(ap, model_dir, greedy_default):
    gc = generation_defaults(model_dir)
    ap.add_argument("--temp", type=float, default=0.0 if greedy_default else gc.get("temperature", 1.0),
                    help="sampling temperature (0 = greedy)")
    ap.add_argument("--top-k", type=int, default=gc.get("top_k", 0), help="top-k sampling (0 = off)")
    ap.add_argument("--top-p", type=float, default=gc.get("top_p", 1.0), help="top-p sampling (1 = off)")


def make_sampler_from(args):
    return make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)


def load_all(args, root):
    """Streaming model + expert cache + tokenizer (which knows every stop token in config.json)."""
    layer_weights = None
    if args.layer_weights != "equal" and Path(args.layer_weights).exists():
        layer_weights = json.loads(Path(args.layer_weights).read_text())["weights"]
    mx.set_cache_limit(4 << 30)  # keep MLX's free-buffer pool small; the expert slabs are our cache
    model, cache = es.build_streaming_model(
        args.model, ms, cache_gb=args.cache_gb, group_max=args.group_max,
        nocache=not args.page_cache, threads=args.threads, reserve_gb=args.reserve_gb,
        prefetch_k=args.prefetch_k, prefetch_threads=args.pf_threads, layer_weights=layer_weights)
    # The tokenizer alone only knows <|end_of_text|>; the model ends a turn with <|eot|>. Both are in config.json.
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    eos = cfg.get("eos_token_id") or cfg.get("text_config", {}).get("eos_token_id")
    tok = load_tokenizer(Path(args.model), eos_token_ids=[eos] if isinstance(eos, int) else eos)
    return model, cache, tok


class Printer:
    """Answer -> stdout; reasoning -> stderr (dimmed on a terminal), so `> file` keeps just the answer."""

    def __init__(self, hide_reasoning=False):
        self.hide = hide_reasoning
        self.dim, self.off = ("\033[2m", "\033[0m") if sys.stderr.isatty() else ("", "")
        self.shown = None  # channel currently being written

    def write(self, chunks):
        for channel, text in chunks:
            if channel == "reasoning" and self.hide:
                continue
            if channel != self.shown:
                if self.shown == "reasoning":
                    sys.stderr.write(self.off + "\n\n")
                if channel == "reasoning":
                    sys.stderr.write(self.dim)
                self.shown = channel
            out = sys.stderr if channel == "reasoning" else sys.stdout
            out.write(text)
            out.flush()

    def close(self):
        """End the output: leave dim mode, end the line, ready for the next message."""
        if self.shown == "reasoning":
            sys.stderr.write(self.off + "\n")
            sys.stderr.flush()
        print(flush=True)
        self.shown = None
