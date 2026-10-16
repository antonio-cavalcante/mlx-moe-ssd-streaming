"""Generate with Muse-Spark-Open-816B-A42B-3bit, streaming experts from SSD.

  .venv/bin/python run_stream.py --prompt "Explain mixture of experts in 3 sentences." --max-tokens 64
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
import expert_stream as es  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from turns import TurnParser  # noqa: E402

ap = argparse.ArgumentParser()
cc.add_model_args(ap, ROOT)
cc.add_sampling_args(ap, ROOT / "models/Muse-Spark-Open-816B-A42B-3bit", greedy_default=True)
ap.add_argument("--prompt", default="Explain mixture-of-experts models in three sentences.")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--raw", action="store_true", help="skip the chat template")
ap.add_argument("--show-markup", action="store_true", help="print the raw token stream incl. <|...|> markers")
args = ap.parse_args()

model, cache, tok = cc.load_all(args, ROOT)

prompt = args.prompt
templated = not args.raw and getattr(tok, "chat_template", None)
if templated:
    prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                     reasoning_effort=args.reasoning_effort)

parser = TurnParser() if templated and not args.show_markup else None
printer = cc.Printer(args.hide_reasoning)
last, after_prefill = None, None
for resp in stream_generate(model, tok, prompt, max_tokens=args.max_tokens,
                            sampler=cc.make_sampler_from(args)):
    if after_prefill is None:  # first token is out: prefill is done
        after_prefill = dict(cache.stats)
    if parser:
        printer.write(parser.feed(resp.text))
    else:
        print(resp.text, end="", flush=True)
    last = resp
if parser:
    printer.write(parser.flush())
printer.close()
if last:
    print(f"\nprompt: {last.prompt_tokens} tok @ {last.prompt_tps:.2f} tok/s | "
          f"generation: {last.generation_tokens} tok @ {last.generation_tps:.2f} tok/s | "
          f"peak mem {mx.get_peak_memory() / es.GB:.1f} GB", file=sys.stderr)
    d = {k: cache.stats[k] - after_prefill[k] for k in after_prefill}
    n = d["hits"] + d["misses"]
    print(f"decode only: hit rate {d['hits'] / max(n, 1):.1%}, read {d['bytes_read'] / es.GB:.1f} GB "
          f"in {d['read_s']:.1f}s ({d['bytes_read'] / es.GB / max(d['read_s'], 1e-9):.1f} GB/s while reading), "
          f"gpu-sync {d['sync_s']:.1f}s, stack {d['stack_s']:.1f}s, "
          f"{d['misses'] / max(last.generation_tokens - 1, 1) / 62:.1f} expert misses/layer/token", file=sys.stderr)
print("overall:", cache.summary(), file=sys.stderr)
