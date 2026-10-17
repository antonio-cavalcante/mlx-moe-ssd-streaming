"""Record what the router sees and picks, for training the prefetch predictor (tools/train_predictor.py).

  # plain text, prefilled in chunks; every token is recorded (fast: ~40-60 tokens/s)
  .venv/bin/python tools/record_traces.py prefill traces/text  FILE1 FILE2 ...  [--tokens 2500]
  # chat prompts (one per line), generated at the default effort; only 1-token decode steps are recorded
  .venv/bin/python tools/record_traces.py decode  traces/chat  PROMPTS.txt  [--max-tokens 380]

Per layer l, OUT/x{l}.f16 holds the MoE input of every recorded token (N x hidden, float16: 16 KB per token and
layer, ~1 MB per token in all) and OUT/i{l}.u8 the experts it picked; OUT/meta.json lists the segments (one per
file / prompt). Plan on 10-15 GB of disk for ~12k tokens; the files can be deleted after training.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prefill", "decode"])
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+", help="prefill: text files; decode: a file with one prompt per line")
    ap.add_argument("--tokens", type=int, default=2500, help="prefill: tokens taken from each file")
    ap.add_argument("--max-tokens", type=int, default=380, help="decode: tokens generated per prompt")
    cc.add_model_args(ap, ROOT)
    args = ap.parse_args()
    args.prefetch_k = 0
    args.predictor = "none"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("x*.f16")):
        sys.exit(f"{out} already holds traces; use a new directory")

    model, cache, tok = cc.load_all(args, ROOT)
    n_layers = len(model.layers)
    files, state = {}, dict(x=None, n=0)
    wanted = (lambda T: T > 1) if args.mode == "prefill" else (lambda T: T == 1)

    def handle(name):
        if name not in files:
            files[name] = open(out / name, "ab")
        return files[name]

    for li, layer in enumerate(model.layers):
        moe, glu = layer.mlp, layer.mlp.experts

        def make(li, moe_cls, glu_cls):
            class RecMoE(moe_cls):
                def __call__(self, x):
                    state["x"] = (np.asarray(x.astype(mx.float16)).reshape(-1, x.shape[-1])
                                  if wanted(x.shape[1]) else None)
                    return super().__call__(x)

            class RecGLU(glu_cls):
                def __call__(self, x, indices):
                    if state["x"] is not None:
                        picked = np.asarray(indices).reshape(-1, indices.shape[-1]).astype(np.uint8)
                        handle(f"x{li}.f16").write(state["x"].tobytes())
                        handle(f"i{li}.u8").write(picked.tobytes())
                        state["n"] += picked.shape[0] if li == n_layers - 1 else 0
                        state["x"] = None
                    return super().__call__(x, indices)
            return RecMoE, RecGLU

        moe.__class__, glu.__class__ = make(li, type(moe), type(glu))

    meta = []
    if args.mode == "prefill":
        for path in args.inputs:
            ids = tok.encode(Path(path).read_text())[:args.tokens]
            c, n0 = make_prompt_cache(model), state["n"]
            for i in range(0, len(ids), 2048):
                model(mx.array(ids[i:i + 2048])[None], cache=c)
                mx.eval([x.state for x in c])
            meta.append(dict(name=Path(path).name, tokens=state["n"] - n0))
            print(meta[-1], file=sys.stderr, flush=True)
    else:
        for prompt in Path(args.inputs[0]).read_text().splitlines():
            if not prompt.strip():
                continue
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           reasoning_effort=args.reasoning_effort)
            n0 = state["n"]
            for _ in stream_generate(model, tok, text, max_tokens=args.max_tokens):
                pass
            meta.append(dict(name=prompt[:40], tokens=state["n"] - n0))
            print(meta[-1], file=sys.stderr, flush=True)
    for f in files.values():
        f.close()
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"recorded {state['n']} tokens in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
