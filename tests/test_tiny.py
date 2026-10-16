"""Streaming loader vs the stock in-memory model on a tiny synthetic checkpoint.

Run: .venv/bin/python -I tests/test_tiny.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import expert_stream as es  # noqa: E402
import muse_spark_1975 as ms  # noqa: E402

CFG = dict(
    model_type="muse_spark", hidden_size=256, num_hidden_layers=5, intermediate_size=512,
    num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=512,
    moe_hidden_size=128, moe_intermediate_size=128, num_local_experts=16,
    num_experts_per_tok=4, sliding_window=8, max_position_embeddings=1024,
    quantization=dict(group_size=64, bits=3, mode="affine"),
)


def make_checkpoint(out: Path):
    mx.random.seed(0)
    model = ms.Model(ms.ModelArgs.from_dict(CFG))
    # Randomize everything so routing/norms/gates are non-degenerate.
    flat = tree_flatten(model.parameters())
    noise = [(k, mx.random.normal(v.shape) * 0.05 if v.size > 1 and "switch_mlp" not in k else v)
             for k, v in flat]
    model.update(tree_unflatten(noise))
    keep_fp32 = model.cast_predicate  # router + residual gates stay float32
    model.update(tree_unflatten(
        [(k, v.astype(mx.bfloat16) if keep_fp32(k) else v)
         for k, v in tree_flatten(model.parameters())]))
    q = CFG["quantization"]
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q["mode"])
    mx.eval(model.parameters())

    params = dict(tree_flatten(model.parameters()))
    names = sorted(params)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    out.mkdir(parents=True, exist_ok=True)
    for fn, ks in shards.items():
        mx.save_safetensors(str(out / fn), {k: params[k] for k in ks})
    (out / "config.json").write_text(json.dumps(CFG))
    return model  # the stock, fully in-memory reference


def check(name, a, b, tol=2e-2):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    d = np.abs(a - b).max()
    ok = d <= tol * max(1.0, np.abs(a).max())
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: max|diff|={d:.3g}")
    return ok


def logits(model, toks, cache=None):
    out = model(mx.array(toks)[None], cache=cache)
    mx.eval(out)
    return out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tiny_ckpt_"))
    ok = True
    try:
        ref = make_checkpoint(tmp)

        def stream(cache_experts, group_max):
            m, c = es.build_streaming_model(tmp, ms, cache_gb=0.0, group_max=group_max,
                                            verbose=False)
            c.capacity = cache_experts
            return m, c

        # lazy build: experts must not be resident
        mx.clear_cache()
        before = mx.get_active_memory()
        m, c = stream(10**6, 48)
        resident_mb = (mx.get_active_memory() - before) / 2**20
        expert_total_mb = (CFG["num_hidden_layers"] * CFG["num_local_experts"]
                           * c.store.expert_bytes) / 2**20
        print(f"resident after build {resident_mb:.1f} MB; all experts would be {expert_total_mb:.1f} MB")

        prompt = np.random.RandomState(1).randint(0, 512, 40).tolist()

        # 1. prefill (many tokens -> sorted path, multi-group path with group_max=4)
        for gm in (48, 4):
            m, c = stream(10**6, gm)
            ok &= check(f"prefill 40 tok, group_max={gm}", logits(ref, prompt), logits(m, prompt))

        # 2. greedy decode with KV cache, tiny cache (forces eviction) vs unlimited
        for cap in (10**6, 6):
            m, c = stream(cap, 48)
            rc, sc = ref.make_cache(), m.make_cache()
            lr, ls = logits(ref, prompt, rc), logits(m, prompt, sc)
            ok &= check(f"cache={cap}: prefill with cache", lr, ls)
            toks_r, toks_s = [], []
            for _ in range(8):
                tr, ts = int(mx.argmax(lr[0, -1])), int(mx.argmax(ls[0, -1]))
                toks_r.append(tr); toks_s.append(ts)
                lr, ls = logits(ref, [tr], rc), logits(m, [ts], sc)
            same = toks_r == toks_s
            ok &= same
            print(f"  {'PASS' if same else 'FAIL'}  cache={cap}: greedy tokens ref={toks_r} stream={toks_s}")
            print("   ", c.summary(), f"| lru size {len(c.lru)}")
            if cap == 6:
                ok &= len(c.lru) <= 6
            else:
                ok &= c.stats["hits"] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
