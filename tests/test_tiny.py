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
from mlx_lm.models import muse_spark as ms  # noqa: E402

LAYERS = 5
THETA = [500000.0, 0, 500000.0, 500000.0, 0]      # 0 = no RoPE (the full-attention layers of the real model)
CFG = dict(
    model_type="muse_spark",
    text_config=dict(
        hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_hidden_layers=LAYERS,
        vocab_size=512, rms_norm_eps=1e-5, post_norm_eps=1e-8, rope_parameters={"rope_theta": 500000.0},
        layer_rope_theta=THETA, sliding_window=8, final_logit_softcapping=20.0, output_multiplier=0.17677669529663687,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        num_local_experts=16, num_experts_per_tok=4, moe_hidden_size=128, moe_intermediate_size=128,
        mlp_layer_types=["sparse"] * LAYERS),
)
# Like the real checkpoint: 3 bits by default, a few modules at other widths.
# Router: a big per-expert bias keeps the chosen experts stable when mlx-lm's bf16 casts add ~0.3% noise
# between two ways of evaluating the same text (one shot vs chunks vs token by token).
ROUTER_STD, BIAS_STD = 0.1, 1.0
MIXED = {
    "model.language_model.layers.1.self_attn.qkvg_proj": dict(group_size=64, bits=4, mode="affine"),
    "model.language_model.layers.2.mlp.pre_expert_proj": dict(group_size=64, bits=5, mode="affine"),
    "model.language_model.layers.3.self_attn.o_proj": dict(group_size=64, bits=8, mode="affine"),
}
CFG["quantization"] = dict(group_size=64, bits=3, mode="affine", **MIXED)


def make_checkpoint(out: Path, dtype=mx.bfloat16, fixed_routing=False):
    """`fixed_routing`: the router ignores the token (zero weights, per-expert bias only), so every token of a layer
    uses the same experts. mlx-lm's bf16 casts make two ways of evaluating the same text (one shot, chunks, token by
    token) differ by ~0.3%, which flips a near-tied top-k choice now and then and changes the logits by tens of
    percent. Tests that compare a cache with a fresh prefill need to be immune to that."""
    mx.random.seed(0)
    model = ms.Model(ms.ModelArgs.from_dict(CFG))
    # Randomize everything so routing/norms/gates are non-degenerate (the experts' own init is fine).
    flat = tree_flatten(model.parameters())
    noise = [(k, v if v.size <= 1 or ".mlp.experts.gate_proj" in k or ".mlp.experts.up_proj" in k
              or ".mlp.experts.down_proj" in k
              else 1 + mx.random.normal(v.shape) * 0.1 if "norm" in k and k.endswith("weight")
              else mx.zeros(v.shape) if fixed_routing and k.endswith("mlp.gate.weight")
              else mx.random.normal(v.shape) * (ROUTER_STD if k.endswith("mlp.gate.weight")
                                                else BIAS_STD if k.endswith("e_score_correction_bias") else 0.05))
             for k, v in flat]
    model.update(tree_unflatten(noise))
    for layer in model.layers:                # the residual gates' coefficients, from random gates
        layer.add_attn_alpha, layer.add_attn_beta = ms.residual_gate_coefficients(mx.random.normal((256,)) * 0.5)
        layer.add_ffn_alpha, layer.add_ffn_beta = ms.residual_gate_coefficients(mx.random.normal((256,)) * 0.5)
    keep_fp32 = model.cast_predicate          # the residual gates' coefficients stay float32
    model.update(tree_unflatten(
        [(k, v.astype(dtype) if keep_fp32(k) else v) for k, v in tree_flatten(model.parameters())]))
    q = CFG["quantization"]
    # (mlx-lm casts the final hidden state to lm_head.weight.dtype, so like the real checkpoint it stays unquantized)
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q["mode"],
                class_predicate=lambda p, m: MIXED.get(p, hasattr(m, "to_quantized") and p != "lm_head"))
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


def check_true(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


def logits(model, toks, cache=None):
    out = model(mx.array(toks)[None], cache=cache)
    mx.eval(out)
    return out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="tiny_ckpt_"))
    ok = True
    try:
        ref = make_checkpoint(tmp)

        def stream(slots, group_max, prefetch_k=0):
            return es.build_streaming_model(tmp, slots_per_layer=slots, group_max=group_max,
                                            prefetch_k=prefetch_k, verbose=False)

        mx.clear_cache()
        m, c = stream(16, 48)
        print(f"all experts would be {LAYERS * 16 * c.store.expert_bytes / 2**20:.1f} MB; "
              f"slabs {c.capacity} experts")

        prompt = np.random.RandomState(1).randint(0, 512, 40).tolist()

        # 1. prefill (many tokens -> stacked path). One group runs the very same call as the stock model (bit
        #    exact). Several groups run it on subsets of the tokens, which takes mlx's UNSORTED gather_qmm kernel
        #    instead of the sorted one (>= 64 entries): the stock SwitchGLU itself differs by ~1e-3 between the
        #    two, and the tiny random router amplifies that through 5 layers. So compare the experts layer by
        #    layer there instead of the logits.
        m, c = stream(16, 48)
        ok &= check("prefill 40 tok, one group", logits(ref, prompt), logits(m, prompt), tol=1e-6)
        for gm in (8, 4, 2):
            m, c = stream(16, gm)
            for l in range(LAYERS):
                h = mx.random.normal((1, 40, 128)).astype(mx.bfloat16)
                idx = mx.random.randint(0, 16, (1, 40, 4)).astype(mx.uint32)
                ok &= check(f"layer {l} experts, 40 tok, group_max={gm}", ref.layers[l].mlp.experts(h, idx),
                            m.layers[l].mlp.experts(h, idx), tol=2e-2)

        # 2. greedy decode with KV cache. slots/layer: 16 = everything fits, 6 = eviction,
        #    4 = exactly top-k, 3 = fewer slots than a token needs (stacked fallback)
        for slots, pf in ((16, 0), (6, 0), (4, 0), (3, 0), (16, 6), (6, 6), (6, 16), (4, 4)):
            m, c = stream(slots, 48, pf)
            rc, sc = ref.make_cache(), m.make_cache()
            lr, ls = logits(ref, prompt, rc), logits(m, prompt, sc)
            ok &= check(f"slots={slots} prefetch={pf}: prefill with cache", lr, ls)
            toks_r, toks_s = [], []
            for _ in range(8):
                tr, ts = int(mx.argmax(lr[0, -1])), int(mx.argmax(ls[0, -1]))
                toks_r.append(tr); toks_s.append(ts)
                lr, ls = logits(ref, [tr], rc), logits(m, [ts], sc)
            same = toks_r == toks_s
            ok &= same
            print(f"  {'PASS' if same else 'FAIL'}  slots={slots} prefetch={pf}: greedy tokens ref={toks_r} stream={toks_s}")
            print("   ", c.summary())
            ok &= c.resident <= c.capacity
            if pf and slots >= 4:
                ok &= c.stats["pf_issued"] > 0
            if (slots, pf) == (4, 4):         # top-4 of 16 experts: guessing would hit 25% of the time
                used = c.stats["pf_used"] / max(c.stats["pf_issued"], 1)
                ok &= check_true(f"the next-layer predictor beats chance (used {used:.0%} of prefetched)", used > 0.4)
            if slots == 16:
                ok &= c.stats["hits"] > 0
            if slots < 4:
                ok &= c.stats["fallback"] > 8  # every decode step must take the stacked path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
