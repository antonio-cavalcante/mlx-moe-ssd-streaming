"""Train the prefetch predictor (src/expert_stream.py: LearnedPredictor) from recorded traces.

  .venv/bin/python tools/train_predictor.py --traces traces/text traces/chat --out meta/predictor.npz

For each layer l and lookahead d (1 and 2) it fits, by ridge regression, a linear map from layer l's router
input to layer l+d's true router logits, starting from "apply that router to this stream" and moving away only
where the data says so. The last segment of the FIRST trace set is held out: it picks the regularization and
calibrates "score margin -> chance of being in the true top-8". The output also has a copy trained on all
data. Takes ~6 minutes per lookahead for 12k tokens (the Gram matrices run on the GPU). The result depends on
this model's weights; it is not committed (see .gitignore).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
import mlx.core as mx  # noqa: E402

BINS, LAMBDAS = 64, (0.1, 0.3, 1.0, 3.0)    # lambda = factor * mean diagonal of the Gram matrix


def router_params(model_dir, n_layers):
    """Gate weight, score bias and pre-feedforward norm gain of every layer (float32)."""
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    shards, out = {}, ([], [], [])
    for l in range(n_layers):
        p = f"model.language_model.layers.{l}."
        for name, dst in zip(("mlp.gate.weight", "mlp.gate.e_score_correction_bias", "pre_feedforward_layernorm.weight"), out):
            if idx[p + name] not in shards:
                shards[idx[p + name]] = mx.load(str(Path(model_dir) / idx[p + name]))
            dst.append(np.array(shards[idx[p + name]][p + name].astype(mx.float32)))
    return tuple(np.stack(o) for o in out)


class Traces:
    def __init__(self, d, n_layers, dim):
        d = Path(d)
        self.meta = json.loads((d / "meta.json").read_text())
        self.n = sum(m["tokens"] for m in self.meta)
        self.last_start = self.n - self.meta[-1]["tokens"]
        self.x = [np.memmap(d / f"x{l}.f16", dtype=np.float16, mode="r", shape=(self.n, dim)) for l in range(n_layers)]
        self.ids = [np.memmap(d / f"i{l}.u8", dtype=np.uint8, mode="r") .reshape(self.n, -1) for l in range(n_layers)]

    def truth(self, l, sl):
        ids = np.asarray(self.ids[l][sl]).astype(np.int64)
        t = np.zeros((ids.shape[0], 256), dtype=bool)
        np.put_along_axis(t, ids, True, axis=1)
        return t


def sigmoid_scores(z, mult, bias):
    return 1.0 / (1.0 + np.exp(-z * mult)) + bias


def margins(S):
    return S - np.partition(S, -8, axis=1)[:, -8][:, None]


def recall(S, truth):
    top = np.argpartition(-S, 7, axis=1)[:, :8]
    return np.take_along_axis(truth, top, axis=1).mean()


def gram(x, r):
    xm = mx.array(x)
    return np.array(xm.T @ xm), np.array(xm.T @ mx.array(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "meta/predictor.npz"))
    ap.add_argument("--layers-ahead", type=int, choices=[1, 2], default=2, help="train predictors for l+1 (and l+2)")
    cc.add_model_args(ap, ROOT)
    args = ap.parse_args()

    cfg = json.loads((Path(args.model) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    n_layers, dim, mult = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["output_multiplier"]
    W, B, gain = router_params(args.model, n_layers)
    sets = [Traces(d, n_layers, dim) for d in args.traces]
    result = {}
    t0 = time.time()
    for d in range(1, args.layers_ahead + 1):
        Vs, edges, probs = [], [], []
        for l in range(n_layers - d):
            num, den = gain[l + d], gain[l]
            ok = np.abs(den) > 1e-6
            V0 = np.vstack([(np.where(ok, num / np.where(ok, den, 1.0), 0.0)[:, None] * W[l + d].T),
                            np.zeros((1, W.shape[1]), np.float32)]).astype(np.float32)

            def block(ts, sl):
                x = np.hstack([np.asarray(ts.x[l][sl]).astype(np.float32), np.ones((sl.stop - sl.start, 1), np.float32)])
                z = np.asarray(ts.x[l + d][sl]).astype(np.float32) @ W[l + d].T      # the true next logits
                return x, z, z - x @ V0

            first = sets[0]
            fit_sl = slice(0, first.last_start)
            val_sl = slice(first.last_start, first.n)
            xa, za, ra = block(first, fit_sl)
            xv, zv, rv = block(first, val_sl)
            Ga, Ra = gram(xa, ra)
            Gv, Rv = gram(xv, rv)
            scale = np.trace(Ga) / Ga.shape[0]
            truth_v = first.truth(l + d, val_sl)
            S0 = sigmoid_scores(xv @ V0, mult, B[l + d])
            best = (recall(S0, truth_v), None)
            for c in LAMBDAS:
                V = V0 + np.linalg.solve(Ga + c * scale * np.eye(Ga.shape[0], dtype=np.float32), Ra)
                r = recall(sigmoid_scores(xv @ V, mult, B[l + d]), truth_v)
                if r > best[0]:
                    best = (r, c)
            r_best, c_best = best
            V = V0 if c_best is None else V0 + np.linalg.solve(Ga + c_best * scale * np.eye(Ga.shape[0], dtype=np.float32), Ra)
            Sv = sigmoid_scores(xv @ V, mult, B[l + d])                   # held-out scores -> calibration
            m = margins(Sv).ravel()
            e = np.quantile(m, np.linspace(0, 1, BINS + 1)[1:-1])
            bin_ = np.searchsorted(e, m)
            p = np.maximum.accumulate(np.bincount(bin_, weights=truth_v.ravel(), minlength=BINS)
                                      / np.maximum(np.bincount(bin_, minlength=BINS), 1))
            if c_best is not None:                                        # refit on everything
                G, R = Ga + Gv, Ra + Rv
                for ts in sets[1:]:
                    xs, _, rs = block(ts, slice(0, ts.n))
                    gs, rs_ = gram(xs, rs)
                    G, R = G + gs, R + rs_
                V = V0 + np.linalg.solve(G + c_best * scale * np.eye(G.shape[0], dtype=np.float32), R)
            Vs.append(V.astype(np.float16))
            edges.append(e)
            probs.append(p)
            print(f"d={d} layer {l:2d}: held-out recall of the true top-8 {recall(S0, truth_v):.2f} -> {r_best:.2f} "
                  f"(lambda {c_best}) [{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
        suffix = "" if d == 1 else "2"
        result.update({f"V{suffix}": np.stack(Vs), f"edges{suffix}": np.stack(edges), f"p{suffix}": np.stack(probs)})
    np.savez(args.out, **result)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
