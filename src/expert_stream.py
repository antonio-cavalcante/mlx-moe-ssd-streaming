"""Stream routed MoE experts from SSD instead of holding them in RAM.

Only the dense weights (attention, embeddings, routers, pre/post expert projections,
norms) are loaded into memory. For every MoE layer the router runs first, then just the
selected experts are read from the original safetensors shards (by byte offset) into an
LRU cache and the usual gate/up/down SwiGLU runs on them.

Expert tensors are stacked in the checkpoint with the expert index as the leading axis,
so expert `e` of a tensor with shape [E, a, b] is the contiguous byte range
[base + e*stride, base + (e+1)*stride) of the shard file. Reads go straight into the
final MLX buffers (writable memoryview), split into chunks that run in parallel.

Optional prefetch (enable_prefetch): while layer l's expert math runs on the GPU, the
experts layer l+1 will probably need (layer l+1's router applied to layer l's residual
stream) are already being read in the background.
"""

import fcntl
import glob
import json
import os
import re
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU

PROJS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "scales", "biases")
PIECES = tuple((p, k) for p in PROJS for k in KINDS)  # 9 tensors per expert
_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_MX_DTYPE = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
GB = 1 << 30


def read_safetensors_header(path):
    """{name: (abs_offset, nbytes, dtype, shape)} without reading any tensor data."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 100 << 20:
            raise ValueError(f"{path}: implausible safetensors header size {n}")
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, v in header.items():
        if name == "__metadata__":
            continue
        s, e = v["data_offsets"]
        if base + e > size:
            raise ValueError(f"{path}: tensor {name} extends past end of file")
        out[name] = (base + s, e - s, v["dtype"], tuple(v["shape"]))
    return out


class _Pending:
    """An expert whose 9 tensors are being read into `raws` by worker threads."""

    __slots__ = ("raws", "futures")

    def __init__(self):
        self.raws, self.futures = [], []


class ExpertStore:
    """Reads single experts (9 tensors each) out of the safetensors shards."""

    def __init__(self, model_dir, num_layers, num_experts, nocache=True, threads=24,
                 chunk_mb=1.0, prefetch_threads=16):
        self.num_layers, self.num_experts = num_layers, num_experts
        # chunk_mb: each tensor read is split into pieces of this size so more reads are
        # in flight (0 = one read per tensor).
        self.chunk = int(chunk_mb * (1 << 20)) if chunk_mb else 0
        self.loc = {}  # (layer, proj, kind) -> (fd, base, stride, dtype, shape_tail)
        self.fds = []
        for path in sorted(glob.glob(str(Path(model_dir) / "model*.safetensors"))):
            fd = os.open(path, os.O_RDONLY)
            if nocache:
                # Our own LRU is the cache; don't let the OS page cache double-buffer.
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self.fds.append(fd)
            for name, (off, nbytes, dtype, shape) in read_safetensors_header(path).items():
                m = _EXPERT_RE.match(name)
                if not m:
                    continue
                if shape[0] != num_experts or nbytes % num_experts:
                    raise ValueError(f"unexpected expert tensor shape {name}: {shape}")
                layer, proj, kind = int(m.group(1)), m.group(2), m.group(3)
                self.loc[(layer, proj, kind)] = (
                    fd, off, nbytes // num_experts, dtype, shape[1:])
        missing = [(l, p, k) for l in range(num_layers) for p, k in PIECES
                   if (l, p, k) not in self.loc]
        if missing:
            raise FileNotFoundError(f"{len(missing)} expert tensors missing, e.g. {missing[0]}")
        self.expert_bytes = sum(self.loc[(0, p, k)][2] for p, k in PIECES)
        self.free = {}  # nbytes -> recycled uint8 buffers (fully overwritten by each read)
        self.pool = ThreadPoolExecutor(threads)  # demand reads
        self.pf_pool = ThreadPoolExecutor(prefetch_threads)  # speculative reads

    def _fill(self, fd, off, view):
        """preadv until `view` (a writable uint8 buffer) is full."""
        done = 0
        while done < len(view):
            got = os.preadv(fd, [memoryview(view)[done:]], off + done)
            if got <= 0:
                raise IOError(f"short read at offset {off + done}")
            done += got

    def start_read(self, layer, experts, pool=None, chunk=None):
        """Allocate buffers and queue the reads without waiting: {expert: _Pending}."""
        pool = pool or self.pool
        chunk = self.chunk if chunk is None else chunk
        raws, fresh = {}, []
        for e in experts:
            for i in range(len(PIECES)):
                n = self.loc[(layer,) + PIECES[i]][2]
                spare = self.free.get(n)
                if spare:
                    raws[(e, i)] = spare.pop()
                else:
                    raws[(e, i)] = mx.zeros((n,), dtype=mx.uint8)
                    fresh.append(raws[(e, i)])
        if fresh:
            mx.eval(fresh)
        out = {}
        for e in experts:
            p = out[e] = _Pending()
            for i in range(len(PIECES)):
                raw = raws[(e, i)]
                fd, base, stride, _, _ = self.loc[(layer,) + PIECES[i]]
                view = np.frombuffer(memoryview(raw), dtype=np.uint8)
                step = chunk or stride
                p.raws.append(raw)
                p.futures.extend(pool.submit(self._fill, fd, base + e * stride + o, view[o:o + step])
                                 for o in range(0, stride, step))
        return out

    def recycle(self, raws):
        for r in raws:
            self.free.setdefault(r.nbytes, []).append(r)

    @staticmethod
    def done(p):
        return all(f.done() for f in p.futures)

    def finish(self, layer, p):
        """Wait for a pending expert; returns its 9 mx.arrays in PIECES order."""
        for f in p.futures:
            f.result()
        arrs = []
        for i, raw in enumerate(p.raws):
            _, _, _, dtype, tail = self.loc[(layer,) + PIECES[i]]
            arrs.append(raw.view(_MX_DTYPE[dtype]).reshape(tail))
        return tuple(arrs)

    def read_many(self, layer, experts):
        """Blocking parallel read; returns {expert: (9 mx.arrays, raw buffers)}."""
        pend = self.start_read(layer, experts)
        return {e: (self.finish(layer, p), p.raws) for e, p in pend.items()}


class ExpertCache:
    """LRU of (layer, expert) -> 9 mx.arrays, bounded by a byte budget."""

    def __init__(self, store, budget_bytes):
        self.store = store
        self.capacity = max(0, int(budget_bytes // store.expert_bytes))
        self.lru = OrderedDict()  # (layer, expert) -> (arrays, raw buffers)
        self._grave = []  # raw buffers of evicted entries, reusable after the next GPU sync
        self._dropped = []  # unclaimed speculative reads, reusable once they finish
        self.inflight = {}  # (layer, expert) -> _Pending, speculative reads in progress
        self.resid = {}  # layer -> last-token residual stream (set by prefetch hooks)
        self.prefetch_k = 0
        self.stats = dict(hits=0, misses=0, bytes_read=0, read_s=0.0, stack_s=0.0,
                          sync_s=0.0, calls=0, pf_issued=0, pf_used=0, pf_wasted=0,
                          pf_wait_s=0.0, pf_submit_s=0.0, predict_s=0.0)

    def _admit(self, key, arrs, raws, oldest=False):
        self.lru[key] = (arrs, raws)
        if oldest:  # unclaimed speculation: first to go, never displaces a hot expert
            self.lru.move_to_end(key, last=False)
        while len(self.lru) > self.capacity:
            self._grave.extend(self.lru.popitem(last=False)[1][1])

    def _reclaim(self):
        """Called right after a GPU sync: nothing pending can still read evicted buffers."""
        if self._grave:
            self.store.recycle(self._grave)
            self._grave = []
        if self._dropped:
            keep = []
            for p in self._dropped:
                if self.store.done(p):
                    self.store.recycle(p.raws)
                else:
                    keep.append(p)
            self._dropped = keep

    def get(self, layer, experts, admit=True):
        """Return {expert: arrays}. Cached and prefetched experts are free; the rest
        are read in parallel from disk."""
        self._reclaim()
        got, missing, claimed = {}, [], []
        for e in experts:
            key = (layer, int(e))
            if key in self.lru:
                self.lru.move_to_end(key)
                got[int(e)] = self.lru[key][0]
            elif key in self.inflight:
                claimed.append(int(e))
            else:
                missing.append(int(e))
        self.stats["hits"] += len(got)
        if claimed:
            t = time.perf_counter()
            for e in claimed:
                p = self.inflight.pop((layer, e))
                arrs = self.store.finish(layer, p)
                got[e] = arrs
                if admit:
                    self._admit((layer, e), arrs, p.raws)
            self.stats["pf_wait_s"] += time.perf_counter() - t
            self.stats["pf_used"] += len(claimed)
            self.stats["hits"] += len(claimed)
        self.stats["misses"] += len(missing)
        if missing:
            t = time.perf_counter()
            loaded = self.store.read_many(layer, missing)  # {e: (arrays, raws)}
            self.stats["read_s"] += time.perf_counter() - t
            self.stats["bytes_read"] += len(missing) * self.store.expert_bytes
            got.update({e: arrs for e, (arrs, _) in loaded.items()})
            if admit:
                for e, (arrs, raws) in loaded.items():
                    self._admit((layer, e), arrs, raws)
        if self.inflight:
            self._sweep(layer)
        return got

    def _sweep(self, layer):
        """Speculative reads for this layer nobody claimed: resolve now so they never
        pile up. Finished ones are kept only if there is spare room (and evicted first);
        unfinished ones are dropped (their buffers live until the read ends)."""
        for key in [k for k in self.inflight if k[0] == layer]:
            p = self.inflight.pop(key)
            self.stats["pf_wasted"] += 1
            if len(self.lru) < self.capacity and self.store.done(p):
                self._admit(key, self.store.finish(layer, p), p.raws, oldest=True)
            else:
                self._dropped.append(p)  # buffers return to the pool once the read ends

    def prefetch(self, layer, experts):
        """Start background reads for experts that are neither cached nor in flight."""
        new = [e for e in dict.fromkeys(int(x) for x in experts)
               if (layer, e) not in self.lru and (layer, e) not in self.inflight]
        if not new:
            return
        for e, p in self.store.start_read(layer, new, pool=self.store.pf_pool, chunk=0).items():
            self.inflight[(layer, e)] = p
        self.stats["pf_issued"] += len(new)
        self.stats["bytes_read"] += len(new) * self.store.expert_bytes

    def summary(self):
        s = self.stats
        total = s["hits"] + s["misses"]
        out = (f"cache {len(self.lru)}/{self.capacity} experts, hit rate "
               f"{s['hits'] / max(total, 1):.1%} ({s['hits']}/{total}), read "
               f"{s['bytes_read'] / GB:.1f} GB (demand wait {s['read_s']:.1f}s), "
               f"stack {s['stack_s']:.1f}s, gpu-sync {s['sync_s']:.1f}s")
        if s["pf_issued"]:
            out += (f" | prefetch issued {s['pf_issued']}, used {s['pf_used']} "
                    f"({s['pf_used'] / s['pf_issued']:.0%}), unclaimed {s['pf_wasted']}, "
                    f"waited {s['pf_wait_s']:.1f}s, submit {s['pf_submit_s']:.1f}s, "
                    f"predict-graph {s['predict_s']:.1f}s")
        return out


class StreamingSwitchGLU(nn.Module):
    """Drop-in for SwitchGLU(x, indices) that pulls the needed experts from disk."""

    def __init__(self, layer_idx, cache, template, group_max=48, admit_max_tokens=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache = cache
        self.group_max = group_max
        self.admit_max_tokens = admit_max_tokens  # long prefills read-through, not admit
        # Keep the template out of the module tree (no parameters, not quantized/saved).
        object.__setattr__(self, "_glu", template)
        # Tiny stand-ins, put back after every call so the template never pins the
        # last stacked expert weights (that was ~11 GB at decode, 30+ GB after prefill).
        object.__setattr__(self, "_placeholders",
                           [getattr(getattr(template, p), k) for p, k in PIECES])
        object.__setattr__(self, "next_layer", None)  # set by enable_prefetch

    def _run(self, xs, local_inds, group, admit):
        arrs = self.cache.get(self.layer_idx, group, admit)
        t = time.perf_counter()
        stacked = [mx.stack([arrs[int(e)][i] for e in group]) for i in range(len(PIECES))]
        self.cache.stats["stack_s"] += time.perf_counter() - t
        for pi, (proj, kind) in enumerate(PIECES):
            setattr(getattr(self._glu, proj), kind, stacked[pi])
        y = self._glu(xs, mx.array(local_inds.astype(np.uint32)))
        # The lazy graph keeps `stacked` alive until evaluated; the module must not.
        for (proj, kind), arr in zip(PIECES, self._placeholders):
            setattr(getattr(self._glu, proj), kind, arr)
        return y

    def _predict(self):
        """Lazy top-k expert ids for the NEXT layer from this layer's residual stream."""
        nxt, r = self.next_layer, self.cache.resid.get(self.layer_idx)
        if nxt is None or r is None:
            return None
        g = nxt.mlp.gate
        sc = nxt.pre_feedforward_layernorm.plain(r).astype(mx.float32) @ g.weight.T
        sc = mx.sigmoid(sc * g.output_multiplier) + g.e_score_correction_bias
        k = min(self.cache.prefetch_k, sc.shape[-1])
        return mx.argpartition(-sc, kth=k - 1, axis=-1)[..., :k]

    def __call__(self, x, indices):
        n_tok = int(np.prod(indices.shape[:-1]))
        speculate = self.cache.prefetch_k > 0 and n_tok <= self.admit_max_tokens
        t = time.perf_counter()
        pred = self._predict() if speculate else None
        self.cache.stats["predict_s"] += time.perf_counter() - t
        t = time.perf_counter()
        # Also flushes the previous layers' pending GPU work.
        mx.eval(indices) if pred is None else mx.eval(indices, pred)
        self.cache.stats["sync_s"] += time.perf_counter() - t
        self.cache.stats["calls"] += 1
        idx = np.array(indices)
        lead, k = idx.shape[:-1], idx.shape[-1]
        xf = x.reshape(-1, x.shape[-1])
        idxf = idx.reshape(-1, k)
        admit = n_tok <= self.admit_max_tokens
        uniq = np.unique(idxf)
        groups = [uniq[i:i + self.group_max] for i in range(0, len(uniq), self.group_max)]

        if len(groups) == 1:
            y = self._run(xf, np.searchsorted(groups[0], idxf), groups[0], admit)
            if speculate:
                if pred is not None:
                    t = time.perf_counter()
                    self.cache.prefetch(self.layer_idx + 1, np.array(pred).reshape(-1))
                    self.cache.stats["pf_submit_s"] += time.perf_counter() - t
                mx.async_eval(y)  # GPU starts the expert math now, overlapping the reads
            return y.reshape(*lead, k, -1)

        # Many distinct experts (prefill): handle one group of experts at a time and
        # scatter the (token, slot) rows back into place.
        pieces, orders = [], []
        for g in groups:
            t_sel, j_sel = np.nonzero(np.isin(idxf, g))
            local = np.searchsorted(g, idxf[t_sel, j_sel])[:, None]
            y = self._run(xf[mx.array(t_sel.astype(np.uint32))], local, g, admit)[:, 0]
            mx.eval(y)  # release this group's stacked weights before the next one
            pieces.append(y)
            orders.append(t_sel * k + j_sel)
        inv = np.argsort(np.concatenate(orders))
        y = mx.concatenate(pieces)[mx.array(inv.astype(np.uint32))]
        return y.reshape(*lead, k, -1)


def _hook_norm(norm, layer_idx, cache):
    """Record each layer's last-token residual (the input of pre_feedforward_layernorm)."""
    cls = type(norm)

    class Hooked(cls):
        def __call__(self, x):
            cache.resid[layer_idx] = x[:, -1:] if x.shape[1] <= 8 else None
            return super().__call__(x)

        def plain(self, x):  # the original norm without recording (used for predictions)
            return cls.__call__(self, x)

    norm.__class__ = Hooked


def enable_prefetch(model, cache, k=12):
    """Predict layer l+1's experts during layer l and read them in the background."""
    cache.prefetch_k = k
    layers = model.layers
    for i, layer in enumerate(layers):
        _hook_norm(layer.pre_feedforward_layernorm, i, cache)
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        object.__setattr__(layer.mlp.switch_mlp, "next_layer", nxt)


def build_streaming_model(model_dir, model_module, cache_gb=None, group_max=48,
                          nocache=True, threads=24, reserve_gb=10.0, prefetch_k=0,
                          prefetch_threads=16, verbose=True):
    """Build `model_module.Model` with dense weights resident and experts streamed.

    cache_gb=None sizes the expert cache from the GPU's recommended working set.
    prefetch_k>0 turns on speculative next-layer reads (top-k predicted experts).
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    quant = config["quantization"]
    args = model_module.ModelArgs.from_dict(config)
    model = model_module.Model(args)  # expert tensors stay lazy and are never evaluated

    # Dense weights: mx.load is lazy, so only what we evaluate below is read.
    weights = {}
    for path in sorted(glob.glob(str(model_dir / "model*.safetensors"))):
        for k, v in mx.load(path).items():
            if ".mlp.switch_mlp." not in k:
                weights[k] = v
    weights = model.sanitize(weights)

    store = ExpertStore(model_dir, args.num_hidden_layers, args.num_local_experts,
                        nocache=nocache, threads=threads, prefetch_threads=prefetch_threads)

    resident = sum(v.nbytes for v in weights.values())
    if cache_gb is None:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = info["max_recommended_working_set_size"] / GB
        cache_gb = max(1.0, limit - resident / GB - reserve_gb)
    cache = ExpertCache(store, cache_gb * GB)

    for i, layer in enumerate(model.layers):
        old = layer.mlp.switch_mlp
        template = SwitchGLU(old.gate_proj.input_dims, old.gate_proj.output_dims, 1,
                             activation=old.activation)
        nn.quantize(template, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant.get("mode", "affine"))
        layer.mlp.switch_mlp = StreamingSwitchGLU(i, cache, template, group_max=group_max)

    nn.quantize(
        model, group_size=quant["group_size"], bits=quant["bits"],
        mode=quant.get("mode", "affine"),
        class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in weights,
    )
    model.eval()
    model.load_weights(list(weights.items()), strict=False)

    have = {k for k, _ in tree_flatten(model.parameters())}
    missing = sorted(k for k in have - set(weights) if ".switch_mlp." not in k)
    extra = sorted(set(weights) - have)
    if missing or extra:
        raise ValueError(f"weight mismatch; missing {missing[:5]} extra {extra[:5]}")
    mx.eval(model.parameters())
    if prefetch_k > 0:
        enable_prefetch(model, cache, prefetch_k)
    if verbose:
        print(f"resident {resident / GB:.1f} GB, expert cache {cache_gb:.1f} GB "
              f"({cache.capacity} of {args.num_hidden_layers * args.num_local_experts} experts, "
              f"{store.expert_bytes / 2**20:.1f} MB each)"
              + (f", prefetch top-{prefetch_k}" if prefetch_k else ""), flush=True)
    return model, cache
