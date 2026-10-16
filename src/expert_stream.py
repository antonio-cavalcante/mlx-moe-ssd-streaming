"""Stream routed MoE experts from SSD instead of holding them in RAM.

Only the dense weights (attention, embeddings, routers, pre/post expert projections,
norms) are loaded into memory. For every MoE layer the router runs first, then just the
selected experts are read from the original safetensors shards (by byte offset) into an
LRU cache and the usual gate/up/down SwiGLU runs on them.

Expert tensors are stacked in the checkpoint with the expert index as the leading axis,
so expert `e` of a tensor with shape [E, a, b] is the contiguous byte range
[base + e*stride, base + (e+1)*stride) of the shard file.
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
_NP_DTYPE = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}
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


class ExpertStore:
    """Reads single experts (9 tensors each) out of the safetensors shards."""

    def __init__(self, model_dir, num_layers, num_experts, nocache=True, threads=24):
        self.num_layers, self.num_experts = num_layers, num_experts
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
        self.pool = ThreadPoolExecutor(threads)

    def _read(self, layer, expert, piece):
        fd, base, stride, dtype, tail = self.loc[(layer,) + piece]
        buf = np.empty(stride, dtype=np.uint8)
        got = os.preadv(fd, [memoryview(buf)], base + expert * stride)
        if got != stride:
            raise IOError(f"short read ({got}/{stride}) layer {layer} expert {expert} {piece}")
        return buf

    def read_many(self, layer, experts):
        """Parallel read; returns {expert: tuple of 9 mx.arrays in PIECES order}."""
        jobs = [(e, i) for e in experts for i in range(len(PIECES))]
        bufs = list(self.pool.map(lambda j: self._read(layer, j[0], PIECES[j[1]]), jobs))
        out = {}
        for (e, i), buf in zip(jobs, bufs):
            _, _, _, dtype, tail = self.loc[(layer,) + PIECES[i]]
            arr = mx.array(buf.view(_NP_DTYPE[dtype]).reshape(tail))
            if dtype == "BF16":
                arr = arr.view(mx.bfloat16)
            out.setdefault(e, []).append(arr)
        return {e: tuple(v) for e, v in out.items()}


class ExpertCache:
    """LRU of (layer, expert) -> 9 mx.arrays, bounded by a byte budget."""

    def __init__(self, store, budget_bytes):
        self.store = store
        self.capacity = max(0, int(budget_bytes // store.expert_bytes))
        self.lru = OrderedDict()
        self.stats = dict(hits=0, misses=0, bytes_read=0, read_s=0.0, stack_s=0.0,
                          sync_s=0.0, calls=0)

    def get(self, layer, experts, admit=True):
        """Return {expert: arrays}. Misses are read in parallel from disk."""
        got, missing = {}, []
        for e in experts:
            key = (layer, int(e))
            if key in self.lru:
                self.lru.move_to_end(key)
                got[int(e)] = self.lru[key]
            else:
                missing.append(int(e))
        self.stats["hits"] += len(got)
        self.stats["misses"] += len(missing)
        if missing:
            t = time.perf_counter()
            loaded = self.store.read_many(layer, missing)
            self.stats["read_s"] += time.perf_counter() - t
            self.stats["bytes_read"] += len(missing) * self.store.expert_bytes
            got.update(loaded)
            if admit:
                for e, arrs in loaded.items():
                    self.lru[(layer, e)] = arrs
                while len(self.lru) > self.capacity:
                    self.lru.popitem(last=False)
        return got

    def summary(self):
        s = self.stats
        total = s["hits"] + s["misses"]
        return (f"cache {len(self.lru)}/{self.capacity} experts, hit rate "
                f"{s['hits'] / max(total, 1):.1%} ({s['hits']}/{total}), read "
                f"{s['bytes_read'] / GB:.1f} GB in {s['read_s']:.1f}s, stack {s['stack_s']:.1f}s, "
                f"gpu-sync {s['sync_s']:.1f}s")


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

    def __call__(self, x, indices):
        t = time.perf_counter()
        mx.eval(indices)  # also flushes the previous layers' pending GPU work
        self.cache.stats["sync_s"] += time.perf_counter() - t
        self.cache.stats["calls"] += 1
        idx = np.array(indices)
        lead, k = idx.shape[:-1], idx.shape[-1]
        xf = x.reshape(-1, x.shape[-1])
        idxf = idx.reshape(-1, k)
        n_tok = idxf.shape[0]
        admit = n_tok <= self.admit_max_tokens
        uniq = np.unique(idxf)
        groups = [uniq[i:i + self.group_max] for i in range(0, len(uniq), self.group_max)]

        if len(groups) == 1:
            y = self._run(xf, np.searchsorted(groups[0], idxf), groups[0], admit)
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


def build_streaming_model(model_dir, model_module, cache_gb=None, group_max=48,
                          nocache=True, threads=24, reserve_gb=10.0, verbose=True):
    """Build `model_module.Model` with dense weights resident and experts streamed.

    cache_gb=None sizes the expert cache from the GPU's recommended working set.
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
                        nocache=nocache, threads=threads)

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
    if verbose:
        print(f"resident {resident / GB:.1f} GB, expert cache {cache_gb:.1f} GB "
              f"({cache.capacity} of {args.num_hidden_layers * args.num_local_experts} experts, "
              f"{store.expert_bytes / 2**20:.1f} MB each)", flush=True)
    return model, cache
