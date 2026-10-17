"""Stream routed MoE experts from SSD instead of holding them in RAM.

The model is mlx-lm's own `muse_spark` (nothing is vendored). Only the dense weights
(attention, embeddings, routers, pre/post expert projections, norms) are loaded into memory. For every MoE layer the router runs first, then just the
selected experts are read from the original safetensors shards (by byte offset) and the
usual gate/up/down SwiGLU runs on them.

Expert cache = one slab per layer: for each of the 9 expert tensors an array
[slots, *tail] whose rows are expert slots, plus a per-layer LRU that maps expert -> slot.
Disk reads land directly in a slot (writable memoryview of the MLX buffer, split into
parallel chunks) and the GPU reads experts in place through gather_qmm slot indices, so
there is no per-token copy/stack of expert weights.

Expert tensors are stacked in the checkpoint with the expert index as the leading axis,
so expert `e` of a tensor with shape [E, a, b] is the contiguous byte range
[base + e*stride, base + (e+1)*stride) of the shard file.

Prefill (many tokens, many distinct experts) does not use the slabs for new data: it reads
each group of experts into temporary buffers, stacks them and discards them.

Optional prefetch (enable_prefetch): while layer l's expert math runs on the GPU, the
experts layer l+1 will probably need (layer l+1's router applied to layer l's residual
stream) are already being read into free/oldest slots in the background.
"""

import fcntl
import glob
import json
import os
import re
import struct
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.utils import load_model

PROJS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "scales", "biases")
PIECES = tuple((p, k) for p in PROJS for k in KINDS)  # 9 tensors per expert
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_MX_DTYPE = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
GB = 1 << 30


def read_safetensors_header(path):
    """{name: (abs_offset, nbytes, dtype, shape)} without reading any tensor data."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"{path}: not a safetensors file (only {len(head)} bytes)")
        (n,) = struct.unpack("<Q", head)
        if n > 100 << 20 or 8 + n > size:
            raise ValueError(f"{path}: implausible safetensors header size {n} (file is {size} bytes)")
        try:
            header = json.loads(f.read(n))
        except ValueError as e:
            raise ValueError(f"{path}: unreadable safetensors header: {e}") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    base = 8 + n
    out = {}
    for name, v in header.items():
        if name == "__metadata__":
            continue
        try:
            s, e = v["data_offsets"]
            dtype, shape = v["dtype"], tuple(v["shape"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: malformed entry for tensor {name}") from None
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e):
            raise ValueError(f"{path}: bad data_offsets for tensor {name}: {s}, {e}")
        if base + e > size:
            raise ValueError(f"{path}: tensor {name} extends past end of file (truncated download?)")
        out[name] = (base + s, e - s, dtype, shape)
    return out


def _join(futures):
    """Wait for ALL futures (a failed read must not leave sibling reads still writing into the buffer), then
    raise the first error."""
    wait(futures)
    for f in futures:
        f.result()


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
                # Our own slabs are the cache; don't let the OS page cache double-buffer.
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

    def fill_slot(self, layer, expert, views, slot, pool, chunk=None):
        """Queue reads of one expert's 9 tensors into row `slot` of the layer's slab
        (`views[i]` = flat writable uint8 view of tensor i's slab). Returns the futures."""
        chunk = self.chunk if chunk is None else chunk
        futs = []
        for i in range(len(PIECES)):
            fd, base, stride, _, _ = self.loc[(layer,) + PIECES[i]]
            view = views[i][slot * stride:(slot + 1) * stride]
            step = chunk or stride
            futs.extend(pool.submit(self._fill, fd, base + expert * stride + o, view[o:o + step])
                        for o in range(0, stride, step))
        return futs

    def read_temp(self, layer, experts):
        """Blocking read of whole experts into fresh buffers: {expert: 9 mx.arrays}."""
        raws = {(e, i): mx.zeros((self.loc[(layer,) + PIECES[i]][2],), dtype=mx.uint8)
                for e in experts for i in range(len(PIECES))}
        mx.eval(list(raws.values()))
        futs = []
        for (e, i), raw in raws.items():
            fd, base, stride, _, _ = self.loc[(layer,) + PIECES[i]]
            view = np.frombuffer(memoryview(raw), dtype=np.uint8)
            step = self.chunk or stride
            futs.extend(self.pool.submit(self._fill, fd, base + e * stride + o, view[o:o + step])
                        for o in range(0, stride, step))
        _join(futs)
        out = {}
        for (e, i), raw in raws.items():
            _, _, _, dtype, tail = self.loc[(layer,) + PIECES[i]]
            out.setdefault(e, []).append(raw.view(_MX_DTYPE[dtype]).reshape(tail))
        return {e: tuple(v) for e, v in out.items()}


class ExpertCache:
    """Per-layer slabs of expert slots with a per-layer LRU (expert -> slot)."""

    def __init__(self, store, slots):
        self.store = store
        L = store.num_layers
        self.slots = [max(1, int(s)) for s in slots]
        assert len(self.slots) == L
        self.arrays, self.views = [], []  # per layer: 9 [slots,*tail] arrays / flat uint8 views
        for l in range(L):
            raws, arrs = [], []
            for i in range(len(PIECES)):
                _, _, stride, dtype, tail = store.loc[(l,) + PIECES[i]]
                raws.append(mx.zeros((self.slots[l] * stride,), dtype=mx.uint8))
            mx.eval(raws)
            self.views.append([np.frombuffer(memoryview(r), dtype=np.uint8) for r in raws])
            for i, r in enumerate(raws):
                _, _, _, dtype, tail = store.loc[(l,) + PIECES[i]]
                arrs.append(r.view(_MX_DTYPE[dtype]).reshape((self.slots[l],) + tuple(tail)))
            self.arrays.append(arrs)
        self.order = [OrderedDict() for _ in range(L)]  # expert -> slot, oldest first
        self.free = [list(range(n)) for n in self.slots]
        self.pending = [dict() for _ in range(L)]  # expert -> (slot, futures): prefetch
        self.resid = {}  # layer -> last-token residual stream (set by prefetch hooks)
        self.prefetch_k = 0
        self.trace = None  # set to a list to record (layer, expert ids) of every decode step
        self.stats = dict(hits=0, misses=0, bytes_read=0, read_s=0.0, sync_s=0.0, calls=0,
                          fallback=0, stack_s=0.0, pf_issued=0, pf_used=0, pf_wasted=0,
                          pf_wait_s=0.0, pf_submit_s=0.0, predict_s=0.0)

    @property
    def capacity(self):
        return sum(self.slots)

    @property
    def resident(self):
        return sum(len(o) for o in self.order)

    def _take_slot(self, l, protect):
        """A free slot, else evict the oldest resident expert not in `protect`."""
        if self.free[l]:
            return self.free[l].pop()
        for e in self.order[l]:
            if e not in protect:
                return self.order[l].pop(e)
        return None

    def _settle(self, l, protect):
        """Finished prefetches nobody asked for become the oldest residents."""
        pend = self.pending[l]
        for e in [e for e in pend if e not in protect and all(f.done() for f in pend[e][1])]:
            slot, futs = pend.pop(e)
            try:
                _join(futs)
            except Exception:             # a speculative read failed: nobody needed it, just give the slot back
                self.free[l].append(slot)
                continue
            self.order[l][e] = slot
            self.order[l].move_to_end(e, last=False)
            self.stats["pf_wasted"] += 1

    def lookup(self, l, ids):
        """Make every expert in `ids` (distinct ints) resident in layer l's slab and return
        their slots, or None if the layer cannot hold them all at once."""
        protect = set(ids)
        self._settle(l, protect)
        order, pend = self.order[l], self.pending[l]
        hits = [e for e in ids if e in order]
        claimed = [e for e in ids if e in pend]
        missing = [e for e in ids if e not in order and e not in pend]
        if len(missing) > len(self.free[l]) + sum(1 for e in order if e not in protect):
            return None
        for e in hits:
            order.move_to_end(e)
        s = self.stats
        if claimed:
            t = time.perf_counter()
            for e in claimed:
                slot, futs = pend.pop(e)
                try:
                    _join(futs)
                except BaseException:     # the slot holds garbage: unmapped and free, never a hit
                    self.free[l].append(slot)
                    raise
                order[e] = slot
            s["pf_wait_s"] += time.perf_counter() - t
            s["pf_used"] += len(claimed)
        if missing:
            t = time.perf_counter()
            new, futs = [], []
            try:
                for e in missing:
                    slot = self._take_slot(l, protect)
                    new.append((e, slot))
                    futs.extend(self.store.fill_slot(l, e, self.views[l], slot, self.store.pool))
                _join(futs)
            except BaseException:         # e.g. an I/O error: the slots hold garbage, so they are free, not mapped
                wait(futs)
                self.free[l].extend(slot for _, slot in new)
                raise
            for e, slot in new:
                order[e] = slot
            s["read_s"] += time.perf_counter() - t
            s["bytes_read"] += len(missing) * self.store.expert_bytes
        s["hits"] += len(hits) + len(claimed)
        s["misses"] += len(missing)
        return [order[e] for e in ids]

    def prefetch(self, l, ids):
        """Start background reads into free/oldest slots for experts not yet resident."""
        order, pend = self.order[l], self.pending[l]
        ids = list(dict.fromkeys(int(x) for x in ids))
        protect = set(ids)
        issued = 0
        for e in ids:
            if e in order or e in pend:
                continue
            slot = self._take_slot(l, protect)
            if slot is None:
                break
            pend[e] = (slot, self.store.fill_slot(l, e, self.views[l], slot,
                                                  self.store.pf_pool, chunk=0))
            issued += 1
        self.stats["pf_issued"] += issued
        self.stats["bytes_read"] += issued * self.store.expert_bytes

    def fetch_group(self, l, group):
        """Per-expert 9-tuples for a temporary stacked call (prefill, or more distinct
        experts than slots): slab rows for residents, fresh reads for the rest. Nothing is
        admitted, so a long prompt cannot flush the decode working set."""
        order, out, missing = self.order[l], {}, []
        for e in group:
            e = int(e)
            if e in order:
                slot = order[e]
                out[e] = tuple(self.arrays[l][i][slot] for i in range(len(PIECES)))
            else:
                missing.append(e)
        s = self.stats
        s["hits"] += len(out)
        s["misses"] += len(missing)
        if missing:
            t = time.perf_counter()
            out.update(self.store.read_temp(l, missing))
            s["read_s"] += time.perf_counter() - t
            s["bytes_read"] += len(missing) * self.store.expert_bytes
        return out

    def summary(self):
        s = self.stats
        total = s["hits"] + s["misses"]
        out = (f"slabs {self.resident}/{self.capacity} experts, hit rate "
               f"{s['hits'] / max(total, 1):.1%} ({s['hits']}/{total}), read "
               f"{s['bytes_read'] / GB:.1f} GB (demand wait {s['read_s']:.1f}s), "
               f"gpu-sync {s['sync_s']:.1f}s, fallback calls {s['fallback']}")
        if s["pf_issued"]:
            out += (f" | prefetch issued {s['pf_issued']}, used {s['pf_used']} "
                    f"({s['pf_used'] / s['pf_issued']:.0%}), unclaimed {s['pf_wasted']}, "
                    f"waited {s['pf_wait_s']:.1f}s, submit {s['pf_submit_s']:.1f}s, "
                    f"predict-graph {s['predict_s']:.1f}s")
        return out


class StreamingSwitchGLU(nn.Module):
    """Drop-in for SwitchGLU(x, indices) that pulls the needed experts from disk."""

    def __init__(self, layer_idx, cache, template, post_expert_norm=None, group_max=48, admit_max_tokens=8):
        super().__init__()
        if post_expert_norm is not None:     # mlx-lm's MoE reads this norm off its `experts` module
            self.post_expert_norm = post_expert_norm
        self.layer_idx = layer_idx
        self.cache = cache
        self.group_max = group_max
        self.admit_max_tokens = admit_max_tokens  # more tokens than this = prefill path
        # Keep the template out of the module tree (no parameters, not quantized/saved).
        # Its weights are this layer's slabs; prefill temporarily swaps in stacked experts.
        object.__setattr__(self, "_glu", template)
        object.__setattr__(self, "_slab", cache.arrays[layer_idx])
        for (proj, kind), arr in zip(PIECES, self._slab):
            setattr(getattr(template, proj), kind, arr)
        object.__setattr__(self, "next_layer", None)  # set by enable_prefetch
        object.__setattr__(self, "ratio", None)

    def _run_stacked(self, xs, local_inds, group):
        """Prefill path: stack the group's experts, run, put the slabs back."""
        arrs = self.cache.fetch_group(self.layer_idx, group)
        t = time.perf_counter()
        stacked = [mx.stack([arrs[int(e)][i] for e in group]) for i in range(len(PIECES))]
        self.cache.stats["stack_s"] += time.perf_counter() - t
        for (proj, kind), arr in zip(PIECES, stacked):
            setattr(getattr(self._glu, proj), kind, arr)
        y = self._glu(xs, mx.array(local_inds.astype(np.uint32)))
        for (proj, kind), arr in zip(PIECES, self._slab):
            setattr(getattr(self._glu, proj), kind, arr)
        return y

    def _predict(self):
        """Lazy top-k expert ids for the NEXT layer, from this layer's residual stream.

        `r` is what this layer's router saw: the residual stream normalized by this layer's
        pre-feedforward norm. The next layer's router expects the same stream normalized by ITS
        norm, which differs only by the per-channel gain ratio `self.ratio`."""
        nxt, r = self.next_layer, self.cache.resid.get(self.layer_idx)
        if nxt is None or r is None:
            return None
        moe = nxt.mlp
        sc = (r.astype(mx.float32) * self.ratio) @ moe.gate.weight.T
        sc = mx.sigmoid(sc * moe.router_multiplier) + moe.gate.e_score_correction_bias
        k = min(self.cache.prefetch_k, sc.shape[-1])
        return mx.argpartition(-sc, kth=k - 1, axis=-1)[..., :k]

    def __call__(self, x, indices):
        cache, st = self.cache, self.cache.stats
        n_tok = int(np.prod(indices.shape[:-1]))
        speculate = cache.prefetch_k > 0 and n_tok <= self.admit_max_tokens
        t = time.perf_counter()
        pred = self._predict() if speculate else None
        st["predict_s"] += time.perf_counter() - t
        t = time.perf_counter()
        # Also flushes the previous layers' pending GPU work.
        mx.eval(indices) if pred is None else mx.eval(indices, pred)
        st["sync_s"] += time.perf_counter() - t
        st["calls"] += 1
        idx = np.array(indices)
        lead, k = idx.shape[:-1], idx.shape[-1]
        xf = x.reshape(-1, x.shape[-1])
        idxf = idx.reshape(-1, k)
        uniq = np.unique(idxf)
        if cache.trace is not None and n_tok == 1:
            cache.trace.append((self.layer_idx, uniq.copy()))

        slots = None
        if n_tok <= self.admit_max_tokens and len(uniq) <= cache.slots[self.layer_idx]:
            slots = cache.lookup(self.layer_idx, uniq.tolist())
        if slots is not None:  # decode: experts are read in place from the slab
            lut = np.zeros(cache.store.num_experts, dtype=np.uint32)
            lut[uniq] = slots
            y = self._glu(xf, mx.array(lut[idxf]))
            if speculate:
                if pred is not None:
                    t = time.perf_counter()
                    cache.prefetch(self.layer_idx + 1, np.array(pred).reshape(-1))
                    st["pf_submit_s"] += time.perf_counter() - t
                mx.async_eval(y)  # GPU starts the expert math now, overlapping the reads
            return y.reshape(*lead, k, -1)

        st["fallback"] += 1
        groups = [uniq[i:i + self.group_max] for i in range(0, len(uniq), self.group_max)]
        if len(groups) == 1:
            y = self._run_stacked(xf, np.searchsorted(groups[0], idxf), groups[0])
            return y.reshape(*lead, k, -1)

        # Many distinct experts (prefill): handle one group of experts at a time and
        # scatter the (token, slot) rows back into place.
        pieces, orders = [], []
        for g in groups:
            t_sel, j_sel = np.nonzero(np.isin(idxf, g))
            local = np.searchsorted(g, idxf[t_sel, j_sel])[:, None]
            y = self._run_stacked(xf[mx.array(t_sel.astype(np.uint32))], local, g)[:, 0]
            mx.eval(y)  # release this group's stacked weights before the next one
            pieces.append(y)
            orders.append(t_sel * k + j_sel)
        inv = np.argsort(np.concatenate(orders))
        y = mx.concatenate(pieces)[mx.array(inv.astype(np.uint32))]
        return y.reshape(*lead, k, -1)


def _hook_moe(moe, layer_idx, cache):
    """Record each layer's last-token router input (the residual stream after the pre-feedforward norm)."""
    cls = type(moe)

    class Hooked(cls):
        def __call__(self, x):
            cache.resid[layer_idx] = x[:, -1:] if x.shape[1] <= 8 else None
            return super().__call__(x)

    moe.__class__ = Hooked


def _gain_ratio(num, den):
    """num / den per channel; 0 where den is ~0 (that channel carries nothing to rescale)."""
    num, den = num.astype(mx.float32), den.astype(mx.float32)
    return mx.where(mx.abs(den) > 1e-6, num / mx.where(mx.abs(den) > 1e-6, den, 1.0), 0.0)


def enable_prefetch(model, cache, k=4):
    """Predict layer l+1's experts during layer l and read them in the background."""
    cache.prefetch_k = k
    layers = model.layers
    for i, layer in enumerate(layers):
        _hook_moe(layer.mlp, i, cache)
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        glu = layer.mlp.experts
        object.__setattr__(glu, "next_layer", nxt)
        if nxt is not None:
            ratio = _gain_ratio(nxt.pre_feedforward_layernorm.weight, layer.pre_feedforward_layernorm.weight)
            mx.eval(ratio)
            object.__setattr__(glu, "ratio", ratio)


def split_slots(total, num_layers, weights=None):
    """Divide `total` expert slots over the layers (equal, or proportional to `weights`)."""
    w = np.ones(num_layers) if weights is None else np.asarray(weights, dtype=float)
    return np.maximum(1, np.floor(w / w.sum() * total)).astype(int).tolist()


def build_streaming_model(model_dir, cache_gb=None, slots_per_layer=None, layer_weights=None, group_max=48,
                          nocache=True, threads=24, reserve_gb=10.0, prefetch_k=0, prefetch_threads=16,
                          verbose=True):
    """Load mlx-lm's `muse_spark` with the dense weights resident and the experts streamed.

    The expert slabs get `slots_per_layer` slots per layer (an int or a list), or, by
    default, `cache_gb` worth of slots split over the layers (`layer_weights` = relative
    shares, default equal). cache_gb=None sizes it from the GPU's recommended working set.
    prefetch_k>0 turns on speculative next-layer reads (top-k predicted experts).
    """
    model_dir = Path(model_dir)
    # lazy=True: nothing is read yet. Without a vision_config the model has no vision tower (text only)
    # and its weights are dropped. mlx-lm builds each module with the bits the checkpoint gives it.
    model, config = load_model(model_dir, lazy=True, model_config={"vision_config": None})
    text = model.args.text
    n_layers, n_experts = text.num_hidden_layers, text.num_local_experts

    store = ExpertStore(model_dir, n_layers, n_experts, nocache=nocache, threads=threads,
                        prefetch_threads=prefetch_threads)
    resident = sum(v.nbytes for k, v in tree_flatten(model.parameters()) if not _EXPERT_RE.match(k))
    if slots_per_layer is None:
        if cache_gb is None:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            limit = info["max_recommended_working_set_size"] / GB
            cache_gb = max(1.0, limit - resident / GB - reserve_gb)
        slots_per_layer = split_slots(int(cache_gb * GB // store.expert_bytes), n_layers, layer_weights)
    elif isinstance(slots_per_layer, int):
        slots_per_layer = [slots_per_layer] * n_layers
    cache = ExpertCache(store, slots_per_layer)

    for i, layer in enumerate(model.layers):
        old = layer.mlp.experts
        qs = [getattr(old, p) for p in PROJS]
        if not all(hasattr(q, "scales") and getattr(q, "biases", None) is not None
                   and getattr(q, "global_scale", None) is None for q in qs) \
                or len({(q.group_size, q.bits, q.mode) for q in qs}) != 1:
            raise ValueError(f"layer {i}: the experts must be affine-quantized with one setting "
                             "(an unquantized or mixed-precision layer cannot be streamed)")
        template = SwitchGLU(text.moe_hidden_size, text.moe_intermediate_size, 1, activation=old.activation)
        nn.quantize(template, group_size=qs[0].group_size, bits=qs[0].bits, mode=qs[0].mode)
        layer.mlp.experts = StreamingSwitchGLU(i, cache, template, getattr(old, "post_expert_norm", None),
                                               group_max=group_max)
        del old, qs
    mx.eval(model.parameters())  # the dense weights; the experts left the model above and are never read
    if prefetch_k > 0:
        enable_prefetch(model, cache, prefetch_k)
    if verbose:
        total = n_layers * n_experts
        print(f"resident {resident / GB:.1f} GB, expert slabs {cache.capacity * store.expert_bytes / GB:.1f} GB "
              f"({cache.capacity} of {total} experts, {min(cache.slots)}-{max(cache.slots)} per layer, "
              f"{store.expert_bytes / 2**20:.1f} MB each)"
              + (f", prefetch top-{prefetch_k}" if prefetch_k else ""), file=sys.stderr, flush=True)
    return model, cache
