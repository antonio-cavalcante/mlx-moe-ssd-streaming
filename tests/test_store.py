"""ExpertStore / ExpertCache: header validation, slot accounting, and what happens when a read fails.

Run: .venv/bin/python -I tests/test_store.py   (no real model needed)
"""
import json
import os
import shutil
import struct
import sys
import tempfile
from concurrent.futures import wait
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import expert_stream as es  # noqa: E402
import test_tiny as tt  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


def raises(fn, exc=Exception, match=""):
    try:
        fn()
    except exc as e:
        return match in str(e)
    return False


def write_st(path, header, data=b"", raw_header=None, n=None):
    h = raw_header if raw_header is not None else json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(h) if n is None else n) + h + data)


def header_tests(tmp):
    print("safetensors header validation")
    p = tmp / "h.safetensors"
    good = {"a": {"dtype": "U32", "shape": [2, 2], "data_offsets": [0, 16]}, "__metadata__": {"x": "y"}}
    write_st(p, good, b"\0" * 16)
    check("a valid file parses", es.read_safetensors_header(p) == {"a": (8 + len(json.dumps(good)), 16, "U32", (2, 2))})
    p.write_bytes(b"abc")
    check("shorter than the length prefix", raises(lambda: es.read_safetensors_header(p), ValueError, "not a safetensors"))
    write_st(p, good, b"", n=10**6)
    check("header length beyond the file", raises(lambda: es.read_safetensors_header(p), ValueError, "implausible"))
    write_st(p, None, raw_header=b"{not json")
    check("invalid JSON header", raises(lambda: es.read_safetensors_header(p), ValueError, "unreadable"))
    write_st(p, None, raw_header=b"[1, 2]")
    check("header that is not an object", raises(lambda: es.read_safetensors_header(p), ValueError, "not an object"))
    write_st(p, {"a": {"dtype": "U32", "shape": [1]}})
    check("entry without data_offsets", raises(lambda: es.read_safetensors_header(p), ValueError, "malformed"))
    write_st(p, {"a": {"dtype": "U32", "shape": [1], "data_offsets": [8, 4]}}, b"\0" * 16)
    check("end before start", raises(lambda: es.read_safetensors_header(p), ValueError, "bad data_offsets"))
    write_st(p, {"a": {"dtype": "U32", "shape": [1], "data_offsets": [-4, 4]}}, b"\0" * 16)
    check("negative offset", raises(lambda: es.read_safetensors_header(p), ValueError, "bad data_offsets"))
    write_st(p, good, b"\0" * 8)
    check("data truncated (interrupted download)", raises(lambda: es.read_safetensors_header(p), ValueError, "past end"))


def truth(store, l, e, i):
    """Expert e's bytes of tensor i, read straight from the shard (independent of the cache code)."""
    fd, base, stride, _, _ = store.loc[(l,) + es.PIECES[i]]
    return os.pread(fd, stride, base + e * stride)


def slab(cache, l, slot, i):
    stride = cache.store.loc[(l,) + es.PIECES[i]][2]
    return bytes(cache.views[l][i][slot * stride:(slot + 1) * stride])


def intact(cache, l):
    """Every slot is in exactly one place (free, resident or pending) and resident slots hold their expert."""
    slots = list(cache.free[l]) + list(cache.order[l].values()) + [s for s, _ in cache.pending[l].values()]
    if sorted(slots) != list(range(cache.slots[l])):
        return False
    return all(slab(cache, l, s, i) == truth(cache.store, l, e, i)
               for e, s in cache.order[l].items() for i in range(len(es.PIECES)))


def cache_tests(tmp):
    print("ExpertCache: LRU, accounting")
    model, cache = es.build_streaming_model(tmp, slots_per_layer=4, prefetch_k=0, verbose=False)
    l = 1
    got = cache.lookup(l, [0, 1, 2, 3])
    check("4 misses fill the layer", got is not None and sorted(got) == [0, 1, 2, 3] and cache.stats["misses"] == 4)
    check("slots hold the right bytes", intact(cache, l))
    cache.lookup(l, [0, 1])
    check("re-asking is all hits", cache.stats["hits"] == 2 and cache.stats["misses"] == 4)
    cache.lookup(l, [9])
    check("the least recently used expert (2) is evicted", 2 not in cache.order[l] and {0, 1, 3, 9} == set(cache.order[l]))
    check("accounting after an eviction", intact(cache, l))
    check("more distinct experts than slots -> None (stacked fallback), nothing changed",
          cache.lookup(l, [4, 5, 6, 7, 8]) is None and intact(cache, l) and 9 in cache.order[l])
    got = cache.lookup(l, [10, 11, 12, 13])
    check("a full replacement is fine", got is not None and sorted(got) == [0, 1, 2, 3] and intact(cache, l))
    t = cache.fetch_group(l, [10, 5])
    check("fetch_group: resident row + fresh read, nothing admitted",
          bytes(memoryview(t[5][0])) == truth(cache.store, l, 5, 0) and 5 not in cache.order[l])

    print("ExpertCache: failed reads")
    store = cache.store
    real = store._fill
    calls = []

    def flaky(fd, off, view):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("injected read error")
        return real(fd, off, view)

    l = 2
    cache.lookup(l, [0, 1])
    store._fill = flaky
    calls.clear()
    check("a failing demand read raises", raises(lambda: cache.lookup(l, [4, 5]), OSError, "injected"))
    store._fill = real
    check("every slot is still accounted for", intact(cache, l), (cache.free[l], dict(cache.order[l])))
    check("the experts of the failed read are not mapped", 4 not in cache.order[l] and 5 not in cache.order[l])
    got = cache.lookup(l, [4, 5])
    check("the same lookup works afterwards, with the right bytes", got is not None and intact(cache, l)
          and slab(cache, l, got[0], 0) == truth(store, l, 4, 0))

    l = 3
    cache.lookup(l, [0])
    store._fill = flaky
    calls.clear()
    cache.prefetch(l, [6, 7])
    for e, (_, futs) in list(cache.pending[l].items()):
        wait(futs)
    store._fill = real
    check("failed prefetch: a claim raises", raises(lambda: cache.lookup(l, [6, 7]), OSError, "injected"))
    check("failed prefetch: slots accounted for", intact(cache, l), (cache.free[l], dict(cache.order[l])))
    got = cache.lookup(l, [6, 7])
    check("and it can be read again", got is not None and intact(cache, l))

    l = 4
    store._fill = flaky
    calls.clear()
    cache.prefetch(l, [8])
    wait(cache.pending[l][8][1])
    store._fill = real
    cache.lookup(l, [1])             # nobody asked for 8: a failed speculative read is just dropped
    check("failed unclaimed prefetch is dropped quietly", 8 not in cache.order[l] and 8 not in cache.pending[l]
          and intact(cache, l))

    print("ExpertCache: file truncated under us")
    l = 0
    fd = store.loc[(l, "gate_proj", "weight")][0]
    size = os.fstat(fd).st_size
    path = next(p for p in sorted(Path(tmp).glob("model*.safetensors")))
    # find the shard that holds layer 0's gate_proj weights and cut off its tail
    for p in Path(tmp).glob("model*.safetensors"):
        if os.stat(p).st_ino == os.fstat(fd).st_ino:
            path = p
    with open(path, "r+b") as f:
        f.truncate(size // 2)
    bad = [e for e in range(16) if store.loc[(l, "gate_proj", "weight")][1] + (e + 1) * store.loc[(l, "gate_proj", "weight")][2] > size // 2]
    check("(precondition) some experts now lie past the end", bool(bad))
    check("a short read is an error, not silent garbage", raises(lambda: cache.lookup(l, bad[:2]), OSError, "short read"))
    check("slots accounted for after the short read", intact(cache, l))

    print("split_slots")
    check("equal split", es.split_slots(100, 4) == [25, 25, 25, 25])
    check("weighted split follows the weights", es.split_slots(100, 4, [1, 1, 2, 4]) == [12, 12, 25, 50])
    check("every layer keeps at least one slot", min(es.split_slots(3, 6)) == 1)
    check("never more than the total (when it fits)", sum(es.split_slots(1000, 62, list(range(1, 63)))) <= 1000)


def store_tests(tmp):
    print("ExpertStore on a damaged checkpoint")
    work = Path(tempfile.mkdtemp(prefix="store_bad_"))
    try:
        broken = work / "broken"
        shutil.copytree(tmp, broken)
        shard = sorted(broken.glob("model*.safetensors"))[1]
        with open(shard, "r+b") as f:
            f.truncate(os.path.getsize(shard) - 100)
        check("a truncated shard is refused when the store opens",
              raises(lambda: es.ExpertStore(broken, 5, 16), ValueError, "past end"))
        missing = work / "missing"
        shutil.copytree(tmp, missing)
        next(missing.glob("model-00002*.safetensors")).unlink()
        check("a missing shard is refused", raises(lambda: es.ExpertStore(missing, 5, 16), FileNotFoundError, "missing"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="store_ckpt_"))
    try:
        tt.make_checkpoint(tmp)
        header_tests(tmp)
        store_tests(tmp)
        cache_tests(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
