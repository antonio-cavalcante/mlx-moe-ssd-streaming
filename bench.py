"""Benchmark one generation and report where the time and the SSD traffic go.

  .venv/bin/python bench.py --corpus some_text.txt --context-tokens 4000 --max-tokens 200 --iostat

The prompt is the first --context-tokens tokens of the corpus plus a request to summarize it. Output (stderr, one block):
prefill/decode speed, how the expert lookups were served (already in RAM / prefetched / demand miss), SSD bytes
by kind, the fraction of the decode wall time with at least one read in flight, and (with --iostat) what macOS
itself says the disk did meanwhile (`iostat -d -w 1 disk0`), as an independent check of our own counters.
A JSON line with the numbers goes to stdout.
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
import expert_stream as es  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402

GB = es.GB


class Sampler(threading.Thread):
    """Every `dt` seconds: (time, cumulative SSD bytes, cumulative busy seconds) from our IO meter."""

    def __init__(self, io, dt=0.25):
        super().__init__(daemon=True)
        self.io, self.dt, self.rows, self.stop = io, dt, [], threading.Event()

    def run(self):
        while not self.stop.wait(self.dt):
            s = self.io.snapshot()
            self.rows.append((time.perf_counter(), s["demand_bytes"] + s["prefetch_bytes"], s["busy_s"]))


class IostatWatcher:
    def __init__(self, disk="disk0"):
        self.p = subprocess.Popen(["iostat", "-d", "-w", "1", disk], stdout=subprocess.PIPE, text=True)
        self.rows = []  # (time, MB/s)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            f = line.split()
            if len(f) == 3 and re.fullmatch(r"[\d.]+", f[2]):
                self.rows.append((time.perf_counter(), float(f[2])))

    def close(self):
        self.p.terminate()


def build_prompt(tok, corpus, n_tokens, effort):
    ids = tok.encode(Path(corpus).read_text())
    if len(ids) < n_tokens:
        sys.exit(f"{corpus} has only {len(ids)} tokens")
    text = tok.decode(ids[:n_tokens])
    msg = f"{text}\n\n---\nSummarize the text above in detail, then list ten specific observations about it."
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                   reasoning_effort=effort)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_model_args(ap, ROOT)
    ap.add_argument("--corpus", required=True, help="text file the prompt is cut from")
    ap.add_argument("--context-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--iostat", action="store_true", help="also record `iostat` for the whole disk")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    model, cache, tok = cc.load_all(args, ROOT)
    io = cache.store.io
    prompt = build_prompt(tok, args.corpus, args.context_tokens, args.reasoning_effort)
    watcher = IostatWatcher() if args.iostat else None
    sampler = Sampler(io)
    t_start = time.perf_counter()
    mx.reset_peak_memory()
    st0, io0 = dict(cache.stats), io.snapshot()
    st1 = io1 = t_first = last = None
    for resp in stream_generate(model, tok, prompt, max_tokens=args.max_tokens):  # greedy
        if t_first is None:
            t_first, st1, io1 = time.perf_counter(), dict(cache.stats), io.snapshot()
            sampler.start()
        last = resp
    t_end = time.perf_counter()
    st2, io2 = dict(cache.stats), io.snapshot()
    sampler.stop.set()
    if watcher:
        time.sleep(1.2)
        watcher.close()

    n_dec = last.generation_tokens - 1
    wall = t_end - t_first
    d = {k: st2[k] - st1[k] for k in st1}
    dio = {k: io2[k] - io1[k] for k in io1}
    lookups = d["hits"] + d["misses"]
    in_ram = d["hits"] - d["pf_used"]
    disk_gb = (dio["demand_bytes"] + dio["prefetch_bytes"]) / GB
    r = dict(label=args.label, prompt_tokens=last.prompt_tokens, prefill_s=round(t_first - t_start, 1),
             prefill_tps=round(last.prompt_tps, 2), decode_tokens=n_dec, decode_s=round(wall, 1),
             decode_tps=round(last.generation_tps, 2), peak_gib=round(mx.get_peak_memory() / GB, 1),
             lookups_per_token=round(lookups / max(n_dec, 1), 1),
             served_from_ram=round(in_ram / max(lookups, 1), 4),
             served_by_prefetch=round(d["pf_used"] / max(lookups, 1), 4),
             demand_miss=round(d["misses"] / max(lookups, 1), 4),
             reported_hit_rate=round(d["hits"] / max(lookups, 1), 4),
             ssd_gb_per_token=round(disk_gb / max(n_dec, 1), 3),
             demand_gb=round(dio["demand_bytes"] / GB, 1), prefetch_gb=round(dio["prefetch_bytes"] / GB, 1),
             prefetch_issued=d["pf_issued"], prefetch_used=d["pf_used"],
             busy_frac=round(dio["busy_s"] / wall, 3), gb_s_while_busy=round(disk_gb / max(dio["busy_s"], 1e-9), 2),
             gb_s_average=round(disk_gb / wall, 2), reads_in_flight_avg=round(dio["thread_s"] / max(dio["busy_s"], 1e-9), 1),
             demand_wait_frac=round(d["read_s"] / wall, 3), pf_wait_frac=round(d["pf_wait_s"] / wall, 3),
             gpu_sync_frac=round(d["sync_s"] / wall, 3))
    rows = [x for x in sampler.rows]
    if len(rows) > 2:
        rates = [(b[1] - a[1]) / GB / (b[0] - a[0]) for a, b in zip(rows, rows[1:])]
        r["timeline_0.25s_gb_s"] = dict(min=round(min(rates), 1), median=round(sorted(rates)[len(rates) // 2], 1),
                                        max=round(max(rates), 1),
                                        idle_buckets=round(sum(x < 0.5 for x in rates) / len(rates), 3))
    if watcher:
        v = [mbs for t, mbs in watcher.rows if t_first <= t <= t_end + 1.0]
        if v:
            r["iostat_mb_s"] = dict(seconds=len(v), mean=round(sum(v) / len(v)), min=round(min(v)), max=round(max(v)),
                                    gb_per_token=round(sum(v) / 1024 / max(n_dec, 1) * wall / len(v), 3))
    print(f"--- {args.label or 'bench'}: {last.prompt_tokens} prompt tokens, {n_dec} decoded", file=sys.stderr)
    print(f"prefill {r['prefill_s']}s ({r['prefill_tps']} tok/s) | decode {r['decode_tps']} tok/s | peak {r['peak_gib']} GiB",
          file=sys.stderr)
    print(f"expert lookups/token {r['lookups_per_token']}: from RAM {r['served_from_ram']:.1%}, prefetched "
          f"{r['served_by_prefetch']:.1%}, demand miss {r['demand_miss']:.1%}  (the usual 'hit rate' = RAM + prefetched = "
          f"{r['reported_hit_rate']:.1%})", file=sys.stderr)
    print(f"SSD: {r['ssd_gb_per_token']} GB/token (demand {r['demand_gb']} GB + prefetch {r['prefetch_gb']} GB; prefetch "
          f"used {r['prefetch_used']}/{r['prefetch_issued']}), busy {r['busy_frac']:.0%} of decode wall time, "
          f"{r['gb_s_while_busy']} GB/s while busy, {r['gb_s_average']} GB/s on average, {r['reads_in_flight_avg']} reads in flight",
          file=sys.stderr)
    print(f"decode wall time: waiting for demand reads {r['demand_wait_frac']:.0%}, waiting for prefetch "
          f"{r['pf_wait_frac']:.0%}, gpu sync {r['gpu_sync_frac']:.0%}", file=sys.stderr)
    if "timeline_0.25s_gb_s" in r:
        print("our counters per 0.25 s:", r["timeline_0.25s_gb_s"], file=sys.stderr)
    if "iostat_mb_s" in r:
        print("iostat disk0:", r["iostat_mb_s"], file=sys.stderr)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
