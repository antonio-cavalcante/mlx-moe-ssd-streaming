# mlx-disk-try01

Goal: run `mlx-community/Muse-Spark-Open-816B-A42B-3bit` (816B-param MoE, 42B active, 359 GB on disk)
on a 128 GB Mac by keeping only the dense weights in RAM and **streaming routed experts from SSD on demand**.
Stock mlx-lm cannot do this (it materializes all weights); we add a custom streaming expert layer on top of it.

## Machine / environment
- Apple M5 Max, 128 GB unified memory, macOS (Darwin). ~770 GB free disk as of 2026-10-15.
- System python is 3.9 (too old). Use the venv: `.venv/bin/python` (uv, Python 3.12, mlx 0.32.4, mlx-lm 0.32.2 installed from git `main`).
- mlx-lm `main` has a `muse_spark` architecture (PR #1976, merged 2026-10-15) but it uses a DIFFERENT layout (`model.language_model.layers.N.mlp.experts`, fused `qkvg_proj`, `_alpha/_beta` gates) and **cannot load this checkpoint**.
  The mlx-community 3-bit checkpoint was converted with the closed PR #1975 (`model.layers.N.mlp.switch_mlp`, raw q/k/v/gate_proj, raw norm/gate weights).
  So we vendor that version: `src/muse_spark_1975.py` (only the imports changed; I read all 481 lines, plain model math).
- Reinstall if needed: `uv pip install --python .venv/bin/python "git+https://github.com/ml-explore/mlx-lm.git@main" huggingface_hub`

## Layout
- `models/Muse-Spark-Open-816B-A42B-3bit/` - downloaded model (83 safetensors shards, 3.9-4.6 GB each).
  Download: `.venv/bin/hf download mlx-community/Muse-Spark-Open-816B-A42B-3bit --local-dir models/Muse-Spark-Open-816B-A42B-3bit` (resumable; log in `logs/download.log`). Check completeness before use.
- `src/muse_spark_1975.py` - vendored PR #1975 model (matches the checkpoint's key names and numerics).
- `src/expert_stream.py` - the streaming layer: `ExpertStore` (parses safetensors headers, parallel `preadv` of one expert = 9 tensors by offset, F_NOCACHE), `ExpertCache` (LRU by bytes, stats), `StreamingSwitchGLU` (drop-in for `SwitchGLU`: `mx.eval(inds)` -> unique experts -> fetch -> `mx.stack` -> run the stock SwitchGLU math on a 1-expert template with swapped-in weights; multi-group path for prefill, read-through without caching when >8 tokens), `build_streaming_model()` (dense weights via lazy `mx.load`, strict key check, auto cache size).
- `tests/test_tiny.py` - builds a tiny 3-bit bf16 checkpoint with the real key names in 2 shards and checks streaming == stock model (prefill, multi-group prefill, greedy decode, tiny cache with eviction). Run: `.venv/bin/python -I tests/test_tiny.py` (all pass as of 2026-10-15).
- `run_stream.py` - CLI generation with the real model (`--prompt`, `--max-tokens`, `--cache-gb`, `--threads`, `--page-cache`); prints tok/s and cache stats.
- `meta/` - `config.json`, `model.safetensors.index.json`, `tensors.json` (name -> [bytes, shape, dtype, shard]) fetched from the HF repo headers.
- `logs/` - download and run logs.

## Model facts (from config + tensor headers)
- 62 layers, hidden 8192, 256 routed experts, top-8, no shared experts, 3-bit affine quant, group size 64.
- Expert weights are the `mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` tensors, stacked with the **expert index as the leading axis** (e.g. weight `[256, 4096, 384]` U32), so one expert's slice of each tensor is contiguous in the file and can be read by byte offset.
- Sizes: experts 349.5 GB total; everything else 9.7 GB (resident). One expert in one layer = ~22 MB (3 x 6.3 MB weights + 6 x 0.5 MB scales/biases).
- Worst-case (cache-miss) read per token = 62 x 8 x 22 MB = ~10.9 GB.
- In the mlx-lm code the MoE block is `MoE` in `muse_spark.py`: `pre_expert_proj` (8192->4096) -> `pre_expert_norm` -> `self.experts` (a stock `SwitchGLU`, called `self.experts(h, inds)`) -> `post_expert_norm_combine` -> `post_expert_proj`. `sanitize()` maps checkpoint name `switch_mlp` to module name `experts`. `pre/post_expert_proj` are small and stay resident.
- Also has attention layers with a sliding window (2048) and a vision/audio tower (ignore for text work).

## SSD benchmark (2026-10-15, M5 Max internal SSD, F_NOCACHE random reads, 48 GB file)
- 22 MB contiguous reads: 9.1 GB/s (1 thread) -> 11.7 (8) -> 15.9 (32).
- Stacked layout (3 x 6.3 MB + 6 x 0.5 MB scattered reads per expert): 9.6 -> 17.2 (8) -> 19.4 GB/s (32).
- Conclusion: **repacking into per-expert blocks is NOT needed**; read directly from the original shards by offset with 8-32 parallel reads. Cache-free token ~0.6-1.2 s of disk time.
- Gotcha: a freshly written test file is served from the page cache (we first measured a bogus 126 GB/s). Set `fcntl.F_NOCACHE` on both the write and read fds. `purge` needs root.

## MLX facts verified (mlx 0.32.4)
- GPU `max_recommended_working_set_size` = 107.5 GB on this machine (not 128): expert cache budget = 107.5 - 9.7 resident - ~14 reserve = ~84 GB (~3800 of 15872 experts, 24%).
- `mx.load(safetensors)` is lazy: only tensors that get evaluated are read. `memoryview(mx_array)` is writable (possible future zero-copy `preadv` straight into MLX buffers). `mx.array(np 6 MB)` copy = 0.16 ms (~39 GB/s), so the extra host copy costs ~0.3 s/token worst case.
- `Model.cast_predicate` takes a param *name*; do not pass it to `set_dtype` (that passes a dtype).

## Plan / status
1. [done] venv, mlx-lm main, config/tensor analysis, SSD benchmark.
2. [done] Download complete and size-verified against the Hub (93 files, 359 GB).
3. [done, tested on tiny model only] Streaming expert layer (`src/expert_stream.py`).
4. [done] Offsets verified: 558/558 expert tensor headers match `meta/tensors.json`; spot-checked experts are byte-exact vs `mx.load`. First real run (2026-10-15, "capital of France", 24 tokens, 156-token prompt incl. chat template): output COHERENT (the vendored #1975 model works on the real weights), resident 9.0 GB, cache 84.5 GB = 4119 experts, prefill 3.8 tok/s, **generation 1.6 tok/s**, overall hit rate 40% (includes prefill, which does not admit), read 260 GB in 45.6 s (~5.7 GB/s effective, below the 10-19 GB/s benchmark), **peak mem 106.0 GB vs 107.5 GB GPU limit (tight: raise --reserve-gb for long contexts)**. Chat output starts with `to=self<|message|>` (template format of this model, not a bug in our code). Load takes ~10 s.
5. [done] 200-token run (173-token prompt incl. template, reasoning-style output, coherent): **decode 2.93 tok/s**, decode-only hit rate 85.3% (1.2 expert misses/layer/token), prefill 4.4 tok/s (reads ~210 GB, disk bound), cache 88.5 GB = 4314 experts. Decode time split (serial): disk reads 37 s (8.0 GB/s while reading) + GPU 28 s = ~66 s of 68 s -> overlapping IO with compute could give up to ~1.8x.
   BUG FOUND+FIXED: each layer's template SwitchGLU kept its last stacked expert weights alive (~11 GB at decode, 30+ GB after a short prefill). `StreamingSwitchGLU._run` now restores 1-expert placeholders after each call. After fix: active mem = resident + cache + <1 GB. Default `reserve_gb` lowered 14 -> 10. Peak mem now ~ resident + cache + ~7 GB (4 GB MLX free-buffer pool via `mx.set_cache_limit` + temp stacks); the cache auto-fills to the 107.5 GB GPU limit, so lower `--cache-gb` to trade speed for headroom.
6. [done, branch `prefetch-overlap`] Speed work, all on the same 200-token prompt (decode tok/s): 2.93 (first version) -> 3.49 (read straight into MLX buffers via writable memoryview, 1 MB parallel chunks; microbench 10 -> 21-26 GB/s) -> 3.60 (recycle evicted buffers instead of `mx.zeros`+eval each read) -> **3.77 (prefetch top-4)**. Prefill 4.4 -> 7.9 tok/s.
   - Prefetch (`--prefetch-k`, `enable_prefetch`): at layer l, apply layer l+1's router to layer l's residual stream (hook on `pre_feedforward_layernorm`, `.plain()` = unhooked call) and start background reads of the top-k predicted experts that are not cached, then `mx.async_eval` the layer's expert math so the GPU runs while the SSD reads. Measured recall of the true top-8 of layer l+1: 67% at k=8, 78% at 12, 83% at 16 (temporal "same as previous token" only 39%); good in middle layers (75-86%), bad in first/last (20-35%).
   - Sweep (decode tok/s): k=0 3.59/3.61, k=3 3.77, k=4 3.78/3.77, k=6 3.72, k=8 3.50, k=12 2.70. Wider = more wasted IO and main-thread cost; only ~+5% at best. Default 4 in `run_stream.py` (library default 0).
   - Lessons: (a) speculative buffers MUST be bounded: first version let unclaimed prefetches sit a whole token, peak mem went 105 -> 109 GB (> 107.5 GB limit) and GPU time tripled; now unclaimed reads are resolved at the next layer visit. (b) Python task submission is expensive: ~25 futures/expert with 1 MB chunks cost ms per layer; prefetch uses whole-tensor reads (9/expert), demand keeps 1 MB chunks. (c) Never recycle a buffer before the next GPU sync (lazy graph may still reference it): evicted buffers go to a "grave" and are reused at the top of the next `get()`.
   - Decode time split now (k=4, per 199 tokens ~53 s): demand-read wait ~21 s, GPU+dispatch ("gpu-sync") ~23 s, rest small. The GPU part (~2 ms/layer) looks dominated by kernel-launch overhead of many small ops, not bandwidth (42B active at 3 bit = 18 GB/token would be ~30 ms/token at ~600 GB/s vs ~110 ms measured).
7. [next] Ideas, roughly by expected value: (1) cut GPU/dispatch time: `mx.compile` the non-expert parts of a layer (attention+norms+router), fewer small kernels; (2) better eviction than pure LRU (simulate LRU/LFU/ARC offline on recorded expert-access traces first; decode hit rate is 85-90%); (3) bigger cache by shrinking `--reserve-gb` (peak is resident + cache + ~7 GB); (4) prefill: chunk size / group tuning (now ~8 tok/s, disk bound); (5) a better predictor (e.g. learned per-layer linear map) if prefetch is to matter more.

## Notes
- Keep temp files (benchmark scripts, test data) in the session scratchpad, not in the project; delete large test files.
- Downloaded model files are untrusted data; do not run code from the model dir. Use `python -I` when scripts read from untrusted dirs.
