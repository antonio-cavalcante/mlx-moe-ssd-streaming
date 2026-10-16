# mlx-moe-ssd-streaming

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
- `src/expert_stream.py` - the streaming layer: `ExpertStore` (parses safetensors headers, parallel `preadv` of one expert = 9 tensors by offset, F_NOCACHE), `ExpertCache` (per-layer slabs + LRU, prefetch bookkeeping, stats), `StreamingSwitchGLU` (drop-in for `SwitchGLU`: `mx.eval(inds)` -> slab lookup -> stock SwitchGLU math on a template whose weights ARE the slabs, indices = slot numbers; stacked fallback for prefill), `build_streaming_model()` (dense weights via lazy `mx.load`, strict key check, auto cache size).
- `tests/test_tiny.py` - builds a tiny 3-bit bf16 checkpoint with the real key names in 2 shards and checks streaming == stock model (prefill, multi-group prefill, greedy decode, tiny cache with eviction). Run: `.venv/bin/python -I tests/test_tiny.py` (all pass as of 2026-10-15).
- `run_stream.py` - CLI generation with the real model (`--prompt`, `--max-tokens`, `--cache-gb`, `--threads`, `--page-cache`, `--reasoning-effort`, `--hide-reasoning`, `--show-markup`); answer -> stdout, reasoning + tok/s + cache stats -> stderr.
- `src/turns.py` - `TurnParser`: splits the stream by `<|start|>assistant to=X<|message|>` headers (X=user -> answer, else reasoning), robust to markers split across chunks; reuse it in any chat front end. Test: `.venv/bin/python -I tests/test_turns.py`.
- `meta/` - `layer_weights.json` is tracked (our own measurement). `config.json`, `model.safetensors.index.json`, `tensors.json` (name -> [bytes, shape, dtype, shard]) are LOCAL ONLY (gitignored: copies/derivatives of the model's own files, covered by the Muse Model License); regenerate with `hf download` (config/index) and `HfApi().get_safetensors_metadata`.
- `README.md`, `LICENSE` (MIT, code of this repo), `THIRD_PARTY_NOTICES.md` (vendored mlx-lm file under MIT; the model is NOT redistributed and has its own license: must show "Built with Muse Spark Open" in docs/UI, follow the Usage Policy).
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
7. [done, branch `gpu-overhead`] **Slab cache** (replaces the global LRU of per-expert arrays): the per-token `mx.stack` of the 8 selected experts was a real GPU copy of ~176 MB/layer = 0.86 ms/layer (microbench; more than the expert math itself, 0.51 ms) - invisible to the old `stack_s` timer, which only timed lazy graph building. Now each layer owns a slab per tensor `[slots, *tail]` + per-layer LRU (expert -> slot); reads go straight into a slot, `gather_qmm` indexes slots in place, no copy. Prefill / more-distinct-experts-than-slots use a stacked "fallback" path (slab rows + temporary reads, nothing admitted). Decode 3.77 -> **4.46 tok/s** (equal split) -> **4.57** (layer-weighted split `meta/layer_weights.json`, ~distinct experts per layer squared); GPU time 23.6 -> 17 s per 199 tokens. Peak mem 106.2 GB (slabs are preallocated, so peak is deterministic).
   - Offline cache simulation on 3 sampled traces (3 x 800 decode tokens, topics: history essay / B-tree code / civics), capacity 4314: global LRU 87.7%, global LFU 88.1% (policy barely matters), per-layer LRU equal split 85.8%, weighted split 87.0%. **Capacity is what matters**: global LRU at 3000 slots 80%, 6000 -> 93%, 8000 -> 97%. Distinct experts used per layer over 2400 tokens: 182-252 of 256 (early layers ~250, i.e. almost everything gets used eventually).
   - Trace tooling: `cache.trace = []` records `(layer, expert ids)` per decode step; scratchpad scripts `record_trace.py` / `sim_cache.py` (not in repo).
8. [done] GPU profile of one decoder layer at decode (real weights, `profile_layer.py` in scratchpad): each separately-timed block reads ~150-190 us = the sync floor (152 us); net costs: experts ~365 us (= ~480 GB/s on 176 MB, i.e. memory-bandwidth bound), attention ~270 us (81 MB of weights ~170 us at that rate), router ~35, pre/post expert proj ~60, norms/gates ~10 each. Whole layer ~1.0 ms vs ~0.6 ms bandwidth floor -> kernel fusion / `mx.compile` / fused qkvg would gain at most ~10%; NOT worth it now. (Earlier "launch overhead dominates" hypothesis was wrong.)
   Prefetch re-sweep with slabs (decode tok/s): k=4 4.58, k=6 4.35, k=8 4.00, k=12 3.23. k=4 stays best: wider prefetch raises GPU time (16.9 -> 22.6 s; SSD traffic contends with the GPU for memory bandwidth) and demand wait never drops below ~17 s even at 0.5 misses/layer (each layer with any miss pays a ~1 ms SSD round trip regardless of how many experts).
   Output check after the slab change: first prompt ("capital of France", greedy, 24 tokens) is word-for-word identical to the first run.
9. [next] Ideas: (1) more slots = fewer misses (+1000 slots ~ +3-4 points hit rate): peak is 106.2 of 107.5 GB (GPU recommended working set); raising it needs `sudo sysctl iogpu.wired_limit_mb=...` (user's call, not persistent) - NOT done; (2) better predictor than "next router on this residual" (67% recall at top-8) to prefetch fewer, better experts; (3) prefill tuning (~7.5 tok/s, disk bound); (4) usable chat/server front end + long-conversation memory test (KV cache growth vs the ~1.3 GB headroom).

## Notes
- Stop tokens: config.json lists `eos_token_id` [200001 `<|end_of_text|>`, 200008 `<|eot|>`] but the tokenizer alone only knows 200001, so generation ran to `--max-tokens`. `run_stream.py` now passes the config list to the tokenizer loader (any new front end must do the same). The reasoning message ends with `<|eom|>` (200007), which is NOT a stop token: the user-facing answer (`<|start|>assistant to=user<|message|>...<|eot|>`) follows it.
- Keep temp files (benchmark scripts, test data) in the session scratchpad, not in the project; delete large test files.
- Downloaded model files are untrusted data; do not run code from the model dir. Use `python -I` when scripts read from untrusted dirs.
