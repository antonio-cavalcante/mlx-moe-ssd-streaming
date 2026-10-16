# mlx-moe-ssd-streaming

Run a Mixture-of-Experts model that is **bigger than your RAM** on a Mac, by keeping only
the always-needed weights in memory and streaming the routed experts from the SSD on
demand.

Built with Muse Spark Open.

It was written to run `mlx-community/Muse-Spark-Open-816B-A42B-3bit` (816B parameters, 42B
active per token, 359 GB on disk) on a 128 GB MacBook. It is a small layer on top of
[MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm),
not a fork of either.

## Results

Apple M5 Max, 128 GB unified memory, internal SSD, macOS. One 200-token generation from a
173-token prompt (`run_stream.py`, defaults):

| | |
|---|---|
| Generation | **~4.6 tokens/s** |
| Prompt processing | ~7.5 tokens/s (disk-bound) |
| Resident dense weights | 9.0 GB |
| Expert cache (slabs) | 87.8 GB = 4,280 of 15,872 experts (27%) |
| Peak memory | 98.9 GB (GPU recommended working set: 107.5 GB) |
| Decode cache hit rate | ~89% |

These are single-machine, short-run numbers, not a benchmark suite. Speed depends on your
SSD, memory bandwidth and the text being generated. `CLAUDE.md` has the full development
log: what was measured, what helped, and what did not.

## How it works

- **Dense weights stay in RAM** (attention, embeddings, routers, pre/post-expert
  projections, norms): 9 GB. The 349 GB of routed experts stay on disk.
- **Per layer, the router runs first**, then only the 8 selected experts (of 256) are read
  from the original safetensors shards by byte offset - no conversion or repacking.
- **Slab cache.** Each layer owns a slab of expert slots (one array per expert tensor) with
  its own LRU. Reads go straight into a slot (writable `memoryview` of the MLX buffer, in
  parallel 1 MB chunks, `F_NOCACHE`), and the GPU reads experts in place through
  `gather_qmm` slot indices - no per-token copy of expert weights. Slab sizes follow
  `meta/layer_weights.json` (layers differ in how many experts they use).
- **Prefetch.** While layer *l* computes, the experts layer *l+1* will probably need
  (its router applied to layer *l*'s residual stream) are read in the background
  (`--prefetch-k`, default 4; about 67% recall at top-8).
- **Prefill** (many tokens, nearly every expert needed) reads each group of experts into
  temporary buffers and never evicts the decode working set.

## Units

Sizes printed by this repo (and quoted here) are GB = 2^30 bytes (GiB), the unit the GPU limit is reported in.
mlx-lm's own `peak_memory` field is in decimal GB (1e9 bytes), about 7% larger.

## Requirements

- Apple Silicon Mac with a lot of unified memory (developed on 128 GB) and a fast internal
  SSD. The streaming path uses macOS/Metal specifics (`F_NOCACHE`, MLX GPU).
- ~360 GB free disk space for the model.
- Python 3.12 (3.9 is too old for current mlx-lm), [`uv`](https://github.com/astral-sh/uv).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    "git+https://github.com/ml-explore/mlx-lm.git@main" huggingface_hub

# ~359 GB; resumable. By downloading the model you accept its license and usage policy.
.venv/bin/hf download mlx-community/Muse-Spark-Open-816B-A42B-3bit \
    --local-dir models/Muse-Spark-Open-816B-A42B-3bit
```

Tested with mlx 0.32.4 and mlx-lm 0.32.2 (from `main`).

## Run

```bash
.venv/bin/python run_stream.py --prompt "Explain mixture-of-experts routing." --max-tokens 200
```

Useful flags: `--cache-gb` (expert slab size; default: the GPU working-set limit minus the
resident weights minus `--reserve-gb`), `--reserve-gb` (headroom for the KV cache and
temporaries, default 10), `--prefetch-k`, `--threads`, `--layer-weights equal`.
The model is a reasoning model: it thinks before it answers, and generation stops by itself when
the answer is done. The answer goes to stdout; the reasoning (dimmed on a terminal) and the
statistics go to stderr, so `> answer.txt` keeps only the answer. `--reasoning-effort low|medium|high|xhigh`
sets how long it thinks (default `medium`; the chat template's own default is `xhigh`), `--hide-reasoning` drops it, and
`--show-markup` prints the raw token stream. Output is greedy by default; `--temp`, `--top-k` and `--top-p` sample.

Lower `--cache-gb` if you need more free memory; it trades speed for headroom.

## Chat

```bash
.venv/bin/python chat.py
```

A multi-turn conversation that keeps its KV cache between turns, so each message only
processes its own new tokens. Commands: `/reset`, `/effort low|medium|high|xhigh`, `/context`,
`/stats`, `/quit`, and `"""` to start and end a multi-line message; Ctrl-C stops an answer.
Sampling defaults to the model's own recommendation (`generation_config.json`: temperature 1,
top-k 64). `--max-context` (default 16384 tokens) limits the conversation; the KV cache costs
about 64 KB per token.

Measured on the M5 Max (reasoning effort low): an 11,000-token conversation built from three
pasted source files peaked at 100.2 GB of the 107.5 GB limit (about 7 GB of headroom), decoded at 3.6-4.3 tok/s, and still
recalled the first file. A big prefill runs at about 50-70 tok/s (4,500 tokens in 70-90 s); the first
message of a session pays about 25 s for the system prompt. Short follow-up messages are fed in
groups of 8 tokens through the cached decode path, which beat one big prefill chunk (6.2 s vs 8.9 s
to the first token for a 35-token message).

The sliding-window attention layers cannot be rewound once they wrap, so the cache is append-only:
the model's own reasoning from earlier turns stays in the context. If the cache state is ever in
doubt (for example Ctrl-C in the middle of a step) it is rebuilt from the stored messages.

## Tests

```bash
.venv/bin/python -I tests/test_tiny.py
.venv/bin/python -I tests/test_turns.py
.venv/bin/python -I tests/test_chat.py
```

Builds a tiny 3-bit checkpoint (2 shards, the real tensor names) and checks that the
streaming model matches the stock in-memory model for prefill and greedy decoding, across
slab sizes (larger than, equal to, and smaller than a token's expert count), eviction, and
prefetch. `test_turns.py` checks the reasoning/answer splitter. `test_chat.py` runs multi-turn sessions on a
tiny float32 checkpoint and checks that the incremental KV cache gives the same logits as a fresh
prefill of the same tokens (including a wrapped sliding window, a cut-short turn and a cache rebuild).
None of them needs the real model.

## Layout

| Path | What |
|---|---|
| `src/expert_stream.py` | the streaming layer: `ExpertStore`, `ExpertCache` (slabs + LRU + prefetch), `StreamingSwitchGLU`, `build_streaming_model` |
| `src/muse_spark_1975.py` | vendored model code (see below) |
| `src/turns.py` | splits the streamed text into reasoning and answer |
| `src/chat_session.py` | multi-turn conversation with a persistent KV cache |
| `src/cli_common.py` | arguments, model loading and output helpers shared by the two programs |
| `run_stream.py` | one-shot generation with timing and cache statistics |
| `chat.py` | interactive chat |
| `tests/test_tiny.py` | correctness test on a tiny synthetic checkpoint |
| `tests/test_turns.py` | unit test for the reasoning/answer splitter |
| `tests/test_chat.py` | multi-turn KV-cache correctness on a tiny checkpoint |
| `meta/layer_weights.json` | per-layer slab sizing, measured from sampled decode traces |
| `CLAUDE.md` | development log / project notes |

## Why a vendored model file

The mlx-community 3-bit checkpoint was converted with the version of the Muse Spark model
from mlx-lm pull request #1975 (tensor names like `model.layers.N.mlp.switch_mlp...`).
That PR was closed in favor of #1976, whose model code uses a different layout and cannot
load this checkpoint. `src/muse_spark_1975.py` is the #1975 file with only its imports
changed. If the checkpoint is reconverted for the merged layout, the vendored file can go.

## Limitations

- Developed and measured on one machine and one model. Other Macs and other MoE models are
  untested; the expert-offset logic assumes the stacked `switch_mlp` layout of this
  checkpoint.
- Peak memory is kept below the GPU working-set limit by `--reserve-gb`. Conversations were tested up
  to about 11,000 tokens (peak 100.2 of 107.5 GB); longer ones grow the KV cache by ~64 KB per token, so
  reduce `--cache-gb` or raise `--reserve-gb` if you raise `--max-context` a lot.
- Greedy and sampled output was checked for coherence on a handful of prompts, not
  evaluated for quality.

## License

MIT for the code in this repository - see [LICENSE](LICENSE). Third-party code and the
model's separate license: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This
project is not affiliated with Meta, Apple or the mlx-community.
