"""Chat with Muse Spark Open 816B-A42B, streaming experts from SSD.

  .venv/bin/python chat.py [--reasoning-effort low|medium|high|xhigh] [--max-context 16384]

The conversation keeps its KV cache between turns, so only the new message is processed each time.
Answers go to stdout; reasoning (dimmed), timings and statistics go to stderr.
"""
import argparse
import readline  # noqa: F401  (line editing and history for input())
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402
from chat_session import ChatSession, ContextFull  # noqa: E402

HELP = """commands:  /reset            start a new conversation
           /effort LEVEL      low|medium|high|xhigh (starts a new conversation)
           /context           tokens used / limit
           /stats             expert-cache statistics and peak memory
           /quit              leave (or Ctrl-D)
           \"\"\"               start/end a multi-line message
Ctrl-C while the model is answering stops the answer; the conversation continues."""

ap = argparse.ArgumentParser()
cc.add_model_args(ap, ROOT)
cc.add_sampling_args(ap, ROOT / cc.DEFAULT_MODEL, greedy_default=False)
ap.add_argument("--max-tokens", type=int, default=2048, help="most tokens per reply, reasoning included")
ap.add_argument("--max-context", type=int, default=16384,
                help="most tokens in the conversation (the KV cache costs ~64 KB per token)")
ap.add_argument("--small-step", type=int, default=8,
                help="short messages are fed to the model this many tokens at a time (<= 8 keeps the cached decode path)")
ap.add_argument("--small-below", type=int, default=48,
                help="messages up to this many tokens use --small-step; longer ones are prefilled in big chunks")
args = ap.parse_args()

model, cache, tok = cc.load_all(args, ROOT)
session = ChatSession(model, tok, reasoning_effort=args.reasoning_effort, sampler=cc.make_sampler_from(args),
                      max_context=args.max_context, small_step=args.small_step, small_below=args.small_below)
printer = cc.Printer(args.hide_reasoning)


def err(msg):
    print(msg, file=sys.stderr, flush=True)


def prompt(text):
    """input() writes its prompt to stdout when stdin is not a terminal; keep stdout for the answers only."""
    if sys.stdin.isatty():
        return input(text)
    sys.stderr.write(text)
    sys.stderr.flush()
    return input()


def read_message():
    line = prompt("\nyou> ")
    if line.strip() == '"""':
        lines = []
        while (more := prompt("...  ")).strip() != '"""':
            lines.append(more)
        return "\n".join(lines)
    return line


err("Muse Spark Open 816B-A42B (3-bit), routed experts streamed from SSD. Built with Muse Spark Open.")
err(f"reasoning effort {session.effort}, up to {args.max_context} tokens of context. /help for commands.")
while True:
    try:
        text = read_message().strip()
    except EOFError:
        break
    except KeyboardInterrupt:
        err("\n(Ctrl-D or /quit to leave)")
        continue
    if not text:
        continue
    if text.startswith("/"):
        cmd, _, rest = text.partition(" ")
        if cmd in ("/quit", "/exit"):
            break
        elif cmd == "/reset":
            session.reset()
            err("new conversation")
        elif cmd == "/effort":
            if rest.strip() in ("low", "medium", "high", "xhigh"):
                session.reset(rest.strip())
                err(f"reasoning effort {session.effort}; new conversation")
            else:
                err("usage: /effort low|medium|high|xhigh")
        elif cmd == "/context":
            err(f"{session.context} / {args.max_context} tokens")
        elif cmd == "/stats":
            err(cache.summary())
            err(f"peak memory {mx.get_peak_memory() / 2**30:.1f} GB, active {mx.get_active_memory() / 2**30:.1f} GB")
        else:
            err(HELP)
        continue

    try:
        turn = session.send(text, on_chunks=printer.write, max_tokens=args.max_tokens)
    except ContextFull as e:
        err(f"[context full: {e}. /reset starts a new conversation]")
        continue
    printer.close()
    notes = [f"ctx {turn.context}/{args.max_context}",
             f"fed {turn.prompt_tokens} tok, first token after {turn.prompt_s:.1f}s",
             f"{turn.generation_tokens} tok @ {turn.generation_tps:.1f} tok/s",
             f"peak {mx.get_peak_memory() / 2**30:.1f} GB"]
    if turn.rebuilt:
        notes.append("cache rebuilt from the messages")
    if turn.finish == "length":
        notes.append(f"cut off after {args.max_tokens} tokens: raise --max-tokens or use /effort low")
    elif turn.finish == "interrupted":
        notes.append("interrupted")
    err("[" + " | ".join(notes) + "]")
