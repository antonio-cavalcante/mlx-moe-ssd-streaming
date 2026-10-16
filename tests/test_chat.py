"""ChatSession on a tiny synthetic checkpoint: the incremental KV cache must equal a fresh prefill.

Run: .venv/bin/python -I tests/test_chat.py   (no real model needed)
"""
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import chat_session as cs  # noqa: E402
import expert_stream as es  # noqa: E402
import muse_spark_1975 as ms  # noqa: E402
import test_tiny as tt  # noqa: E402

SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
TEMPLATE = (
    "{{ '<|begin_of_text|><|start|>system<|message|>effort ' ~ reasoning_effort ~ '<|eot|>' }}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}{{ '<|start|>user<|message|>' ~ m.content ~ '<|eot|>' }}"
    "{%- else -%}{{ '<|start|>assistant to=user<|message|>' ~ m.content ~ '<|eot|>' }}{%- endif -%}"
    "{%- endfor -%}{%- if add_generation_prompt -%}{{ '<|start|>assistant' }}{%- endif -%}")


def make_tokenizer():
    chars = [chr(c) for c in range(32, 127)]
    vocab = {t: i for i, t in enumerate(SPECIALS + chars)}
    t = Tokenizer(models.WordLevel(vocab, unk_token="?"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    t.decoder = decoders.Fuse()
    t.add_special_tokens(SPECIALS)
    hf = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|end_of_text|>", chat_template=TEMPLATE)
    return TokenizerWrapper(hf, eos_token_ids=[vocab["<|end_of_text|>"], vocab["<|eot|>"]])


def last_logits_match(name, ref, sess, probe=7):
    """Next-token logits after the session's cache vs a fresh prefill of the same token history."""
    got = sess.model(mx.array([[probe]]), cache=sess.cache)[0, -1]
    want = ref(mx.array([sess.history + [probe]]))[0, -1]
    mx.eval(got, want)
    spread = float((want - want.mean()).abs().max())
    rel = float(mx.abs(got - want).max()) / spread
    # float32 model: a correct cache differs by <1% of the logit spread, a wrong history by 70-100%
    print(f"  {'PASS' if rel < 0.05 else 'FAIL'}  {name}: relative diff {rel:.4f}")
    return rel < 0.05


def main():
    tmp = Path(tempfile.mkdtemp(prefix="chat_ckpt_"))
    ok = True
    try:
        ref = tt.make_checkpoint(tmp, mx.float32)
        tok = make_tokenizer()
        model, cache = es.build_streaming_model(tmp, ms, slots_per_layer=6, prefetch_k=4, verbose=False)
        sess = cs.ChatSession(model, tok, max_context=400, small_step=4, small_below=30, big_step=16)

        texts = ["hello there", "and now a much longer second message so that it takes the big prefill path",
                 "ok", "again"]
        for i, text in enumerate(texts):
            before = sess.context
            t = sess.send(text, max_tokens=7)
            print(f"  turn {i}: fed {t.prompt_tokens} tok, generated {t.generation_tokens} ({t.finish}), "
                  f"context {before} -> {t.context}")
            ok &= t.context == before + t.prompt_tokens + t.generation_tokens
            ok &= {c.offset for c in sess.cache} == {sess.context}
            ok &= not t.rebuilt
        ok &= len(sess.messages) == 2 * len(texts)
        assert sess.context > 8, "the sliding window (8) must have wrapped"
        hist = list(sess.history)
        # the first turn's tokens are exactly the rendered first prompt
        first = tok.encode(sess._render([{"role": "user", "content": texts[0]}], True), add_special_tokens=False)
        ok &= hist[:len(first)] == first
        print(f"  {'PASS' if hist[:len(first)] == first else 'FAIL'}  first turn tokens == the rendered prompt")
        ok &= last_logits_match(f"after {len(texts)} turns: cache == fresh prefill of the {len(hist)}-token history",
                                ref, sess)

        # a turn cut short by max_tokens is closed with <|eot|> before the next one
        s2 = cs.ChatSession(model, tok, max_context=400, small_step=4, small_below=30)
        t = s2.send("hello", max_tokens=3)
        if t.finish == "length":
            n = s2.context
            s2.send("again", max_tokens=3)
            ok &= s2.history[n] == s2.eot
            print(f"  {'PASS' if s2.history[n] == s2.eot else 'FAIL'}  cut-short turn is closed with <|eot|>")
        ok &= last_logits_match("cut-short flow: cache == fresh prefill", ref, s2)

        # unknown cache state -> rebuilt from the messages, still consistent
        s2._dirty = True
        t = s2.send("after a reset of the cache", max_tokens=4)
        ok &= t.rebuilt
        print(f"  {'PASS' if t.rebuilt else 'FAIL'}  dirty cache is rebuilt ({t.prompt_tokens} tokens re-fed)")
        ok &= last_logits_match("rebuilt: cache == fresh prefill", ref, s2)

        # cancelled after two tokens: the cache stays consistent and usable
        s4 = cs.ChatSession(model, tok, max_context=400, small_step=4, small_below=30)
        n = []
        t = s4.send("hello", on_chunks=None, max_tokens=20, cancel=lambda: n.append(1) or len(n) >= 2)
        ok &= t.finish == "interrupted" and t.generation_tokens == 2 and {c.offset for c in s4.cache} == {s4.context}
        print(f"  {'PASS' if t.finish == 'interrupted' else 'FAIL'}  cancel() stops after {t.generation_tokens} tokens, "
              f"cache consistent ({s4.context})")
        ok &= last_logits_match("after cancel: cache == fresh prefill", ref, s4)
        s4.send("go on", max_tokens=3)
        ok &= last_logits_match("turn after a cancel: cache == fresh prefill", ref, s4)

        # the system prompt changes (effort here; the date in the real template): rebuilt, not an error
        s4.effort = "high"
        t = s4.send("one more", max_tokens=3)
        ok &= t.rebuilt and t.reused == 0
        print(f"  {'PASS' if t.rebuilt and t.reused == 0 else 'FAIL'}  changed system prompt -> rebuilt from the messages")
        ok &= last_logits_match("after prefix change: cache == fresh prefill", ref, s4)

        # restore(): adopt a client's conversation, rebuild lazily
        s4.restore([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "low")
        t = s4.send("c", max_tokens=3)
        ok &= t.rebuilt and len(s4.messages) == 4 and s4.effort == "low"
        ok &= last_logits_match("after restore: cache == fresh prefill", ref, s4)

        # context limit
        s3 = cs.ChatSession(model, tok, max_context=40)
        try:
            s3.send("x" * 60)
            ok = False
            print("  FAIL  ContextFull not raised")
        except cs.ContextFull as e:
            print(f"  PASS  ContextFull: {e}")

        # reset keeps working
        s2.reset("low")
        ok &= s2.context == 0 and not s2.messages and s2.effort == "low"
        s2.send("fresh start", max_tokens=3)
        ok &= last_logits_match("after reset: cache == fresh prefill", ref, s2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
