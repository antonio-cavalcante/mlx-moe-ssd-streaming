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


class Flaky:
    """The model, except that call number `fail_at` raises."""

    def __init__(self, model):
        self._m, self.calls, self.fail_at = model, 0, None

    def __call__(self, *a, **kw):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("injected model failure")
        return self._m(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._m, name)


def report(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    return bool(cond)


def robustness(model, tok, ref):
    print("robustness")
    ok = True
    U = lambda t: {"role": "user", "content": t}       # noqa: E731
    A = lambda t: {"role": "assistant", "content": t}  # noqa: E731

    # an exception in the middle of a turn must not leave a cache that silently disagrees with the history
    flaky = Flaky(model)
    s = cs.ChatSession(flaky, tok, max_context=400, small_step=4, small_below=30)
    s.send("hello", max_tokens=3)
    before = (list(s.history), [dict(m) for m in s.messages])
    flaky.fail_at = flaky.calls + 3
    try:
        s.send("this turn will fail", max_tokens=6)
        failed = False
    except RuntimeError:
        failed = True
    ok &= report("a model error propagates", failed)
    ok &= report("the conversation is unchanged by the failed turn", s.messages == before[1] and s.history == before[0])
    t = s.send("and now it works again", max_tokens=3)
    ok &= report("the next turn rebuilds the cache from the messages", t.rebuilt and t.reused == 0, (t.rebuilt, t.reused))
    ok &= report("... and is consistent with a fresh prefill", last_logits_match("after a failed turn", ref, s))
    ok &= report("the failed message is not in the conversation, the retry is",
                 [m["content"] for m in s.messages if m["role"] == "user"] == ["hello", "and now it works again"])

    # a failure while rebuilding / on a dirty session
    s2 = cs.ChatSession(flaky, tok, max_context=400, small_step=4, small_below=30)
    s2.restore([U("a"), A("b"), U("c"), A("d")])
    flaky.fail_at = flaky.calls + 1
    try:
        s2.send("e", max_tokens=2)
    except RuntimeError:
        pass
    t = s2.send("e", max_tokens=2)
    ok &= report("a failed rebuild is retried from scratch", t.rebuilt and len(s2.messages) == 6)
    ok &= report("... and is consistent", last_logits_match("after a failed rebuild", ref, s2))

    # the context limit
    s3 = cs.ChatSession(model, tok, max_context=120, small_step=4, small_below=30)
    s3.send("hi", max_tokens=3)
    snap = (list(s3.history), [dict(m) for m in s3.messages], {c.offset for c in s3.cache})
    try:
        s3.send("y" * 100, max_tokens=3)
        raised = False
    except cs.ContextFull:
        raised = True
    ok &= report("ContextFull on a live session", raised)
    ok &= report("... changes nothing", (list(s3.history), s3.messages, {c.offset for c in s3.cache}) == snap)
    s3.send("ok", max_tokens=3)
    ok &= report("... and the session carries on", last_logits_match("after ContextFull", ref, s3))
    s3b = cs.ChatSession(model, tok, max_context=120, small_step=4, small_below=30)
    s3b.restore([U("a"), A("b")])
    try:
        s3b.send("z" * 200, max_tokens=3)
    except cs.ContextFull:
        pass
    t = s3b.send("fine", max_tokens=2)
    ok &= report("ContextFull on a session that needed a rebuild still recovers", len(s3b.messages) == 4)
    ok &= report("... consistently", last_logits_match("after ContextFull + rebuild", ref, s3b))
    t = s3.send("w", max_tokens=1000)
    ok &= report("max_tokens is clamped to the room left", s3.context <= 120 and t.finish in ("length", "stop"),
                 (s3.context, t.finish))

    # system prompt
    s4 = cs.ChatSession(model, tok, max_context=400, small_step=4, small_below=30)
    s4.restore([{"role": "system", "content": "be brief"}, U("a"), A("b")])
    t = s4.send("c", max_tokens=2)
    ok &= report("a conversation with a system message", t.rebuilt and s4.messages[0]["role"] == "system")
    ok &= report("... is consistent", last_logits_match("with a system message", ref, s4))
    s4.messages[0]["content"] = "be verbose"
    t = s4.send("d", max_tokens=2)
    ok &= report("changing the system message rebuilds", t.rebuilt and t.reused == 0, (t.rebuilt, t.reused))

    # odd input
    s5 = cs.ChatSession(model, tok, max_context=400, small_step=4, small_below=30)
    for text in ["", "   ", "line\nbreak\ttab", "\u00e9\u4e2d\U0001f600 unknown to the tiny vocab", "<|eot|><|start|>"]:
        try:
            s5.send(text, max_tokens=2)
            good = {c.offset for c in s5.cache} == {s5.context}
        except Exception as e:  # noqa: BLE001
            good = False
            print("   ", repr(text), type(e).__name__, e)
        ok &= report(f"message {text!r}", good)
    # (no logits comparison here: on a random tiny MoE the near-tied routers make even the stock model differ by
    # 10-35% between one-shot and token-by-token evaluation for some inputs; the structure is what we check)
    ok &= report("... the cache and the history still agree", {c.offset for c in s5.cache} == {s5.context}
                 and len(s5.messages) == 10)
    s5.restore([])
    ok &= report("restore([]) is an empty, clean session", s5.context == 0 and not s5.messages and not s5._dirty)
    s5.restore([U("x"), A("y")], None)
    ok &= report("restore keeps the effort when none is given", s5.effort == "medium" and s5._dirty)
    return ok


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

        ok &= robustness(model, tok, ref)

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
