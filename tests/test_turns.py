"""TurnParser: reasoning/answer split, independent of chunking. Run: .venv/bin/python -I tests/test_turns.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from turns import TurnParser  # noqa: E402


def run(text, step, in_header=True):
    p, out = TurnParser(in_header), []
    for i in range(0, len(text), step):
        out += p.feed(text[i:i + step])
    out += p.flush()
    merged = {"reasoning": "", "answer": ""}
    for ch, t in out:
        merged[ch] += t
    return merged


STREAM = (" to=self<|message|>We should say hi.<|eom|><|start|>assistant to=user<|message|>"
          "Hello <|world|> and a < b!<|eot|>")
for step in (1, 2, 3, 5, 7, 1000):  # markers split across chunks at every offset
    r = run(STREAM, step)
    assert r == {"reasoning": "We should say hi.", "answer": "Hello <|world|> and a < b!"}, (step, r)

assert run("plain text", 4, in_header=False) == {"reasoning": "", "answer": "plain text"}
r = run(" to=commentary<|message|>working<|eom|><|start|>assistant to=user<|message|>done<|eot|>", 3)
assert r == {"reasoning": "working", "answer": "done"}, r
assert run(" to=self<|message|>cut off mid", 4) == {"reasoning": "cut off mid", "answer": ""}

# edge cases
assert run("", 3) == {"reasoning": "", "answer": ""}
assert run("<|eot|>", 2) == {"reasoning": "", "answer": ""}, "an end marker alone shows nothing"
assert run(" to=user<|message|>a<|eot|><|eot|><|eom|>b", 3) == {"reasoning": "", "answer": "ab"}, "repeated end markers"
assert run(" to=user<|message|>half a marker <|me", 4) == {"reasoning": "", "answer": "half a marker <|me"}, "flushed at the end"
assert run(" to=user<|message|>x < y <| z |> w", 2) == {"reasoning": "", "answer": "x < y <| z |> w"}, "near-markers are text"
assert run("<|message|>no header", 5) == {"reasoning": "", "answer": "no header"}, "empty header = the user"
assert run(" to=self<|message|>a<|start|>assistant to=user<|message|>b<|start|> to=self<|message|>c", 4) == {
    "reasoning": "ac", "answer": "b"}, "channels switch back and forth"
assert run(" to=  user  <|message|>spaces", 3)["answer"] == "spaces", "spaces around the recipient"
assert run(" to=USER<|message|>case", 3)["reasoning"] == "case", "the recipient is case-sensitive: anything but 'user' is reasoning"

# the split must never change the result: random streams, random chunk boundaries
import random
rnd = random.Random(1)
PIECES = ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>", " to=user", " to=self", "assistant",
          "word ", "x", "<", "<|", "|>", "<|mess", "a < b", "\n", "é", " "]
for _ in range(300):
    text = "".join(rnd.choice(PIECES) for _ in range(rnd.randint(1, 25)))
    for in_header in (True, False):
        want = run(text, 10**6, in_header)
        got = [run(text, k, in_header) for k in (1, 2, 3, 5, 8)]
        assert all(g == want for g in got), (text, in_header, want, got)
        p, out = TurnParser(in_header), []
        i = 0
        while i < len(text):
            n = rnd.randint(1, 7)
            out += p.feed(text[i:i + n])
            i += n
        out += p.flush()
        merged = {"reasoning": "", "answer": ""}
        for ch, t in out:
            merged[ch] += t
        assert merged == want, (text, in_header, merged, want)
print("ALL PASS")
