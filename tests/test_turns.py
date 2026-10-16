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
print("ALL PASS")
