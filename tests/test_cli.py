"""cli_common: sampling defaults and the answer/reasoning printer.

Run: .venv/bin/python -I tests/test_cli.py
"""
import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import cli_common as cc  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def printed(hide, chunks_list):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        p = cc.Printer(hide)
        for chunks in chunks_list:
            p.write(chunks)
        p.close()
    return out.getvalue(), err.getvalue()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="cli_"))
    try:
        print("generation_config.json")
        check("missing file -> {}", cc.generation_defaults(tmp) == {})
        (tmp / "generation_config.json").write_text("{not json")
        check("invalid JSON -> {}", cc.generation_defaults(tmp) == {})
        (tmp / "generation_config.json").write_text("[1, 2]")
        check("not an object -> {}", cc.generation_defaults(tmp) == {})
        (tmp / "generation_config.json").write_text(json.dumps({"temperature": 0.7, "top_k": 20}))
        ap = argparse.ArgumentParser()
        cc.add_sampling_args(ap, tmp, greedy_default=False)
        a = ap.parse_args([])
        check("the model's settings are the defaults", (a.temp, a.top_k, a.top_p) == (0.7, 20, 1.0))
        ap = argparse.ArgumentParser()
        cc.add_sampling_args(ap, tmp, greedy_default=True)
        check("greedy_default overrides temperature only", ap.parse_args([]).temp == 0.0
              and ap.parse_args(["--temp", "0.3"]).temp == 0.3 and ap.parse_args([]).top_k == 20)
        ap = argparse.ArgumentParser()
        cc.add_sampling_args(ap, tmp / "nowhere", greedy_default=False)
        a = ap.parse_args([])
        check("without a file: temperature 1, no top-k/top-p", (a.temp, a.top_k, a.top_p) == (1.0, 0, 1.0))
        check("make_sampler_from works", callable(cc.make_sampler_from(a)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("Printer")
    out, err = printed(False, [[("reasoning", "think "), ("reasoning", "more")], [("answer", "Hi"), ("answer", "!")]])
    check("answer on stdout, reasoning on stderr", out == "Hi!\n" and err.strip().startswith("think more"), (out, err))
    out, err = printed(True, [[("reasoning", "secret")], [("answer", "Hi")]])
    check("--hide-reasoning: nothing from the reasoning anywhere", out == "Hi\n" and "secret" not in err)
    out, err = printed(False, [])
    check("an empty reply is just a newline", out == "\n" and err == "")
    out, err = printed(False, [[("reasoning", "r")], [("answer", "a")], [("reasoning", "r2")], [("answer", "b")]])
    check("channels can alternate", out == "ab\n" and "r" in err and "r2" in err)
    print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
