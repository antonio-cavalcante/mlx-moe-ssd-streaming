#!/bin/sh
# Every test, none of them needs the real model. Usage: tests/run_all.sh
cd "$(dirname "$0")/.." || exit 1
status=0
for t in turns cli store tiny chat server lan; do
    printf '%-8s ' "$t"
    if out=$(.venv/bin/python -I "tests/test_$t.py" 2>&1); then
        echo "ok ($(printf '%s\n' "$out" | grep -c '  PASS') checks)"
    else
        echo "FAILED"
        printf '%s\n' "$out" | grep -v '  PASS' | tail -20
        status=1
    fi
done
exit $status
