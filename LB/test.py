from __future__ import print_function

import io
import os
import signal
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_DIR = os.path.join(ROOT, "LB", "tests")
sys.path.insert(0, os.path.join(ROOT, "LB"))

from LBi import run

_RUN_TIMEOUT_S = 240


def _timeout_handler(signum, frame):
    raise TimeoutError(
        "timed out after {}s waiting for test output".format(_RUN_TIMEOUT_S)
    )


def clean_data(data):
    """Strip blank lines and trim each line (like test_compiler.py)."""
    return [
        line
        for line in (s.strip() for s in data.strip().split("\n"))
        if len(line) > 0
    ]


def discover_cases():
    cases = []
    for name in sorted(os.listdir(SUITE_DIR)):
        if not name.endswith(".LB"):
            continue
        source = os.path.join(SUITE_DIR, name)
        oracle = source + ".out"
        if not os.path.isfile(oracle):
            continue
        stdin_path = source + ".in"
        if not os.path.isfile(stdin_path):
            stdin_path = None
        cases.append(pytest.param(source, oracle, stdin_path, id=name[:-3]))
    return cases


@pytest.mark.parametrize("source_path,oracle_path,stdin_path", discover_cases())
def test_lb(source_path, oracle_path, stdin_path, capsys, monkeypatch):

    with open(oracle_path) as f:
        expected = clean_data(f.read())

    with open(source_path) as f:
        source = f.read()

    if stdin_path is not None:
        with open(stdin_path) as f:
            stdin_text = f.read()
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_RUN_TIMEOUT_S)
    try:
        run(source)
        got = clean_data(capsys.readouterr().out)
    finally:
        signal.alarm(0)
    assert got == expected, (
        "oracle mismatch for {}\nexpected: {!r}\ngot:      {!r}\n"
        "(check the interpreter first; if it looks right, the test or "
        "oracle may be wrong)"
    ).format(os.path.basename(source_path), expected, got)
