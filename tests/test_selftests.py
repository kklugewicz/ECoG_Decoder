"""The self-tests built into the decoder scripts. No data needed, a few
seconds each: causality, streaming equivalence, filter alignment, the
competition metric (a perfect decoder must score r = 1), ablation bookkeeping."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    (["decoder/model.py"], "All model self-tests passed."),
    (["decoder/protocol.py"], "All protocol self-tests passed."),
    (["decoder/prep.py", "--selftest"], "All prep self-tests passed."),
    (["decoder/ablate.py", "--selftest"], "All ablate self-tests passed."),
]


@pytest.mark.parametrize("args,expect", CASES, ids=[c[0][0] for c in CASES])
def test_selftest(args, expect):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert expect in r.stdout
