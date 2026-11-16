"""End to end on the real data: the shipped models reproduce the paper's
numbers. Runs only after `python decoder/prep.py` (about 2 minutes).

  * FP32: decoder/score.py on all 15 checkpoints matches
    results/main/main_results.csv.
  * INT8: the deployed PTQ model of each subject (seed 0) matches
    results/main/int8_ptq_results.csv (needs executorch).
"""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

import protocol as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_DATA = all(os.path.exists(P.prep_path(s)) for s in P.SUBJECTS)
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="run python decoder/prep.py first")


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_fp32_scores_match_paper(tmp_path):
    out = tmp_path / "scores.csv"
    r = subprocess.run([sys.executable, "decoder/score.py", "--tag", "paper",
                        "--csv", str(out)], cwd=ROOT, capture_output=True,
                       text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    paper = {(x["subject"], x["seed"], x["row"]): x
             for x in read(os.path.join(ROOT, "results", "main", "main_results.csv"))}
    mine = {(x["subject"], x["seed"], x["row"]): x for x in read(out)}
    assert set(mine) == set(paper)
    worst = max(abs(float(mine[k][c]) - float(paper[k][c]))
                for k in paper for c in paper[k] if c not in ("subject", "seed", "row"))
    assert worst < 1e-5, f"largest difference {worst:.2e}"
    official = [float(x["official_r"]) for x in mine.values() if x["row"] == "official"]
    assert round(float(np.mean(official)), 3) == 0.533


@pytest.mark.parametrize("sub", P.SUBJECTS)
def test_int8_scores_match_paper(sub):
    pytest.importorskip("executorch.runtime")
    from export_int8 import pte_predict_region
    d = P.load_prep(sub)
    x = P.normalised_features(d)
    lo, hi = P.split_bounds(d)[2]
    pte = P.pte_path(P.CKPT_DIR, sub, 0, "paper_ptq")
    pred, _ = pte_predict_region(pte, x, lo, hi, 284 + 10 - 1, 10)
    s = P.official_score(pred, d["dg_raw"], lo, P.DECIM * lo, d["dg_raw"].shape[0],
                         mode="zoh", dg_mean=d["dg_mean"].numpy().ravel(),
                         dg_std=d["dg_std"].numpy().ravel())
    paper = {(x["subject"], int(x["seed"])): float(x["official_r"])
             for x in read(os.path.join(ROOT, "results", "main", "int8_ptq_results.csv"))}
    assert abs(s["official_r"] - paper[(sub, 0)]) < 1e-4
