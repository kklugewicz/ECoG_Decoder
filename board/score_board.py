"""Score the board's INT8 predictions with decoder/protocol.py (run on the
host) and compare with the same .pte run on the x86 host
(results/main/int8_ptq_results.csv).

    python board/score_board.py                          # your run, board/results
    python board/score_board.py --results results/board  # the paper's run

Writes board_int8_scores.csv into the results folder.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "decoder"))
import protocol as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"),
                    help="folder with preds_<sub>_s<seed>.npy from board_bench.py accuracy")
    ap.add_argument("--host-csv", default=os.path.join(REPO, "results", "main",
                                                       "int8_ptq_results.csv"))
    ap.add_argument("--data-dir", default=P.DATA_DIR)
    a = ap.parse_args()

    host = {(r["subject"], int(r["seed"])): float(r["official_r"])
            for r in csv.DictReader(open(a.host_csv))}
    rows, diffs = [], []
    for sub in P.SUBJECTS:
        d = P.load_prep(sub, a.data_dir)
        lo = P.split_bounds(d)[2][0]           # first test sample at 100 Hz
        for seed in range(5):
            pred = np.load(os.path.join(a.results, f"preds_{sub}_s{seed}.npy")).astype(np.float64)
            s = P.official_score(pred, d["dg_raw"], lo, P.DECIM * lo, d["dg_raw"].shape[0],
                                 mode="zoh", dg_mean=d["dg_mean"].numpy().ravel(),
                                 dg_std=d["dg_std"].numpy().ravel())
            h = host[(sub, seed)]
            diffs.append(s["official_r"] - h)
            rows.append(dict(subject=sub, seed=seed, board_official_r=s["official_r"],
                             host_official_r=h, diff=s["official_r"] - h,
                             coverage=s["coverage"],
                             **{f"r_{P.FINGER_NAMES[i]}": float(s["r"][i]) for i in range(5)}))
            print(f"{sub} s{seed}: board {s['official_r']:.6f}  host {h:.6f}  "
                  f"diff {s['official_r'] - h:+.2e}  coverage {s['coverage']:.3f}")
        b = [r["board_official_r"] for r in rows if r["subject"] == sub]
        print(f"  {sub} board mean {np.mean(b):.4f} +- {np.std(b, ddof=1):.4f}")
    print(f"\nboard mean over 15: {np.mean([r['board_official_r'] for r in rows]):.4f} | "
          f"max |board - host| {np.max(np.abs(diffs)):.2e}")
    with open(os.path.join(a.results, "board_int8_scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
