"""Dump each subject's model-ready test input for the board (run on the host).

x is the causal running-z-scored feature stream (protocol.normalised_features),
sliced from HIST samples before the test start so even the largest exported
window has full history for the first test label. Saved as float32 .npy plus a
small JSON so the board knows where the test region starts.

    python board/dump_inputs.py        # data/prep -> board/inputs (~100 MB)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "decoder"))
import protocol as P  # noqa: E402

HIST = 400                     # > RF + 50 - 1 = 333, the largest export window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=P.DATA_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "inputs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for sub in P.SUBJECTS:
        d = P.load_prep(sub, a.data_dir)
        x = P.normalised_features(d)
        lo, hi = P.split_bounds(d)[2]
        seg = x[:, lo - HIST:hi].numpy().astype(np.float32)
        np.save(os.path.join(a.out, f"{sub}_x.npy"), seg)
        meta[sub] = dict(offset=HIST, n_test=hi - lo, rows=seg.shape[0], test_lo=lo)
        print(sub, seg.shape, f"{seg.nbytes / 1e6:.1f} MB")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
