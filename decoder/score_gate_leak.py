"""
score_gate_leak.py — the second measurement the `no-gate` row needs.

A time-pooling gate (ECA / squeeze-excite) computes its channel weights from a
mean over the whole input window. Under dense supervision that means the
output at timestep t was trained with information from timesteps > t. The
exported decoder never has that information: it is invoked on a window that
ENDS at t and only its last output is used.

So every checkpoint is scored three ways, on the whole test recording, with
the competition metric:

    dense-chunk   predict_range over 20k-sample chunks (what score reports;
                  for a pooling gate this pools over 200 s of future)
    train-proto   tiled WINDOW-length windows, outputs read from the receptive
                  field onwards — exactly how training windows were supervised
    exported      one window ending at t per output, last step only — what the
                  deployed single-step graph computes. Strictly causal.

For a causal model all three are identical, which is the built-in check that
the scorer is right. For a pooling gate, train-proto > exported is the leak.

Usage:
    python decoder/score_gate_leak.py --tag abl_eca-gate --csv eca.csv
    python decoder/score_gate_leak.py --tag abl_reference --seeds 0 --csv check.csv
"""
import argparse
import csv
import os

import numpy as np
import torch

import protocol as P
from score import load_model


@torch.no_grad()
def predict_train_proto(model, x, lo, hi, window=P.WINDOW):
    rf = model.receptive_field
    emit = window - rf
    out = []
    for t in range(lo, hi, emit):
        n = min(emit, hi - t)
        seg = x[:, t - rf:t + n].unsqueeze(0)
        out.append(model(seg)[0, :, rf:rf + n].T)
    return torch.cat(out, 0)


@torch.no_grad()
def predict_exported(model, x, lo, hi, window=P.WINDOW, batch=128):
    out = []
    for t0 in range(lo, hi, batch):
        ts = range(t0, min(t0 + batch, hi))
        seg = torch.stack([x[:, t - window + 1:t + 1] for t in ts], 0)
        out.append(model(seg)[:, :, -1])
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=P.DATA_DIR)
    ap.add_argument("--ckpt-dir", default=P.CKPT_DIR)
    ap.add_argument("--subjects", nargs="*", default=P.SUBJECTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    here = a.ckpt_dir

    rows = []
    for sub in a.subjects:
        d = P.load_prep(sub, a.data_dir)
        x = P.normalised_features(d)
        lo, hi = P.split_bounds(d)[2]
        n_raw = d["dg_raw"].shape[0]
        for seed in a.seeds:
            ck = P.ckpt_path(here, sub, seed, a.tag)
            if not os.path.exists(ck):
                print(f"{sub} s{seed}: missing {os.path.basename(ck)}")
                continue
            model, _ = load_model(ck, int(d["n_electrodes"]), int(d["n_bands"]))
            preds = {
                "dense-chunk": P.predict_range(model, x, lo, hi),
                "train-proto": predict_train_proto(model, x, lo, hi),
                "exported": predict_exported(model, x, lo, hi),
            }
            res = {k: P.official_score(v.numpy().astype(np.float64), d["dg_raw"],
                                       lo, P.DECIM * lo, n_raw, mode="zoh")["official_r"]
                   for k, v in preds.items()}
            print(f"{sub} s{seed}: " + "  ".join(f"{k} {v:.4f}" for k, v in res.items())
                  + f"  | leak (train-proto - exported) {res['train-proto'] - res['exported']:+.4f}",
                  flush=True)
            rows.append(dict(subject=sub, seed=seed, **res))

    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "seed", "dense-chunk",
                                           "train-proto", "exported"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
