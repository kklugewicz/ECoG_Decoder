"""
score.py — the number that goes in the paper, plus the ladder that
explains it.

THE HEADLINE ROW
----------------
`official` is desc_4.pdf section V, implemented literally: a 200,000 x 5
prediction at 1000 Hz for the whole test recording, correlated against the
stored `test_dg`, averaged over fingers 1,2,3,5. It is directly comparable to
the competition leaderboard and to every ds4 paper.

[FIX-9] WHICH NUMBER IS THE HEADLINE. If the contribution is a deployable
INT8 decoder, then the INT8 score is the result and FP32 is a reference row.
v3.0 published the FP32 mean and reported the quantised scores only in the
export log, where the mean was ~0.02 lower. `--int8-csv` ingests the per
(subject, seed) INT8 scores that `export_int8.py` writes and puts them in the
same table, so the two cannot be quoted selectively.

THE LADDER
----------
Each row changes exactly one thing, so the deltas are attributable.

  official        dense 100 Hz emission, zero-order hold to 1 kHz, RAW target,
                  whole recording.                                  <- publish
  official-linear same, linear interpolation between the 100 Hz samples.
                  Reads 10 ms ahead. Fine for a submission, not for a latency
                  claim.
  emit-20Hz       the SAME predictions, but only every 5th one is emitted and
                  held for 50 ms — i.e. what a sliding-window decoder that
                  reads one label per forward pass produces. The gap to
                  `official` is the output-rate refund, measured rather than
                  argued.
  smoothed-target the processed target instead of the raw glove trace, scored
                  at 100 Hz. An internal development metric. It is not
                  comparable to anything published, and the gap to `official`
                  is the price of the 25 Hz dataglove and its stair-step
                  upsampling.
  ensemble        mean prediction across seeds, scored as `official`. One
                  model per seed, so N x the compute. Report separately.

[FIX-10] VALIDATION -> TEST DIAGNOSTIC. The per-seed validation score is
printed alongside the test score and their rank correlation is reported. If
selection is uninformative on a recording this short, that is a finding about
ds4 and it belongs in the paper rather than in a reviewer's rebuttal.

Usage:
    python decoder/score.py
    python decoder/score.py --subjects sub1 --seeds 0
    python decoder/score.py --int8-csv int8_results.csv --latex paper/
    python decoder/score.py --csv results.csv
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch

from model import build_model, mac_report, deployment_cost
import protocol as P

ROWS = ["official", "official-linear", "emit-20Hz", "smoothed-target"]

# Published ds4 reference points. All non-causal, all offline, all evaluated
# on the same public test recording.
REFERENCES = [
    ("Liang \\& Bougrain 2012 (winner)", [0.45, 0.39, 0.59], "non-causal"),
    ("Flamary \\& Rakotomamonjy 2012",   [0.48, 0.24, 0.56], "non-causal"),
    ("Xie et al. 2018 (LSTM)",           [0.56, 0.41, 0.58], "non-causal"),
    ("Yao et al. 2022 (LightGBM)",       [0.52, 0.47, 0.61], "non-causal"),
    ("FingerFlex 2022 (U-Net)",          [0.66, 0.62, 0.74], "non-causal"),
]


def load_model(ckpt_path, n_el, n_bands):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("n_electrodes") not in (None, n_el):
        raise ValueError(f"{ckpt_path}: trained on {ck['n_electrodes']} electrodes, "
                         f"data has {n_el}. Wrong prep directory?")
    if ck.get("n_bands") not in (None, n_bands):
        raise ValueError(f"{ckpt_path}: trained on {ck['n_bands']} bands, "
                         f"data has {n_bands}.")
    # [FIX-7] restore architecture deltas recorded at training time
    m = build_model(n_el, n_bands=n_bands, preset=ck.get("preset", "tiny"),
                    **ck.get("trunk_kw", {}))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def decimate_emission(pred, factor=5):
    """[FIX-11] Keep every `factor`-th prediction and hold it, simulating a
    decoder that emits one label per sliding window. (v3.0 had two dead lines
    here that allocated a copy and an index array it then discarded.)"""
    idx = np.arange(len(pred))
    return pred[(idx // factor) * factor]


def score_all(pred, d, region, dg_mean, dg_std):
    lo, hi = region
    raw_lo, raw_hi = P.DECIM * lo, d["dg_raw"].shape[0]
    rows = {}
    rows["official"] = P.official_score(pred, d["dg_raw"], lo, raw_lo, raw_hi,
                                        mode="zoh", dg_mean=dg_mean, dg_std=dg_std)
    rows["official-linear"] = P.official_score(pred, d["dg_raw"], lo, raw_lo, raw_hi,
                                               mode="linear", dg_mean=dg_mean,
                                               dg_std=dg_std)
    rows["emit-20Hz"] = P.official_score(decimate_emission(pred, 5), d["dg_raw"],
                                         lo, raw_lo, raw_hi, mode="zoh",
                                         dg_mean=dg_mean, dg_std=dg_std)
    # Against the processed target, at 100 Hz, no upsampling involved.
    tgt = d["dg"][lo:hi].numpy().astype(np.float64)
    n = min(len(tgt), len(pred))
    rows["smoothed-target"] = P.official_score(pred[:n], tgt[:n], 0, 0, n, decim=1,
                                               mode="zoh")
    return rows


def read_int8_csv(path):
    """(subject, seed) -> official r, as written by export_int8.py."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["subject"], int(row["seed"]))] = float(row["official_r"])
    return out


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# ======================================================================
# LaTeX emission — the paper reads these files, so it can never disagree
# with the run that produced them.
# ======================================================================
def write_latex(outdir, per_subject, agg, int8_per_subject, int8_agg,
                footprint, refund, val_test, n_seeds):
    os.makedirs(outdir, exist_ok=True)

    def ms(v):
        m, s = P.mean_std(v)
        return f"{m:.3f} $\\pm$ {s:.3f}" if len(v) > 1 else f"{m:.3f}"

    subs = sorted({s for s, r in per_subject if r == "official"})

    # --- main results ------------------------------------------------
    with open(os.path.join(outdir, "tab_results.tex"), "w") as f:
        f.write("% AUTO-GENERATED by score.py --latex. Do not edit.\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Decoder & Sub.\\,1 & Sub.\\,2 & Sub.\\,3 & Mean \\\\\n\\midrule\n")
        for name, vals, kind in REFERENCES:
            f.write(f"{name} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} "
                    f"& {np.mean(vals):.2f} \\\\\n")
        f.write("\\midrule\n")
        row = " & ".join(ms(per_subject.get((s, "official"), [])) for s in subs)
        f.write(f"\\textbf{{This work, FP32}} & {row} & "
                f"{ms(agg.get('official', []))} \\\\\n")
        if int8_agg:
            row = " & ".join(ms(int8_per_subject.get(s, [])) for s in subs)
            f.write(f"\\textbf{{This work, INT8 (deployed)}} & {row} & "
                    f"{ms(int8_agg)} \\\\\n")
        if agg.get("ensemble"):
            row = " & ".join(f"{per_subject.get((s, 'ensemble'), [float('nan')])[0]:.3f}"
                             for s in subs)
            f.write(f"\\quad + {n_seeds}-seed ensemble (FP32) & {row} & "
                    f"{P.mean_std(agg['ensemble'])[0]:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # --- protocol ladder ---------------------------------------------
    blurb = {
        "official": "Competition criterion (published)",
        "official-linear": "\\quad + linear interpolation (reads 10\\,ms ahead)",
        "emit-20Hz": "\\quad one label per window (20\\,Hz emission)",
        "smoothed-target": "\\quad vs.\\ smoothed target at 100\\,Hz (internal)",
        "ensemble": f"\\quad {n_seeds}-seed ensemble",
    }
    with open(os.path.join(outdir, "tab_ladder.tex"), "w") as f:
        f.write("% AUTO-GENERATED by score.py --latex. Do not edit.\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Evaluation protocol & Mean $r$ & $\\Delta$ vs.\\ published \\\\\n")
        f.write("\\midrule\n")
        base = P.mean_std(agg["official"])[0] if agg.get("official") else float("nan")
        for r in ROWS + ["ensemble"]:
            if not agg.get(r):
                continue
            m, s = P.mean_std(agg[r])
            d = "---" if r == "official" else f"{m - base:+.4f}"
            f.write(f"{blurb[r]} & {m:.4f} $\\pm$ {s:.4f} & {d} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # --- numbers the prose interpolates ------------------------------
    with open(os.path.join(outdir, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by score.py --latex. Do not edit.\n")

        def cmd(name, val):
            f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")

        cmd("NumSeeds", n_seeds)
        if agg.get("official"):
            m, s = P.mean_std(agg["official"])
            cmd("FPmean", f"{m:.3f}")
            cmd("FPstd", f"{s:.3f}")
        if int8_agg:
            m, s = P.mean_std(int8_agg)
            cmd("INTmean", f"{m:.3f}")
            cmd("INTstd", f"{s:.3f}")
            if agg.get("official"):
                cmd("QuantGap", f"{P.mean_std(agg['official'])[0] - m:.3f}")
        if not np.isnan(refund):
            cmd("RateRefund", f"{refund:+.4f}")
        if not np.isnan(val_test):
            cmd("ValTestRho", f"{val_test:.2f}")
        if footprint:
            f0 = next(iter(footprint.values()))
            cmd("Params", f"{f0['params']:,}")
            cmd("KMACcached", f"{f0['kmac']:.1f}")
            cmd("MMACcached", f"{f0['mmac_s']:.2f}")
            cmd("MMACstateless", f"{f0['mmac_s_sl']:.1f}")
            if "pte_kb" in f0:
                cmd("PteKB", f"{f0['pte_kb']:.0f}")
    print(f"\nwrote tab_results.tex, tab_ladder.tex, numbers.tex to {outdir}/")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=P.DATA_DIR)
    ap.add_argument("--ckpt-dir", default=P.CKPT_DIR,
                    help="where the checkpoints (and any .pte files) are")
    ap.add_argument("--subjects", nargs="*", default=P.SUBJECTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="flexnet")
    ap.add_argument("--pte-tag", default=None,
                    help="prefix of the .pte files whose size is reported "
                         "(default: --tag); the shipped INT8 models are paper_ptq")
    ap.add_argument("--int8", action="store_true", help="report .pte sizes")
    ap.add_argument("--int8-csv", default="int8_results.csv",
                    help="per (subject,seed) INT8 scores from export_int8.py")
    ap.add_argument("--emit", type=int, default=25,
                    help="labels per .pte invocation, for the footprint table")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ensemble-csv", default=None,
                    help="also write the per-subject 5-seed ensemble scores here")
    ap.add_argument("--latex", default=None, help="directory for .tex tables")
    a = ap.parse_args()
    script_dir = a.ckpt_dir
    int8 = read_int8_csv(a.int8_csv)

    print("=" * 84)
    print("BCI Competition IV ds4 — ECoGFlexNet")
    print("metric: mean Pearson r over fingers 1,2,3,5 vs the raw 1000 Hz "
          "test_dg, whole recording")
    print(f"emission {P.FS:.0f} Hz | delay {P.DELAY} "
          f"({1000 * P.DELAY / P.FS:.0f} ms) | seeds {a.seeds}")
    print("=" * 84)

    agg = {r: [] for r in ROWS}
    agg["ensemble"] = []
    ens_rows = []
    per_subject = defaultdict(list)
    int8_per_subject = defaultdict(list)
    int8_agg = []
    val_pairs = []
    val_by_sub = defaultdict(list)
    csv_rows, footprint = [], {}

    for sub in a.subjects:
        if not os.path.exists(P.prep_path(sub, a.data_dir)):
            print(f"\n[{sub}] no prep file — run prep.py. Skipping.")
            continue
        d = P.load_prep(sub, a.data_dir)
        x = P.normalised_features(d)
        _, _, test = P.split_bounds(d)
        dg_mean = d["dg_mean"].numpy().ravel()
        dg_std = d["dg_std"].numpy().ravel()

        print(f"\n{'=' * 84}\n{sub} | {d['n_electrodes']} electrodes x "
              f"{d['n_bands']} bands | test region processed [{test[0]}, {test[1]}) "
              f"= {d['n_test_raw']} raw samples\n{'=' * 84}")

        preds = []
        for seed in a.seeds:
            ck_path = P.ckpt_path(script_dir, sub, seed, a.tag)
            if not os.path.exists(ck_path):
                print(f"  seed {seed}: missing {os.path.basename(ck_path)}")
                continue
            model, ck = load_model(ck_path, int(d["n_electrodes"]), int(d["n_bands"]))

            t0 = time.perf_counter()
            pred = P.predict_range(model, x, *test).numpy().astype(np.float64)
            dt = time.perf_counter() - t0
            preds.append(pred)

            rows = score_all(pred, d, test, dg_mean, dg_std)
            vr = ck.get("val_official_r", float("nan"))
            print(f"\n  --- seed {seed} --- val r {vr:.4f} "
                  f"| {len(pred)} predictions | {1e6 * dt / len(pred):.0f} us/sample "
                  f"(unbatched fp32 CPU)")
            print(f"    {'row':<22}{'Thumb':>8}{'Index':>8}{'Middle':>8}"
                  f"{'Ring':>8}{'Pinky':>8} |  official r |   RMSE")
            for r in ROWS:
                P.print_fingers(r, rows[r])
                agg[r].append(rows[r]["official_r"])
                per_subject[(sub, r)].append(rows[r]["official_r"])
                csv_rows.append(dict(subject=sub, seed=seed, row=r,
                                     val_official_r=vr,
                                     official_r=rows[r]["official_r"],
                                     **{f"r_{P.FINGER_NAMES[i]}": float(rows[r]["r"][i])
                                        for i in range(5)}))
            if not np.isnan(vr):
                val_pairs.append((vr, rows["official"]["official_r"]))
                val_by_sub[sub].append((vr, rows["official"]["official_r"]))

            # [FIX-9] INT8 alongside FP32, same table, same run
            if (sub, seed) in int8:
                q = int8[(sub, seed)]
                int8_agg.append(q)
                int8_per_subject[sub].append(q)
                print(f"    {'INT8 (.pte)':<22}{'':>40} | official r {q:>7.4f} "
                      f"| gap {rows['official']['official_r'] - q:+.4f}")

            rep = mac_report(model, P.WINDOW)
            dep = deployment_cost(model, emit=a.emit)
            footprint[sub] = dict(params=rep["params"],
                                  kmac=rep["macs_per_sample"] / 1e3,
                                  mmac_s=dep["mmac_s_cached"],
                                  kmac_sl=dep["macs_per_label_stateless"] / 1e3,
                                  mmac_s_sl=dep["mmac_s_stateless"])
            pte = P.pte_path(script_dir, sub, seed, a.pte_tag or a.tag)
            if os.path.exists(pte):
                footprint[sub]["pte_kb"] = os.path.getsize(pte) / 1024.0

        if len(preds) > 1:
            ens = np.mean(np.stack(preds, 0), axis=0)
            s = P.official_score(ens, d["dg_raw"], test[0], P.DECIM * test[0],
                                 d["dg_raw"].shape[0], mode="zoh",
                                 dg_mean=dg_mean, dg_std=dg_std)
            print()
            P.print_fingers(f"ensemble ({len(preds)} seeds)", s)
            agg["ensemble"].append(s["official_r"])
            per_subject[(sub, "ensemble")] = [s["official_r"]]
            ens_rows.append(dict(subject=sub, n_seeds=len(preds),
                                 official_r=s["official_r"],
                                 **{f"r_{P.FINGER_NAMES[i]}": float(s["r"][i])
                                    for i in range(5)}))

    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 84}\nPER SUBJECT — official r\n{'=' * 84}")
    print(f"{'Subject':<10}{'FP32':>18}{'INT8':>18}{'ensemble':>14}")
    print("-" * 84)
    for sub in a.subjects:
        v = per_subject.get((sub, "official"))
        if not v:
            continue
        q = int8_per_subject.get(sub, [])
        e = per_subject.get((sub, "ensemble"))
        m, s = P.mean_std(v)
        qs = f"{P.mean_std(q)[0]:.4f} ± {P.mean_std(q)[1]:.4f}" if q else "—"
        print(f"{sub:<10}{f'{m:.4f} ± {s:.4f}':>18}{qs:>18}"
              f"{(f'{e[0]:.4f}' if e else '—'):>14}")

    print(f"\n{'=' * 84}\nLADDER — mean ± std over subjects x seeds\n{'=' * 84}")
    blurb = {
        "official": "raw 1 kHz test_dg, whole recording",
        "official-linear": "   + linear interp (reads 10 ms ahead)",
        "emit-20Hz": "   same model, one label per window instead of per sample",
        "smoothed-target": "   internal metric: vs the processed target, 100 Hz",
        "ensemble": "   mean over seeds; N models, N x compute",
    }
    for r in ROWS + ["ensemble"]:
        if not agg[r]:
            continue
        m, s = P.mean_std(agg[r])
        print(f"  {r:<18} {f'{m:.4f} ± {s:.4f}':>18}   {blurb[r]}")
    if int8_agg:
        m, s = P.mean_std(int8_agg)
        gap = P.mean_std(agg["official"])[0] - m
        print(f"  {'INT8 .pte':<18} {f'{m:.4f} ± {s:.4f}':>18}   "
              f"<- PUBLISH THIS if the claim is deployment (gap {gap:+.4f})")

    refund = float("nan")
    if agg["official"] and agg["emit-20Hz"]:
        refund = P.mean_std(agg["official"])[0] - P.mean_std(agg["emit-20Hz"])[0]
        print(f"\n  output-rate refund (100 Hz vs 20 Hz emission): {refund:+.4f} r, "
              f"at 1/5 the sustained compute")

    # [FIX-10] does selection actually work on a recording this short?
    #
    # The POOLED correlation across all runs is misleading and must not be
    # the reported figure: subjects differ enormously in difficulty, so
    # pooling measures "is sub3 easier than sub2" (it is) rather than "does
    # validation rank two checkpoints for the SAME subject". On the v3.0 run
    # the pooled value was +0.73 while the within-subject mean was -0.50.
    # Selection is a per-subject decision, so report the per-subject figure.
    rho = float("nan")
    if val_pairs:
        print(f"\n{'=' * 84}\nSELECTION DIAGNOSTIC\n{'=' * 84}")
        rhos = []
        for sub in a.subjects:
            pairs = val_by_sub.get(sub, [])
            if len(pairs) >= 3:
                rs = spearman([p[0] for p in pairs], [p[1] for p in pairs])
                rhos.append(rs)
                print(f"  {sub:<8} Spearman rho(val, test) over "
                      f"{len(pairs)} seeds: {rs:+.3f}")
        if rhos:
            rho = float(np.mean(rhos))
            print(f"  {'mean':<8} {rho:+.3f}   <- REPORT THIS")
        va = np.array([p[0] for p in val_pairs])
        te = np.array([p[1] for p in val_pairs])
        print(f"  (pooled across subjects: {spearman(va, te):+.3f} — inflated "
              f"by between-subject\n   difficulty, not a statement about "
              f"checkpoint selection. Do not quote it.)")
        print("  Near zero or negative means the held-out blocks do not rank "
              "checkpoints and\n  epoch/SWA selection is close to a coin flip. "
              "Report it either way.")

    if footprint:
        print(f"\n{'=' * 84}\nEDGE FOOTPRINT\n{'=' * 84}")
        print(f"{'Subject':<9}{'params':>9}{'kMAC/lbl':>10}{'MMAC/s':>9}"
              f"{'kMAC/lbl':>10}{'MMAC/s':>9}{'.pte KB':>9}")
        print(f"{'':<9}{'':>9}{'(cached)':>10}{'(cached)':>9}"
              f"{'(.pte)':>10}{'(.pte)':>9}")
        for sub, f in footprint.items():
            print(f"{sub:<9}{f['params']:>9,}{f['kmac']:>10.1f}{f['mmac_s']:>9.2f}"
                  f"{f['kmac_sl']:>10.1f}{f['mmac_s_sl']:>9.2f}"
                  f"{f.get('pte_kb', float('nan')):>9.2f}")
        print("\n(cached) = ring-buffer streaming, the firmware limit.")
        print(f"(.pte)   = the stateless artefact export_int8.py writes, emit={a.emit}.")
        print("Quote the second one unless you have written the cached kernel,")
        print("and quote its batching latency with it — see deploy_sweep.py.")

    print(f"\n{'=' * 84}")
    print("published ds4 reference points (all NON-causal, all offline):")
    for name, vals, _ in REFERENCES:
        clean = name.replace("\\&", "&")
        print(f"  {clean:<38} {vals[0]:.2f} / {vals[1]:.2f} / {vals[2]:.2f} "
              f" avg {np.mean(vals):.2f}")
    print("A causal, streaming, INT8 decoder is not competing with those on")
    print("accuracy alone. It competes on accuracy per sustained MAC and on")
    print("being deployable. Say so, and show both axes.")
    print("=" * 84)

    if a.csv and csv_rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nwrote {len(csv_rows)} rows to {a.csv}")

    if a.ensemble_csv and ens_rows:
        with open(a.ensemble_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ens_rows[0].keys()))
            w.writeheader()
            w.writerows(ens_rows)
        print(f"wrote {len(ens_rows)} ensemble rows to {a.ensemble_csv}")

    if a.latex:
        write_latex(a.latex, per_subject, agg, int8_per_subject, int8_agg,
                    footprint, refund, rho, len(a.seeds))


if __name__ == "__main__":
    main()
