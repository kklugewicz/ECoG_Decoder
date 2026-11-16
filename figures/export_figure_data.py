#!/usr/bin/env python3
"""export_figure_data.py - every number behind the paper's figures, from raw results.

    python figures/export_figure_data.py

Needs numpy and scipy only (no torch). Reads the per-seed CSVs and board
results in results/ and the competition test labels in data/raw/, and writes

    figures/data/*.csv              one CSV per figure, read by matlab/*.m
    figures/tex/numbers_v4.tex      macros used by the journal-length paper
    figures/tex/tab_compare_v4.tex  its comparison table (Table 1)
    figures/tex/tab_compare_short.tex  Table I of the 4-page version

Like numbers.tex and numbers_extra.tex: regenerate, never hand-edit.

Prior-work numbers (Lin et al. 2025, FingerFlex, DTCNet, ...) are literature
values typed in once below, with the table or page they come from.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import io as sio
from scipy import stats

SUBJECTS = ["sub1", "sub2", "sub3"]
SEEDS = range(5)
SCORED = [0, 1, 2, 4]                  # ring finger (index 3) is not scored
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
DECIM = 10                             # 100 Hz decoder output, 1000 Hz labels

# Leave-one-out ablation rows, in the order the paper discusses them.
ABLATION_ROWS = [
    ("env-5hz", "5 Hz envelope low-pass (83 ms delay)"),
    ("bands-4", "Four bands instead of seven"),
    ("preset-nano", "Smaller trunk (nano)"),
    ("causal-target", "Causal (not zero-phase) target smoothing"),
    ("no-smooth", "No learned output FIR"),
    ("split-filter", "Train and test filtered separately"),
    ("no-pretrain", "No cross-subject pretraining"),
    ("preset-base", "Larger trunk (base)"),
    ("channels-legacy", "Hand-curated channel list"),
    ("skip-last", "No skip aggregation (last block only)"),
    ("no-gate", "No causal channel gate"),
    ("no-log", "No log compression"),
]


# ---------------------------------------------------------------- helpers
def read_runs(path):
    """{(subject, seed): (test_r, val_r)} for the 'official' scoring row."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["row"] == "official":
                out[(r["subject"], int(r["seed"]))] = (float(r["official_r"]),
                                                        float(r["val_official_r"]))
    return out


def cells():
    return [(s, k) for s in SUBJECTS for k in SEEDS]


def signflip_p(d):
    """Exact two-sided sign-flip permutation p for the mean of paired diffs."""
    d = np.asarray(d, dtype=float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    null = np.abs(signs @ d)
    return float(np.mean(null >= abs(d.sum()) - 1e-12))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def t_ci(x, level=0.95):
    x = np.asarray(x, dtype=float)
    half = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - half, x.mean() + half


def official_r(pred100, dg1000):
    """Competition criterion: zero-order hold to 1000 Hz, Pearson r per finger,
    mean over the four scored fingers."""
    trace = np.repeat(np.asarray(pred100, dtype=float), DECIM, axis=0)[:dg1000.shape[0]]
    r = np.array([np.corrcoef(trace[:, f], dg1000[:, f])[0, 1] for f in range(5)])
    return float(r[SCORED].mean()), r


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            # no commas inside text fields, so the MATLAB reader can split on ','
            w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating))
                        else (v.replace(",", ";") if isinstance(v, str) else v)
                        for v in row])


def fmt_signed(x, nd=3):
    return f"\\ensuremath{{{x:+.{nd}f}}}"


def write_compare_table(path, rows):
    """Table 1 of main_v4.tex: this work against published ds4 decoders."""
    by = {r[0]: r for r in rows}

    def r_cell(v, nd):
        return "--" if v == "" else f"{float(v):.{nd}f}"

    def mem_cell(kb):
        if kb == "":
            return "n.s."
        kb = float(kb)
        if kb < 100:
            return f"{kb:.1f}\\,kB"
        if kb < 1000:
            return f"{kb:.0f}\\,kB"
        return f"{kb / 1000:.2f}\\,MB"

    # csv label, table label, causal, scored at, fingers, on device, decimals, memory note
    spec = [
        ("This work (official criterion)",),
        ("This work, INT8 on board", "INT8 PTQ, run on the board", "yes", "1\\,kHz", "4", "yes", 3, ""),
        ("This work, FP32", "FP32, host", "yes", "1\\,kHz", "4", "--", 3, "$^{d}$"),
        ("This work, 5-seed INT8 ensemble", "Five INT8 seeds averaged", "yes", "1\\,kHz", "4",
         "yes$^{a}$", 3, ""),
        ("Embedded decoder and its offline benchmark",),
        ("Lin et al. RF, deployed (STM32H7)", "Lin et al.\\ \\cite{lin2025}, random forest",
         "win.$^{e}$", "10\\,Hz", "5", "yes$^{b}$", 3, ""),
        ("Lin et al. CNN+LSTM", "Lin et al.\\ \\cite{lin2025}, CNN+LSTM", "win.$^{e}$", "10\\,Hz",
         "5", "--", 3, ""),
        ("Other published decoders, offline",),
        ("Liang & Bougrain (winner)", "Liang and Bougrain \\cite{liang2012} (winner)", "n.s.",
         "1\\,kHz", "4", "--", 2, ""),
        ("Flamary & Rakotomamonjy", "Flamary and Rakotomamonjy \\cite{flamary2012}", "n.s.",
         "1\\,kHz", "4", "--", 2, ""),
        ("Xie et al. (CNN-LSTM)", "Xie et al.\\ \\cite{xie2018}", "n.s.", "n.s.", "n.s.", "--", 2, ""),
        ("Petrosyan et al. (compact CNN)", "Petrosyan et al.\\ \\cite{petrosyan2021}$^{c}$", "yes",
         "n.s.", "n.s.", "--", 2, ""),
        ("Yao et al. (LightGBM)", "Yao et al.\\ \\cite{yao2022}", "yes", "n.s.", "n.s.", "--", 2, ""),
        ("FingerFlex (U-Net)", "FingerFlex \\cite{lomtev2022fingerflex}$^{f}$", "no", "100\\,Hz", "5",
         "--", 2, "$^{d}$"),
        ("DTCNet", "DTCNet \\cite{wang2025dtcnet}", "no", "n.s.", "5", "--", 2, "$^{d}$"),
    ]
    three_dp_mean = ("Yao et al. (LightGBM)", "Lin et al. RF, deployed (STM32H7)", "Lin et al. CNN+LSTM")
    out = ["% AUTO-GENERATED by figures/export_figure_data.py (write_compare_table). Do not edit.",
           "\\begin{tabular}{@{}lccccrcccc@{}}",
           "\\toprule",
           "Decoder & Causal & Scored at & Fingers & On device & Memory & S1 & S2 & S3 & Mean \\\\",
           "\\midrule"]
    first = True
    for sp in spec:
        if len(sp) == 1:
            if not first:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{sp[0]}}}}} \\\\")
            first = False
            continue
        key, label, causal, rate, fingers, dev, nd, mnote = sp
        r = by[key]
        s1, s2, s3, mean, mem = r[2], r[3], r[4], r[5], r[8]
        mean_nd = 3 if key.startswith("This work") or key in three_dp_mean else 2
        cells = [label, causal, rate, fingers, dev, mem_cell(mem) + mnote,
                 r_cell(s1, nd), r_cell(s2, nd), r_cell(s3, nd), r_cell(mean, mean_nd)]
        if key == "This work, INT8 on board":
            cells = [f"\\textbf{{{c}}}" if i in (0, 9) else c for i, c in enumerate(cells)]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(out))
    print(f"wrote {path}")


def write_compare_table_short(path, rows):
    """Column-width comparison for the 4-page IEEE paper (main_ieee_short.tex).
    Fewer rows and columns than tab_compare_v4.tex; protocol notes go in the
    caption. Plain multicolumn{8}{l}{...} so tex2docx_ieee_short.py reads it."""
    by = {r[0]: r for r in rows}

    def r_cell(v, nd):
        return "--" if v == "" else f"{float(v):.{nd}f}"

    def mem_cell(kb):
        if kb == "":
            return "n.s."
        kb = float(kb)
        if kb < 100:
            return f"{kb:.1f}\\,kB"
        if kb < 1000:
            return f"{kb:.0f}\\,kB"
        return f"{kb / 1000:.2f}\\,MB"

    # csv label, table label, causal, device, decimals (per subject), decimals (mean)
    spec = [
        ("This work (official criterion)",),
        ("This work, INT8 on board", "INT8 on board", "yes", "board", 3, 3),
        ("This work, FP32", "FP32 on host", "yes", "--", 3, 3),
        ("This work, 5-seed INT8 ensemble", "5-seed ensemble", "yes", "board$^{a}$", 3, 3),
        ("Published (as reported)",),
        ("Lin et al. RF, deployed (STM32H7)", "Lin et al.\\ \\cite{lin2025}, forest", "win.$^{b}$",
         "MCU$^{c}$", 3, 3),
        ("Lin et al. CNN+LSTM", "Lin et al.\\ \\cite{lin2025}, CNN+LSTM", "win.$^{b}$", "--", 3, 3),
        ("Yao et al. (LightGBM)", "Yao et al.\\ \\cite{yao2022}", "yes", "--", 2, 3),
        ("Liang & Bougrain (winner)", "Liang \\& Bougrain \\cite{liang2012}", "n.s.", "--", 2, 2),
        ("FingerFlex (U-Net)", "FingerFlex \\cite{lomtev2022fingerflex}", "no", "--", 2, 2),
        ("DTCNet", "DTCNet \\cite{wang2025dtcnet}", "no", "--", 2, 2),
    ]
    out = ["% AUTO-GENERATED by figures/export_figure_data.py (write_compare_table_short). Do not edit.",
           "\\begin{tabular}{@{}lcccrrrr@{}}",
           "\\toprule",
           "Decoder & Causal & Device & Memory & S1 & S2 & S3 & Mean \\\\",
           "\\midrule"]
    first = True
    for sp in spec:
        if len(sp) == 1:
            if not first:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{sp[0]}}}}} \\\\")
            first = False
            continue
        key, label, causal, dev, nd, mean_nd = sp
        r = by[key]
        cells = [label, causal, dev, mem_cell(r[8]),
                 r_cell(r[2], nd), r_cell(r[3], nd), r_cell(r[4], nd), r_cell(r[5], mean_nd)]
        if key == "This work, INT8 on board":
            cells = [f"\\textbf{{{c}}}" if i in (0, 7) else c for i, c in enumerate(cells)]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"wrote {path}")


# ------------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    repo = here.parent
    ap.add_argument("--results", type=Path, default=repo / "results",
                    help="the results/ folder of this repository")
    ap.add_argument("--raw", type=Path, default=repo / "data" / "raw",
                    help="folder with sub*_testlabels.mat (BCI Competition IV ds4)")
    ap.add_argument("--out", type=Path, default=here / "data")
    ap.add_argument("--paper", type=Path, default=here / "tex",
                    help="where numbers_v4.tex and the comparison tables go")
    a = ap.parse_args()
    res, out, paper = a.results, a.out, a.paper
    out.mkdir(parents=True, exist_ok=True)
    macros = {}

    # ---------------------------------------------------------- test labels
    dg = {s: sio.loadmat(a.raw / f"{s}_testlabels.mat")["test_dg"]
          for s in SUBJECTS}

    # -------------------------------------- board predictions (INT8, cooled)
    board = res / "board"
    preds = {(s, k): np.load(board / f"preds_{s}_s{k}.npy").astype(float)
             for s, k in cells()}
    board_r = {c: official_r(preds[c], dg[c[0]])[0] for c in cells()}
    macros["VFBoardMean"] = f"{np.mean(list(board_r.values())):.3f}"

    # 5-seed ensemble of the board's own INT8 predictions (equal weights)
    ens = {}
    for s in SUBJECTS:
        p = np.mean([preds[(s, k)] for k in SEEDS], axis=0)
        ens[s] = official_r(p, dg[s])[0]
    for s, name in zip(SUBJECTS, ["One", "Two", "Three"]):
        macros[f"VFBoardEns{name}"] = f"{ens[s]:.3f}"
    macros["VFBoardEnsMean"] = f"{np.mean(list(ens.values())):.3f}"
    macros["VFBoardEnsGain"] = fmt_signed(np.mean(list(ens.values()))
                                          - np.mean(list(board_r.values())))

    # ------------------------------------------- Fig 3: decoded traces
    # Median subject (by mean r) and that subject's median seed.
    subj_mean = {s: np.mean([board_r[(s, k)] for k in SEEDS]) for s in SUBJECTS}
    t_sub = sorted(SUBJECTS, key=lambda s: subj_mean[s])[1]
    seeds_sorted = sorted(SEEDS, key=lambda k: board_r[(t_sub, k)])
    t_seed = seeds_sorted[2]
    g100 = dg[t_sub][::DECIM]                       # glove at 100 Hz
    p100 = preds[(t_sub, t_seed)]
    W = 3000                                        # 30 s
    # Rule fixed in advance: the first 30-s window (5-s steps) in which every
    # scored finger spans at least 75% of its whole-recording range.
    start = 0
    for st in range(0, g100.shape[0] - W + 1, 500):
        seg = g100[st:st + W]
        frac = [np.ptp(seg[:, f]) / np.ptp(g100[:, f]) for f in SCORED]
        if min(frac) >= 0.75:
            start = st
            break
    z = lambda x: (x - x.mean(0)) / x.std(0)      # display only; r is affine-invariant
    gz, pz = z(g100), z(p100)
    rows = []
    for i in range(start, start + W):
        rows.append([i / 100.0] + [gz[i, f] for f in SCORED] + [pz[i, f] for f in SCORED])
    write_csv(out / "fig03_traces.csv",
              ["time_s"] + [f"glove_{FINGERS[f].lower()}" for f in SCORED]
              + [f"decoded_{FINGERS[f].lower()}" for f in SCORED], rows)
    r_all = official_r(p100, dg[t_sub])[1]
    seg_raw = dg[t_sub][start * DECIM:(start + W) * DECIM]
    seg_pred = p100[start:start + W]
    r_win = official_r(seg_pred, seg_raw)[1]
    write_csv(out / "fig03_finger_r.csv", ["finger", "r_whole_recording", "r_window"],
              [[FINGERS[f], r_all[f], r_win[f]] for f in SCORED])
    macros["VFTraceSubject"] = t_sub[-1]
    macros["VFTraceSeed"] = str(t_seed)
    macros["VFTraceStart"] = f"{start / 100:.0f}"
    macros["VFTraceEnd"] = f"{(start + W) / 100:.0f}"
    macros["VFTraceRwhole"] = f"{r_all[SCORED].mean():.3f}"
    macros["VFTraceRwin"] = f"{r_win[SCORED].mean():.3f}"

    # ---------------------------------------- Fig 5: leave-one-out ablation
    abl = {40: res / "ablation_e40", 80: res / "ablation_e80"}
    ref = {ep: read_runs(d / "reference.csv") for ep, d in abl.items()}
    cell_rows, summ_rows, p40 = [], [], []
    for j, (key, label) in enumerate(ABLATION_ROWS, start=1):
        rec = {"row_id": j, "key": key, "label": label}
        for ep, d in abl.items():
            runs = read_runs(d / f"{key}.csv")
            dt = np.array([runs[c][0] - ref[ep][c][0] for c in cells()])
            dv = np.array([runs[c][1] - ref[ep][c][1] for c in cells()])
            for (s, k), x, v in zip(cells(), dt, dv):
                cell_rows.append([j, ep, int(s[-1]), k, x, v])
            lo, hi = t_ci(dt)
            rec.update({f"mean{ep}": dt.mean(), f"ci{ep}_lo": lo, f"ci{ep}_hi": hi,
                        f"nneg{ep}": int((dt < 0).sum()), f"val{ep}": dv.mean(),
                        f"p{ep}": signflip_p(dt)})
            for s in SUBJECTS:
                rec[f"{s}_mean{ep}"] = float(np.mean([x for (ss, _), x in zip(cells(), dt)
                                                      if ss == s]))
        p40.append(rec["p40"])
        summ_rows.append(rec)
    for rec, ph in zip(summ_rows, holm(p40)):
        rec["pholm40"] = ph
    write_csv(out / "fig05_ablation_cells.csv",
              ["row_id", "epochs", "subject", "seed", "delta_test_r", "delta_val_r"],
              cell_rows)
    cols = ["row_id", "key", "label", "mean40", "ci40_lo", "ci40_hi", "p40", "pholm40",
            "nneg40", "val40", "mean80", "ci80_lo", "ci80_hi", "nneg80", "val80",
            "sub1_mean40", "sub2_mean40", "sub3_mean40"]
    write_csv(out / "fig05_ablation_summary.csv", cols,
              [[rec[c] for c in cols] for rec in summ_rows])
    macros["VFAblRefForty"] = f"{np.mean([v[0] for v in ref[40].values()]):.3f}"
    by_key = {r["key"]: r for r in summ_rows}
    for key, name in [("no-gate", "NoGate"), ("no-log", "NoLog"), ("skip-last", "SkipLast")]:
        macros[f"VFVal{name}"] = fmt_signed(by_key[key]["val40"])
        macros[f"VFVal{name}Eighty"] = fmt_signed(by_key[key]["val80"])

    # -------------------------------------------- Fig 6: look-ahead leak
    gate_rows = []
    nog = read_runs(abl[40] / "no-gate.csv")
    with open(res / "main" / "eca_gate_leak.csv", newline="") as fh:
        eca = {(r["subject"], int(r["seed"])): r for r in csv.DictReader(fh)}
    for c in cells():
        e = eca[c]
        gate_rows.append([int(c[0][-1]), c[1], ref[40][c][0], nog[c][0],
                          float(e["dense-chunk"]), float(e["train-proto"]),
                          float(e["exported"])])
    write_csv(out / "fig06_gate_leak.csv",
              ["subject", "seed", "causal_gate", "no_gate", "eca_dense_offline",
               "eca_training_window", "eca_exported_causal"], gate_rows)
    G = np.array([r[2:] for r in gate_rows])
    infl = G[:, 2] - G[:, 4]
    macros["VFLeakAllPositive"] = str(int((infl > 0).sum()))
    macros["VFLeakMin"] = f"{infl.min():+.3f}"
    macros["VFLeakMax"] = f"{infl.max():+.3f}"
    # unsigned versions, for sentences that already say "lower" or "below"
    macros["VFLeakMeanAbs"] = f"{abs(infl.mean()):.3f}"
    macros["VFGateExpBelowCausal"] = f"{abs((G[:, 4] - G[:, 0]).mean()):.3f}"
    macros["VFGateExpBelowNone"] = f"{abs((G[:, 4] - G[:, 1]).mean()):.3f}"

    # ---------------------------- Fig 7: quantisation and training budget
    def read_int8(name):
        with open(res / "main" / name, newline="") as fh:
            return {(r["subject"], int(r["seed"])): r for r in csv.DictReader(fh)}
    ptq, qat = read_int8("int8_ptq_results.csv"), read_int8("int8_qat_results.csv")
    write_csv(out / "fig07a_quantisation.csv",
              ["subject", "seed", "fp32", "int8_ptq", "int8_qat", "int8_ptq_board"],
              [[int(s[-1]), k, float(ptq[(s, k)]["fp32_official_r"]),
                float(ptq[(s, k)]["official_r"]), float(qat[(s, k)]["official_r"]),
                board_r[(s, k)]] for s, k in cells()])
    e120 = read_runs(res / "reference_e120" / "reference.csv")
    write_csv(out / "fig07b_training_budget.csv",
              ["subject", "seed", "epochs_40", "epochs_80", "epochs_120"],
              [[int(s[-1]), k, ref[40][(s, k)][0], ref[80][(s, k)][0], e120[(s, k)][0]]
               for s, k in cells()])

    # ------------------------------------ Fig 8: real-time replay on board
    final = res / "board"
    age_rows, ages = [], {}
    for m, mode in enumerate(["sleep", "spin"], start=1):
        allv = []
        for s in SUBJECTS:
            v = np.load(final / f"paced_{mode}_{s}_age_ms.npy").astype(float)
            allv.append(v)
            age_rows += [[m, int(s[-1]), i, x] for i, x in enumerate(v)]
        ages[mode] = np.concatenate(allv)
    write_csv(out / "fig08a_label_age_raw.csv", ["mode_1sleep_2spin", "subject", "batch",
                                                 "label_age_ms"], age_rows)
    grid = np.round(np.arange(94.0, 150.01, 0.25), 2)
    exc = []
    for x in grid:
        row = [x]
        for mode in ["sleep", "spin"]:
            pe = float(np.mean(ages[mode] > x))
            row.append(pe if pe > 0 else "")
        exc.append(row)
    write_csv(out / "fig08a_label_age_exceedance.csv",
              ["label_age_ms", "p_exceed_sleep", "p_exceed_spin"], exc)
    macros["VFAgeN"] = f"{ages['sleep'].size:,}"
    macros["VFAgeMissSleep"] = str(int((ages["sleep"] > 100).sum()))
    macros["VFAgeMissSpin"] = str(int((ages["spin"] > 100).sum()))

    soak = json.load(open(board / "soak.json"))
    write_csv(out / "fig08b_soak_thermal.csv", ["time_s", "soc_temp_c", "cpu_mhz"],
              [[t, c, mhz] for t, c, mhz in soak["thermal"]["trace"]])

    # ------------------------------ Fig 9: emission batch E on the board
    sweep = json.load(open(board / "sweep.json"))
    sw_rows = []
    for key, v in sorted(sweep.items()):
        s, e = key.split("_e")
        sw_rows.append([int(e), int(s[-1]), v["latency"]["median"], v["latency"]["p95"],
                        v["latency"]["p99"]])
    sw_rows.sort()
    write_csv(out / "fig09a_call_time_vs_E.csv",
              ["emit_E", "subject", "median_ms", "p95_ms", "p99_ms"], sw_rows)
    with open(board / "deploy_board.csv", newline="") as fh:
        dep = list(csv.DictReader(fh))
    write_csv(out / "fig09b_pareto.csv",
              ["emit_E", "window", "kmac_per_label", "mmac_per_s", "max_latency_ms",
               "in_100ms_budget"],
              [[int(r["emit"]), int(r["window"]), float(r["kmac_per_label"]),
                float(r["mmac_s"]), float(r["lat_max_ms"]), int(r["in_budget"] == "True")]
               for r in dep])

    # --------------------------------- Fig 10: does validation pick models?
    sel = []
    # the null control ("channels-all") reproduces the reference bit for bit, so it is
    # left out: including it would count the reference models twice
    keys = ["reference"] + [k for k, _ in ABLATION_ROWS]
    for j, key in enumerate(keys):
        runs = read_runs(abl[40] / f"{key}.csv")
        sel += [[int(s[-1]), j, k, runs[(s, k)][1], runs[(s, k)][0]] for s, k in cells()]
    write_csv(out / "fig10_val_vs_test.csv",
              ["subject", "config_id", "seed", "val_r", "test_r"], sel)
    S = np.array(sel, dtype=float)
    rho = [stats.spearmanr(S[S[:, 0] == i, 3], S[S[:, 0] == i, 4])[0] for i in (1, 2, 3)]
    macros["VFSelRhoSubOne"], macros["VFSelRhoSubTwo"], macros["VFSelRhoSubThree"] = \
        [f"{x:+.2f}" for x in rho]

    # ------------------------- Fig 4 and Table 1: this work vs prior work
    # memory = deployed artefact or model size in kB (1 kB = 1000 bytes)
    pte_bytes = 66836                     # checkpoints/paper_ptq_sub1_s0_int8.pte
    params = 25369
    prior = [
        # label, year, s1, s2, s3, mean, causal, on_device, memory_kB, fingers, scoring, source
        ("Flamary & Rakotomamonjy", 2012, 0.48, 0.24, 0.56, 0.43, "check", 0, "", "4", "official",
         "competition entry"),
        ("Liang & Bougrain (winner)", 2012, 0.45, 0.39, 0.59, 0.48, "check", 0, "", "4", "official",
         "competition entry"),
        ("Xie et al. (CNN-LSTM)", 2018, 0.56, 0.41, 0.58, 0.52, "check", 0, "", "", "",
         "J. Neural Eng. 15:036009"),
        ("Petrosyan et al. (compact CNN)", 2021, 0.45, 0.34, 0.56, 0.45, "yes", 0, "", "", "",
         "as tabulated by DTCNet 2025"),
        ("Yao et al. (LightGBM)", 2022, 0.52, 0.47, 0.61, 0.537, "past only", 0, "", "", "",
         "J. Neural Eng. 19:016037"),
        ("Lin et al. RF, deployed (STM32H7)", 2025, "", "", "", 0.535, "yes", 1, 903.0, "5",
         "10 Hz", "J. Neural Eng. 22:046012, Tables 2-4"),
        ("Lin et al. CNN+LSTM", 2025, 0.567, 0.465, 0.669, 0.567, "window", 0, 3267.9, "5",
         "10 Hz", "J. Neural Eng. 22:046012, Tables 3, 5"),
        ("Lin et al. LSTM", 2025, 0.540, 0.351, 0.615, 0.502, "window", 0, 5752.5, "5", "10 Hz",
         "J. Neural Eng. 22:046012, Tables 3, 5"),
        ("Lin et al. CNN", 2025, 0.516, 0.443, -0.022, 0.312, "window", 0, 2561.2, "5", "10 Hz",
         "J. Neural Eng. 22:046012, Tables 3, 5"),
        ("Lin et al. ANN", 2025, 0.521, 0.328, -0.013, 0.279, "window", 0, 17002.4, "5", "10 Hz",
         "J. Neural Eng. 22:046012, Tables 3, 5"),
        ("Lin et al. SVR", 2025, 0.496, 0.277, 0.572, 0.448, "window", 0, 290431.2, "5", "10 Hz",
         "J. Neural Eng. 22:046012, Tables 3, 5"),
        ("Lin et al. PLS", 2025, 0.496, 0.341, -0.008, 0.276, "window", 0, 109.7, "5", "10 Hz",
         "J. Neural Eng. 22:046012, Tables 3, 5"),
        ("FingerFlex (U-Net)", 2022, 0.66, 0.62, 0.74, 0.67, "no", 0, 2400.0, "5", "100 Hz",
         "arXiv 2211.01960 (~600k parameters; own 6.5/3.5 min split)"),
        ("DTCNet", 2025, 0.71, 0.59, 0.77, 0.69, "no", 0, 2680.0, "5", "",
         "Front. Comput. Neurosci. 19 (550k-790k parameters)"),
    ]
    ptq_mean = {s: np.mean([board_r[(s, k)] for k in SEEDS]) for s in SUBJECTS}
    fp_mean = {s: np.mean([float(ptq[(s, k)]["fp32_official_r"]) for k in SEEDS])
               for s in SUBJECTS}
    ours = [
        ("This work, INT8 on board", 2026, *[ptq_mean[s] for s in SUBJECTS],
         np.mean(list(ptq_mean.values())), "yes", 1, pte_bytes / 1000, "4",
         "official (1000 Hz)", "board_int8_scores.csv"),
        ("This work, FP32", 2026, *[fp_mean[s] for s in SUBJECTS],
         np.mean(list(fp_mean.values())), "yes", 0, params * 4 / 1000, "4",
         "official (1000 Hz)", "int8_ptq_results.csv"),
        ("This work, 5-seed INT8 ensemble", 2026, *[ens[s] for s in SUBJECTS],
         np.mean(list(ens.values())), "yes", 1, 5 * pte_bytes / 1000, "4",
         "official (1000 Hz)", "board predictions averaged; ensemble timing not measured"),
    ]
    write_csv(out / "fig04_accuracy_vs_memory.csv",
              ["method", "year", "r_sub1", "r_sub2", "r_sub3", "r_mean", "causal",
               "measured_on_device", "memory_kB", "fingers_averaged", "scoring", "source"],
              ours + prior)
    paper.mkdir(parents=True, exist_ok=True)
    write_compare_table(paper / "tab_compare_v4.tex", ours + prior)
    write_compare_table_short(paper / "tab_compare_short.tex", ours + prior)
    macros["VFPteKB"] = f"{pte_bytes / 1000:.1f}"
    macros["VFLinFlashKB"] = "903"
    macros["VFFlashRatio"] = f"{903000 / pte_bytes:.1f}"
    macros["VFParamKB"] = f"{params * 4 / 1000:.0f}"

    # --------------------------------------------------- numbers_v4.tex
    paper.mkdir(parents=True, exist_ok=True)
    with open(paper / "numbers_v4.tex", "w") as fh:
        fh.write("% AUTO-GENERATED by figures/export_figure_data.py. Do not edit.\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(list(out.glob('*.csv')))} CSV files to {out}")
    print(f"wrote {len(macros)} macros to {paper / 'numbers_v4.tex'}")
    for k, v in macros.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
