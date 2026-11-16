"""
paper_tables.py — generates every table and number the paper quotes that is
not already emitted by decoder/score.py, decoder/deploy_sweep.py,
board/write_board_numbers.py or figures/export_figure_data.py.

Reads raw results only (per-seed CSVs in results/, board JSON/NPY), so the
prose cannot drift from the runs. Writes into figures/tex/ (or --out):

    tab_ablation.tex   leave-one-out grid, 40-epoch and 80-epoch replication
    tab_gate.tex       time-pooling gate scored three ways vs causal / none
    tab_int8.tex       FP32 vs PTQ vs QAT per subject
    tab_board.tex      on-board measurements
    numbers_extra.tex  every other number the prose interpolates

The input-clipping numbers (ClipTrainMax, ClipTestMax, ClipFracPct, ClipGain)
re-score the sub3 models, so they need the prepared data (decoder/prep.py)
and the shipped checkpoints. --skip-clipping leaves them out.

Usage:  python figures/paper_tables.py
"""
import csv
import glob
import itertools
import json
import os

import argparse

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results")          # set from --results in main()
OUT = os.path.join(HERE, "tex")              # set from --out in main()
DATA_DIR = os.path.join(REPO, "data", "prep")
CKPT_DIR = os.path.join(REPO, "checkpoints")
SUBS = ["sub1", "sub2", "sub3"]
ROWS = [  # (row, macro stem, table label)
    ("env-5hz", "EnvFive", "5\\,Hz envelope low-pass (83\\,ms delay)"),
    ("bands-4", "BandsFour", "Four bands instead of seven"),
    ("preset-nano", "PresetNano", "Smaller trunk (\\texttt{nano})"),
    ("causal-target", "CausalTarget", "Causal (not zero-phase) target smoothing"),
    ("no-smooth", "NoSmooth", "No learned output FIR"),
    ("split-filter", "SplitFilter", "Train/test filtered independently"),
    ("no-pretrain", "NoPretrain", "No cross-subject trunk pretraining"),
    ("preset-base", "PresetBase", "Larger trunk (\\texttt{base})"),
    ("channels-legacy", "ChannelsLegacy", "Hand-curated channel list"),
    ("skip-last", "SkipLast", "No skip aggregation (last block only)"),
    ("no-gate", "NoGate", "No causal channel gate"),
    ("no-log", "NoLog", "No log compression"),
]
NULL_ROW = "channels-all"
macros = {}


def m(name, val):
    macros[name] = val


# ----------------------------------------------------------------------
def load_cells(path, kind="official", col="official_r"):
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, newline="")):
        if r.get("row", kind) == kind:
            out[(r["subject"], int(r["seed"]))] = float(r[col])
    return out


def signflip_p(d):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 1.0
    s = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-15).mean())


def holm(ps):
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def paired(arm, ref):
    k = sorted(set(arm) & set(ref))
    d = np.array([arm[x] - ref[x] for x in k])
    sub = [np.mean([arm[x] - ref[x] for x in k if x[0] == s]) for s in SUBS]
    return dict(d=d, mean=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                p=signflip_p(d), sub=sub, n=len(d))


def f3(x):
    return f"{x:.3f}"


def sgn(x, nd=3):
    if abs(x) < 0.5 * 10 ** -nd:          # no "-0.000"
        x = 0.0
    # \ensuremath: a real minus sign whether the macro lands in text or math
    return f"\\ensuremath{{{x:+.{nd}f}}}"


def ptex(p):
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


# ======================================================================
# 1. Ablation grid
# ======================================================================
def ablation():
    grids = {"e40": os.path.join(RES, "ablation_e40"), "e80": os.path.join(RES, "ablation_e80")}
    res = {}
    for g, d in grids.items():
        ref = load_cells(os.path.join(d, "reference.csv"))
        rows = {r: paired(load_cells(os.path.join(d, f"{r}.csv")), ref) for r, _, _ in ROWS}
        adj = holm(np.array([rows[r]["p"] for r, _, _ in ROWS]))
        for (r, _, _), a in zip(ROWS, adj):
            rows[r]["p_holm"] = a
        null = paired(load_cells(os.path.join(d, f"{NULL_ROW}.csv")), ref)
        res[g] = dict(ref=ref, rows=rows, null=null)
    ref40 = res["e40"]["ref"]
    m("AblRefMean", f3(np.mean(list(ref40.values()))))
    m("AblRefMeanEighty", f3(np.mean(list(res["e80"]["ref"].values()))))
    m("AblNullMaxAbs", f"{max(np.abs(res[g]['null']['d']).max() for g in res):.1f}")
    for r, stem, _ in ROWS:
        a, b = res["e40"]["rows"][r], res["e80"]["rows"][r]
        m(f"Abl{stem}", sgn(a["mean"]))
        m(f"Abl{stem}Eighty", sgn(b["mean"]))
        m(f"Abl{stem}P", ptex(a["p"]))
        m(f"Abl{stem}Holm", ptex(a["p_holm"]))
        m(f"Abl{stem}Neg", str(int((a["d"] < 0).sum())))
    agree = sum(np.sign(res["e40"]["rows"][r]["mean"]) == np.sign(res["e80"]["rows"][r]["mean"])
                for r, _, _ in ROWS)
    m("AblSignAgree", str(int(agree)))
    m("AblRows", str(len(ROWS)))

    with open(os.path.join(OUT, "tab_ablation.tex"), "w") as f:
        f.write("% AUTO-GENERATED by figures/paper_tables.py. Do not edit.\n")
        f.write("\\begin{tabular}{@{}lrrrcr@{}}\n\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{40 epochs} & & 80 ep. \\\\\n")
        f.write("\\cmidrule(lr){2-4}\\cmidrule(l){6-6}\n")
        f.write("Reverted component & $\\Delta r$ & $p_\\text{Holm}$ & S1/S2/S3 & & $\\Delta r$ \\\\\n\\midrule\n")
        f.write(f"Full pipeline ($r$) & {np.mean(list(ref40.values())):.3f} & & & & "
                f"{np.mean(list(res['e80']['ref'].values())):.3f} \\\\\n\\midrule\n")
        for r, _, label in ROWS:
            a, b = res["e40"]["rows"][r], res["e80"]["rows"][r]
            signs = "".join("$+$" if x > 0 else "$-$" for x in a["sub"])
            bold = a["p_holm"] < 0.05
            d = f"\\textbf{{{sgn(a['mean'])}}}" if bold else sgn(a["mean"])
            f.write(f"{label} & {d} & {ptex(a['p_holm'])} & {signs} & & {sgn(b['mean'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return res


# ======================================================================
# 2. Training budget, selection diagnostic
# ======================================================================
def budget_and_selection():
    refs = {g: load_cells(os.path.join(RES, d, "reference.csv"))
            for g, d in [("40", "ablation_e40"), ("80", "ablation_e80"),
                         ("120", "reference_e120")]}
    m("RefFortyMean", f3(np.mean(list(refs["40"].values()))))
    m("RefEightyMean", f3(np.mean(list(refs["80"].values()))))
    m("RefOneTwentyMean", f3(np.mean(list(refs["120"].values()))))
    p = paired(refs["120"], refs["40"])
    m("BudgetDelta", sgn(p["mean"]))
    m("BudgetP", ptex(p["p"]))

    # within-subject Spearman over every run in both grids (excluding the null duplicate)
    for g, d in [("Forty", "ablation_e40"), ("Eighty", "ablation_e80")]:
        v, t, s = [], [], []
        for row in ["reference"] + [r for r, _, _ in ROWS]:
            vv = load_cells(os.path.join(RES, d, f"{row}.csv"), col="val_official_r")
            tt = load_cells(os.path.join(RES, d, f"{row}.csv"))
            for k in vv:
                v.append(vv[k]); t.append(tt[k]); s.append(k[0])
        v, t, s = map(np.array, (v, t, s))
        w = [stats.spearmanr(v[s == x], t[s == x]).statistic for x in SUBS]
        m(f"SelRuns{g}", str(len(v)))
        m(f"SelWithin{g}", f"{np.mean(w):+.2f}")
        m(f"SelPooled{g}", f"{stats.spearmanr(v, t).statistic:+.2f}")


# ======================================================================
# 3. Gate leak
# ======================================================================
def gate():
    eca = {(r["subject"], int(r["seed"])): r for r in
           csv.DictReader(open(os.path.join(RES, "main", "eca_gate_leak.csv")))}
    ref = load_cells(os.path.join(RES, "ablation_e40", "reference.csv"))
    nog = load_cells(os.path.join(RES, "ablation_e40", "no-gate.csv"))
    col = {c: {k: float(v[c]) for k, v in eca.items()} for c in ("dense-chunk", "train-proto", "exported")}
    m("GateDense", f3(np.mean(list(col["dense-chunk"].values()))))
    m("GateTrainProto", f3(np.mean(list(col["train-proto"].values()))))
    m("GateExported", f3(np.mean(list(col["exported"].values()))))
    m("GateCausal", f3(np.mean([ref[k] for k in eca])))
    m("GateNone", f3(np.mean([nog[k] for k in eca])))
    infl = paired(col["dense-chunk"], col["exported"])
    m("GateInflation", sgn(infl["mean"]))
    m("GateInflationPos", str(int((infl["d"] > 0).sum())))
    m("GateInflationP", ptex(infl["p"]))
    vc = paired(col["exported"], ref)
    m("GateExpVsCausal", sgn(vc["mean"]))
    m("GateExpVsCausalP", ptex(vc["p"]))
    vn = paired(col["exported"], nog)
    m("GateExpVsNone", sgn(vn["mean"]))
    m("GateExpVsNoneP", ptex(vn["p"]))
    dc = paired(col["dense-chunk"], ref)
    m("GateDenseVsCausal", sgn(dc["mean"]))
    m("GateDenseVsCausalP", ptex(dc["p"]))

    rows = [
        ("Causal pointwise gate (ours)", ref, "exact"),
        ("No gate", nog, "exact"),
        ("Pooling gate, dense offline scoring", col["dense-chunk"], "leaks 200\\,s"),
        ("Pooling gate, training-window scoring", col["train-proto"], "leaks $\\leq$6\\,s"),
        ("Pooling gate, exported (causal)", col["exported"], "none"),
    ]
    with open(os.path.join(OUT, "tab_gate.tex"), "w") as f:
        f.write("% AUTO-GENERATED by figures/paper_tables.py. Do not edit.\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Gate, scoring & Look-ahead & Mean $r$ & $\\Delta$ vs.\\ ours & $p$ \\\\\n\\midrule\n")
        for name, cells, la in rows:
            k = sorted(eca)
            mean = np.mean([cells[x] for x in k])
            if cells is ref:
                f.write(f"{name} & {la} & {mean:.3f} & --- & --- \\\\\n")
            else:
                pp = paired(cells, ref)
                f.write(f"{name} & {la} & {mean:.3f} & {sgn(pp['mean'])} & {ptex(pp['p'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ======================================================================
# 4. Headline, ensemble, INT8
# ======================================================================
def headline_int8():
    fp = load_cells(os.path.join(RES, "main", "main_results.csv"))
    ptq = load_cells(os.path.join(RES, "main", "int8_ptq_results.csv"), kind=None)
    qat = load_cells(os.path.join(RES, "main", "int8_qat_results.csv"), kind=None)
    m("QATMean", f3(np.mean(list(qat.values()))))
    # score.py --ensemble-csv: one row per subject, the 5-seed FP32 ensemble
    e = [float(r["official_r"]) for r in
         csv.DictReader(open(os.path.join(RES, "main", "ensemble_results.csv"), newline=""))]
    m("EnsembleMean", f3(np.mean(e)))
    m("EnsembleGain", sgn(np.mean(e) - np.mean(list(fp.values()))))
    for tag, cells in (("FP", fp), ("PTQ", ptq), ("QAT", qat)):
        for s, nm in zip(SUBS, ("One", "Two", "Three")):
            m(f"{tag}Sub{nm}", f3(np.mean([cells[(s, k)] for k in range(5)])))
    gap_ptq = {s: np.mean([fp[(s, k)] - ptq[(s, k)] for k in range(5)]) for s in SUBS}
    gap_qat = {s: np.mean([fp[(s, k)] - qat[(s, k)] for k in range(5)]) for s in SUBS}
    for s, nm in zip(SUBS, ("One", "Two", "Three")):
        m(f"GapPTQSub{nm}", sgn(gap_ptq[s]))
        m(f"GapQATSub{nm}", sgn(gap_qat[s]))
    m("GapPTQMaxAbsSubOneTwo", f"{max(abs(fp[(s, k)] - ptq[(s, k)]) for s in SUBS[:2] for k in range(5)):.3f}")
    m("GapQATMax", f"{max(fp[k] - qat[k] for k in fp):.3f}")
    val = load_cells(os.path.join(RES, "main", "main_results.csv"), col="val_official_r")
    m("HeadlineValRhoPooled", f"{stats.spearmanr([val[k] for k in fp], [fp[k] for k in fp]).statistic:+.2f}")
    winner = {"sub1": 0.45, "sub2": 0.39, "sub3": 0.59}
    m("PTQSeedsBelowWinnerSubOne", str(sum(ptq[("sub1", k)] < winner["sub1"] for k in range(5))))

    with open(os.path.join(OUT, "tab_int8.tex"), "w") as f:
        f.write("% AUTO-GENERATED by figures/paper_tables.py. Do not edit.\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Model & Sub.\\,1 & Sub.\\,2 & Sub.\\,3 & Mean \\\\\n\\midrule\n")
        for name, cells in (("FP32", fp), ("INT8 PTQ (deployed)", ptq), ("INT8 QAT, 30 epochs", qat)):
            per = [np.mean([cells[(s, k)] for k in range(5)]) for s in SUBS]
            f.write(f"{name} & " + " & ".join(f"{x:.3f}" for x in per) + f" & {np.mean(per):.3f} \\\\\n")
        f.write("\\midrule\n")
        for name, g in (("Gap, PTQ", gap_ptq), ("Gap, QAT", gap_qat)):
            f.write(f"{name} (FP32$-$INT8) & " + " & ".join(sgn(g[s]) for s in SUBS)
                    + f" & {sgn(np.mean(list(g.values())))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ======================================================================
# 5. Input clipping explains sub3's INT8 "gain"
# ======================================================================
def clipping():
    import sys
    import torch
    sys.path.insert(0, os.path.join(REPO, "decoder"))
    import protocol as P
    from score import load_model
    d = P.load_prep("sub3", DATA_DIR)
    x = P.normalised_features(d)
    tr_iv, _, (lo, hi) = P.split_bounds(d)
    tr = torch.cat([x[:, a:b] for a, b in tr_iv], 1)
    tmin, tmax = tr.min().item(), tr.max().item()
    te = x[:, lo:hi]
    xc = x.clamp(tmin, tmax)
    dl = []
    for seed in range(5):
        mdl, _ = load_model(os.path.join(CKPT_DIR, f"paper_sub3_s{seed}.pt"),
                            int(d["n_electrodes"]), int(d["n_bands"]))
        sc = []
        for xx in (x, xc):
            pr = P.predict_range(mdl, xx, lo, hi).numpy().astype(np.float64)
            sc.append(P.official_score(pr, d["dg_raw"], lo, P.DECIM * lo, d["dg_raw"].shape[0])["official_r"])
        dl.append(sc[1] - sc[0])
    m("ClipTrainMax", f"{max(abs(tmin), abs(tmax)):.0f}")
    m("ClipTestMax", f"{te.abs().max().item():.0f}")
    m("ClipFracPct", f"{100 * ((te < tmin) | (te > tmax)).float().mean().item():.3f}")
    m("ClipGain", sgn(np.mean(dl)))


# ======================================================================
# 6. Board
# ======================================================================
def board():
    rb = os.path.join(RES, "board")
    sw = json.load(open(os.path.join(rb, "sweep.json")))
    soak = json.load(open(os.path.join(rb, "soak.json")))
    acc = json.load(open(os.path.join(rb, "accuracy.json")))
    ages = {t: np.concatenate([np.load(p) for p in sorted(glob.glob(
        os.path.join(rb, f"paced_{t}_sub*_age_ms.npy")))]) for t in ("sleep", "spin")}
    e1 = np.median([sw[f"{s}_e1"]["latency"]["median"] for s in SUBS])
    e50 = np.median([sw[f"{s}_e50"]["latency"]["median"] for s in SUBS])
    m("BoardEOneMs", f"{e1:.1f}")
    m("BoardEFiftyMs", f"{e50:.1f}")
    m("BoardSpinMissPct", f"{100 * (ages['spin'] > 100).mean():.2f}")
    m("BoardAccTempMax", f"{max(v['temp_max'] for v in acc.values()):.0f}")

    rows = [("Per-call compute, $E{=}10$ (median / p95)",
             f"{np.median([v['latency']['median'] for v in acc.values()]):.1f} / "
             f"{np.median([v['latency']['p95'] for v in acc.values()]):.1f}\\,ms")]
    for e in (1, 25, 50):
        rows.append((f"Per-call compute, $E{{=}}{e}$ (median)",
                     f"{np.median([sw[f'{s}_e{e}']['latency']['median'] for s in SUBS]):.2f}\\,ms"))
    a = ages["sleep"]
    rows += [
        ("Label age at 100\\,Hz (median / p99 / p99.9)",
         f"{np.percentile(a, 50):.1f} / {np.percentile(a, 99):.1f} / {np.percentile(a, 99.9):.1f}\\,ms"),
        ("Batches over 100\\,ms budget", f"{int((a > 100).sum())} / {len(a):,} ({100 * (a > 100).mean():.2f}\\%)"),
        ("Worst label age", f"{a.max():.0f}\\,ms"),
        ("Batches that overran their period", str(sum(
            json.load(open(p))["overruns"] for p in glob.glob(
                os.path.join(rb, "paced_sleep_sub*.json"))))),
        (f"Soak, {soak['seconds'] // 60}\\,min: peak SoC temp.", f"{soak['thermal']['temp_max']:.0f}\\,$^\\circ$C"),
        ("Soak: CPU clock (min / max)", f"{soak['thermal']['mhz_min']:.0f} / {soak['thermal']['mhz_max']:.0f}\\,MHz"),
    ]
    with open(os.path.join(OUT, "tab_board.tex"), "w") as f:
        f.write("% AUTO-GENERATED by figures/paper_tables.py. Do not edit.\n")
        f.write("\\begin{tabular}{@{}lr@{}}\n\\toprule\nMeasurement & Value \\\\\n\\midrule\n")
        for k, v in rows:
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ======================================================================
def main():
    global RES, OUT, DATA_DIR, CKPT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES, help="the results/ folder")
    ap.add_argument("--out", default=OUT, help="where the .tex files go")
    ap.add_argument("--data-dir", default=DATA_DIR, help="decoder/prep.py output")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--skip-clipping", action="store_true",
                    help="skip the four numbers that need the prepared data")
    a = ap.parse_args()
    RES, OUT, DATA_DIR, CKPT_DIR = a.results, a.out, a.data_dir, a.ckpt_dir
    os.makedirs(OUT, exist_ok=True)
    ablation()
    budget_and_selection()
    gate()
    headline_int8()
    if a.skip_clipping:
        print("  --skip-clipping: ClipTrainMax/ClipTestMax/ClipFracPct/ClipGain not written")
    elif not os.path.exists(os.path.join(DATA_DIR, "sub3_prep.pt")):
        raise SystemExit(f"no prepared data in {DATA_DIR}: run python decoder/prep.py "
                         "first, or pass --skip-clipping")
    else:
        clipping()
    board()
    m("TorchVersion", "2.12.1")
    m("TorchaoVersion", "0.17.0")
    m("ExecutorchVersion", "1.3.1")
    with open(os.path.join(OUT, "numbers_extra.tex"), "w") as f:
        f.write("% AUTO-GENERATED by figures/paper_tables.py. Do not edit.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(macros)} macros + tab_ablation/gate/int8/board to {OUT}")
    for k, v in macros.items():
        print(f"  \\{k} = {v}")


if __name__ == "__main__":
    main()
