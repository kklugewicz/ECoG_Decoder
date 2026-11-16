"""
deploy_sweep.py — [FIX-12] the emission-batch / latency / compute trade-off.

THE PROBLEM THIS EXISTS TO MAKE HONEST
======================================
`protocol` reports "delay 0 (0 ms)" and the model emits a label for every
input sample, so it is tempting to describe the decoder as zero-latency. The
artefact that actually ships does not behave that way.

A stateless ExecuTorch graph has no memory between invocations, so each call
must re-run the whole receptive field. Fed `RF + E - 1` samples it returns `E`
valid labels. To sustain 100 Hz you therefore invoke it once every `E` samples
— once every 250 ms at the default E = 25 — and every label in that batch
waits for the last sample of the batch to arrive.

    label age at emission  =  (T_batch_end - t) / fs  +  compute time
    newest label           =  compute time                 (~5 ms)
    oldest label           =  (E - 1)/fs + compute         (~245 ms at E=25)
    mean                   =  (E - 1)/2fs + compute        (~125 ms at E=25)

So the honest statement is not "0 ms" but "5 ms for the freshest label,
(E-1)*10 ms + 5 ms for the stalest". Shrinking E fixes the latency and
inflates the arithmetic by the overhead factor (RF + E - 1)/E; growing E does
the reverse. That trade-off is a curve, the curve is a genuine result about
stateless graph runtimes as a deployment target for TCNs, and the cached
ring-buffer implementation sits off the curve entirely at
(RF-independent cost, one-sample latency) — which is precisely the argument
for writing it.

CALIBRATION
-----------
Compute time is modelled as `a + b * MMAC_per_call` and fitted to measurements
you supply. With one point only the slope is fitted; with two or more, both
terms. Measure on the TARGET, not the dev box:

    python decoder/deploy_sweep.py --calib 25:5.03                # x86 dev box
    python decoder/deploy_sweep.py --calib 25:31.8 --calib 50:58.9 --budget 100

Usage:
    python decoder/deploy_sweep.py --electrodes 58 --bands 7
    python decoder/deploy_sweep.py --latex paper/ --figdir paper/
"""

import argparse
import csv
import os

import numpy as np

from model import build_model, deployment_cost, mac_report

EMITS = [1, 2, 5, 10, 25, 50, 100, 200]


def fit_timing(calib, macs_per_sample, rf, fs):
    """Least-squares a + b*MMAC_per_call from (emit, ms) measurements."""
    if not calib:
        return 0.0, 0.0
    mm = np.array([macs_per_sample * (rf + e - 1) / 1e6 for e, _ in calib])
    ms = np.array([t for _, t in calib])
    if len(calib) == 1:
        return 0.0, float(ms[0] / mm[0])
    A = np.stack([np.ones_like(mm), mm], axis=1)
    a, b = np.linalg.lstsq(A, ms, rcond=None)[0]
    return float(a), float(b)


def sweep(model, emits=EMITS, fs=100.0, a=0.0, b=0.0, budget=None):
    rep = mac_report(model, 1)
    rf, mps = rep["receptive_field"], rep["macs_per_sample"]
    rows = []
    for e in emits:
        dep = deployment_cost(model, emit=e, fs=fs)
        mmac_call = dep["macs_per_call"] / 1e6
        compute_ms = a + b * mmac_call
        period_ms = 1000.0 * e / fs
        rows.append(dict(
            emit=e,
            window=dep["window"],
            overhead=dep["overhead"],
            mmac_per_call=mmac_call,
            kmac_per_label=dep["macs_per_label_stateless"] / 1e3,
            mmac_s=dep["mmac_s_stateless"],
            compute_ms=compute_ms,
            period_ms=period_ms,
            duty=compute_ms / period_ms if period_ms > 0 else float("inf"),
            lat_min_ms=compute_ms,
            lat_mean_ms=0.5 * (e - 1) * 1000.0 / fs + compute_ms,
            lat_max_ms=(e - 1) * 1000.0 / fs + compute_ms,
            realtime=compute_ms <= period_ms,
            in_budget=(budget is None or
                       ((e - 1) * 1000.0 / fs + compute_ms) <= budget),
        ))
    # The cached kernel is unwritten, so its latency is not measured. The
    # linear fit extrapolates badly that far below the calibration range
    # (on an overhead-dominated board it goes negative), so quote an upper
    # bound instead: one cached step does strictly less work than an E=1
    # stateless call, which recomputes the whole receptive field.
    e1 = deployment_cost(model, emit=1, fs=fs)["macs_per_call"] / 1e6
    cached_ms = max(0.0, a + b * e1)
    cached = dict(emit="cached", window=1, overhead=1.0,
                  mmac_per_call=mps / 1e6,
                  kmac_per_label=mps / 1e3,
                  mmac_s=mps * fs / 1e6,
                  compute_ms=cached_ms,
                  period_ms=1000.0 / fs,
                  duty=float("nan"),
                  lat_min_ms=cached_ms,
                  lat_mean_ms=cached_ms,
                  lat_max_ms=cached_ms,
                  latency_is_bound=True,
                  realtime=True, in_budget=True)
    return rows, cached, rf, mps


def print_table(rows, cached, rf, budget):
    print(f"\nreceptive field {rf} samples ({rf * 10} ms of history)")
    print(f"{'emit':>6}{'window':>8}{'x RF':>7}{'MMAC/call':>11}"
          f"{'kMAC/lbl':>10}{'MMAC/s':>9}{'compute':>9}{'duty':>7}"
          f"{'lat mean':>10}{'lat max':>9}  ")
    print("-" * 90)
    for r in rows:
        flag = "" if r["realtime"] else "  OVERRUN"
        if r["realtime"] and budget is not None and not r["in_budget"]:
            flag = "  >budget"
        print(f"{r['emit']:>6}{r['window']:>8}{r['overhead']:>7.1f}"
              f"{r['mmac_per_call']:>11.2f}{r['kmac_per_label']:>10.1f}"
              f"{r['mmac_s']:>9.1f}{r['compute_ms']:>8.1f}m{r['duty']:>7.2f}"
              f"{r['lat_mean_ms']:>10.1f}{r['lat_max_ms']:>9.1f}{flag}")
    print("-" * 90)
    c = cached
    print(f"{'cached':>6}{1:>8}{1.0:>7.1f}{c['mmac_per_call']:>11.4f}"
          f"{c['kmac_per_label']:>10.1f}{c['mmac_s']:>9.1f}"
          f"{c['compute_ms']:>8.2f}m{'—':>7}{c['lat_mean_ms']:>10.2f}"
          f"{c['lat_max_ms']:>9.2f}  <- ring-buffer firmware, not yet written; "
          f"latency is an upper bound (E=1 call time)")
    print("\nlatency = age of a label when it leaves the decoder, i.e. batching")
    print("delay + compute. duty = compute time / batch period; > 1 overruns.")


def write_latex(outdir, rows, cached, rf, budget):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "tab_deploy.tex"), "w") as f:
        f.write("% AUTO-GENERATED by deploy_sweep.py --latex. Do not edit.\n")
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
        f.write("$E$ & Window & Overhead & kMAC/label & MMAC/s & "
                "Max latency (ms) \\\\\n\\midrule\n")
        for r in rows:
            mark = "" if r["in_budget"] and r["realtime"] else "$^{\\dagger}$"
            f.write(f"{r['emit']} & {r['window']} & "
                    f"{r['overhead']:.1f}$\\times$ & {r['kmac_per_label']:.0f} & "
                    f"{r['mmac_s']:.1f} & {r['lat_max_ms']:.0f}{mark} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"cached & 1 & 1.0$\\times$ & {cached['kmac_per_label']:.1f} & "
                f"{cached['mmac_s']:.1f} & $\\leq${cached['lat_max_ms']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, "numbers_deploy.tex"), "w") as f:
        f.write("% AUTO-GENERATED by deploy_sweep.py --latex.\n")
        f.write(f"\\newcommand{{\\RF}}{{{rf}}}\n")
        f.write(f"\\newcommand{{\\RFms}}{{{rf * 10}}}\n")
        best = [r for r in rows if r["realtime"] and r["in_budget"]]
        if best:
            r = max(best, key=lambda r: r["emit"])
            f.write(f"\\newcommand{{\\BestEmit}}{{{r['emit']}}}\n")
            f.write(f"\\newcommand{{\\BestEmitMMAC}}{{{r['mmac_s']:.1f}}}\n")
            f.write(f"\\newcommand{{\\BestEmitLat}}{{{r['lat_max_ms']:.0f}}}\n")
        d = [r for r in rows if r["emit"] == 25]
        if d:
            f.write(f"\\newcommand{{\\DefaultEmitLat}}{{{d[0]['lat_max_ms']:.0f}}}\n")
            f.write(f"\\newcommand{{\\DefaultEmitMMAC}}{{{d[0]['mmac_s']:.1f}}}\n")
        f.write(f"\\newcommand{{\\CachedMMAC}}{{{cached['mmac_s']:.2f}}}\n")
        f.write(f"\\newcommand{{\\CachedKMAC}}{{{cached['kmac_per_label']:.1f}}}\n")
    print(f"wrote tab_deploy.tex, numbers_deploy.tex to {outdir}/")


def make_figure(figdir, rows, cached, budget):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lat = [r["lat_max_ms"] for r in rows]
    mm = [r["mmac_s"] for r in rows]
    ax.plot(lat, mm, "o-", color="#1f4e79", lw=1.4, ms=4, zorder=3,
            label="stateless .pte")
    for r in rows:
        if r["emit"] in (1, 5, 25, 100):
            ax.annotate(f"$E$={r['emit']}", (r["lat_max_ms"], r["mmac_s"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.plot([cached["lat_max_ms"]], [cached["mmac_s"]], "*", color="#c0392b",
            ms=13, zorder=4, label="cached (latency upper bound)")
    if budget:
        ax.axvline(budget, ls="--", c="0.5", lw=1)
        # label runs up the budget line from the bottom, where the plot is
        # empty, so the top-right legend cannot cover it
        # x in data units, y in axes fraction: independent of the log scale
        ax.text(budget, 0.03, f"{budget:.0f} ms budget ", transform=ax.get_xaxis_transform(),
                fontsize=7, va="bottom", ha="right", rotation=90, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Worst-case label latency (ms)", fontsize=8)
    ax.set_ylabel("Sustained MMAC/s", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    # top-right is empty: the curve runs top-left to bottom-right, the star
    # sits bottom-left, and the budget label sits at the bottom of its line
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    out = os.path.join(figdir, "fig_pareto.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--electrodes", type=int, default=58)
    ap.add_argument("--bands", type=int, default=7)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--emits", nargs="*", type=int, default=EMITS)
    ap.add_argument("--calib", action="append", default=[],
                    metavar="EMIT:MS",
                    help="measured median ms per call, e.g. 25:5.03. Repeat.")
    ap.add_argument("--budget", type=float, default=100.0,
                    help="latency budget in ms for the feasibility flag")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--latex", default=None)
    ap.add_argument("--figdir", default=None)
    a = ap.parse_args()

    calib = []
    for c in a.calib:
        e, t = c.split(":")
        calib.append((int(e), float(t)))

    model = build_model(a.electrodes, n_bands=a.bands, preset=a.preset).eval()
    rep = mac_report(model, 1)
    coef_a, coef_b = fit_timing(calib, rep["macs_per_sample"],
                                rep["receptive_field"], a.fs)
    print("=" * 90)
    print(f"ECoGFlexNet '{a.preset}' | {a.electrodes} electrodes x {a.bands} "
          f"bands | {rep['params']:,} params")
    if calib:
        print(f"timing model: {coef_a:.2f} + {coef_b:.3f} x MMAC ms/call, "
              f"fitted to {len(calib)} measurement(s) {calib}")
    else:
        print("no --calib given: compute time reported as 0; latency numbers "
              "are batching delay only")
    print("=" * 90)

    rows, cached, rf, _ = sweep(model, a.emits, a.fs, coef_a, coef_b, a.budget)
    print_table(rows, cached, rf, a.budget)

    ok = [r for r in rows if r["realtime"] and r["in_budget"]]
    if ok:
        best = max(ok, key=lambda r: r["emit"])
        print(f"\nLargest batch meeting a {a.budget:.0f} ms budget: E={best['emit']} "
              f"at {best['mmac_s']:.1f} MMAC/s ({best['lat_max_ms']:.0f} ms worst case).")
        print(f"The cached implementation would do the same job at "
              f"{cached['mmac_s']:.2f} MMAC/s and {cached['lat_max_ms']:.3f} ms — "
              f"{best['mmac_s'] / cached['mmac_s']:.0f}x less arithmetic.")
    else:
        print(f"\nNo batch size meets the {a.budget:.0f} ms budget on this "
              f"hardware. Report that, and report the cached target.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")
    if a.latex:
        write_latex(a.latex, rows, cached, rf, a.budget)
    if a.figdir:
        make_figure(a.figdir, rows, cached, a.budget)


if __name__ == "__main__":
    main()
