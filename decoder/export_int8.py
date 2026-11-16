"""
export_int8.py — QAT/PTQ + ExecuTorch (XNNPACK) export for ECoGFlexNet, and an
honest INT8 score on the full test recording.

WHY THE MODEL LOWERS CLEANLY TO XNNPACK
=======================================
An earlier model version failed to lower with

    Failed to create static slice node 45 with code: xnn_status_invalid_parameter

which traced to two slice patterns. model.py removes the first outright and
keeps the second optional:

  1. `h[:, :, :, -1:]` in the head lowered to an open-ended NEGATIVE
     slice_copy(dim=3, start=-1, end=INT64_MAX). model.py's forward returns
     EVERY timestep, so there is no head slice at all — not a negative one,
     not a positive one. The host keeps whichever tail it wants. This is the
     one pattern that is known to have broken.

  2. The causal convolutions. `causal_mode="pad"` uses F.pad + padding=0,
     emitting constant_pad_nd instead of slice_copy.

  3. The per-band split `x[:, b:b+1]` is the only slice left, and it is the
     benign kind: static, positive start, bounded end, n_bands of them.
     If a future toolchain rejects even those, `--spatial grouped` removes
     them by folding the seven convolutions into one grouped convolution that
     reads THE SAME WEIGHT TENSOR, so no retraining is needed. Downside: it
     becomes a depthwise conv with multiplier S, which model.py's comments
     flag as unreliably delegated. `--probe` tries all four combinations and
     reports which ones load.

WHY THE EXPORT WINDOW IS NOT THE TRAINING WINDOW
------------------------------------------------
A causal TCN needs `receptive_field` samples of history to produce ONE
correct output, and it produces a correct output for every sample after that.
Exporting a window of RF + E - 1 samples therefore yields E usable
predictions per invocation, and the arithmetic is amortised over all E.

    E = 1    one label per call. 100% overhead.
    E = 25   250 ms of labels per call, 25x amortisation, 250 ms of jitter.
    E = 50   500 ms of labels per call.

E trades sustained compute against how often the device wakes up; it does
NOT change accuracy, because every emitted sample has the same full history
either way. `--emit` sets it (default 10, the setting used in the paper).

QAT
---
`is_per_channel=True`. A quant gap of +0.0005 measured earlier with dense
convolutions does not transfer: a depthwise kernel's per-output-
channel weight ranges routinely differ by 100x, and one shared scale crushes
the narrow channels to a couple of levels. Per-channel is free on XNNPACK.
`--score` re-measures the gap on the real test recording instead of assuming.

USAGE
-----
    python decoder/export_int8.py --probe            # ~1 min: which lowering loads?
    python decoder/export_int8.py                    # export every checkpoint
    python decoder/export_int8.py --emit 50 --score  # bigger chunks + FP32/INT8 gap
    python decoder/export_int8.py --causal slice --spatial grouped
"""

import argparse
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Use the flatc that ships with the executorch wheel in whichever environment
# is running this script (flatc.exe on Windows, flatc elsewhere).
import sys                                             # noqa: E402
_flatc = os.path.join(os.path.dirname(sys.executable),
                      "flatc.exe" if os.name == "nt" else "flatc")
if os.path.exists(_flatc):
    os.environ.setdefault("FLATC_EXECUTABLE", _flatc)

import csv                                             # noqa: E402
import torch.export                                    # noqa: E402

from model import build_model, mac_report           # noqa: E402
import protocol as P                                # noqa: E402
import train as T                                   # noqa: E402


def remove_dropout(m):
    for name, child in m.named_children():
        if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            setattr(m, name, nn.Identity())
        else:
            remove_dropout(child)


def export_window(model, emit):
    """Samples the exported graph must be fed to emit `emit` valid outputs."""
    return model.receptive_field + emit - 1


def qat_finetune(qat_model, loader, criterion, supervise_from,
                 epochs=30, lr=1e-4, freeze_at=0.6):
    """[FIX-14] Fine-tune a prepared QAT graph module.

    Cosine schedule; after `freeze_at` of the epochs the observers are disabled
    and BN switches to its running statistics, so the last epochs optimise the
    weights against a fixed quantisation grid. pt2e has no freeze_bn_stats:
    moving the exported graph to eval mode is how BN statistics are frozen, and
    gradients still flow through the fake-quant nodes.

    Returns (model, loss history, converged flag).
    """
    from torchao.quantization.pt2e import (disable_observer,
                                           move_exported_model_to_eval,
                                           move_exported_model_to_train)
    opt = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr * 0.02)
    freeze_epoch = int(freeze_at * epochs)
    hist = []
    move_exported_model_to_train(qat_model)
    for ep in range(epochs):
        if ep == freeze_epoch:
            qat_model.apply(disable_observer)
            move_exported_model_to_eval(qat_model)
            print(f"      observers + BN frozen at epoch {ep + 1}")
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            out = qat_model(xb).transpose(1, 2)
            loss = criterion(out[:, supervise_from:], yb[:, supervise_from:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), T.GRAD_CLIP)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        hist.append(tot / max(n, 1))
        if (ep + 1) % 5 == 0 or ep in (0, freeze_epoch):
            print(f"      QAT epoch {ep + 1:>3}/{epochs} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | loss {hist[-1]:.4f}")
    early, late = np.mean(hist[:5]), np.mean(hist[-5:])
    converged = late <= early * 0.98
    if not converged:
        print(f"      WARNING: QAT loss did not improve ({early:.4f} -> "
              f"{late:.4f}). Report this as PTQ with observer calibration, "
              f"not QAT.")
    return qat_model, hist, converged


def record_int8(path, subject, seed, s, fp32_r, pte_kb, lat_p50, lat_p95,
                emit, qat_converged):
    """[FIX-15] One row per export; score.py --int8-csv reads it."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["subject", "seed", "official_r", "fp32_official_r",
                        "quant_gap", "official_rmse", "pte_kb", "lat_p50_ms",
                        "lat_p95_ms", "emit", "qat_converged",
                        *[f"r_{n}" for n in P.FINGER_NAMES]])
        w.writerow([subject, seed, f"{s['official_r']:.6f}", f"{fp32_r:.6f}",
                    f"{fp32_r - s['official_r']:+.6f}",
                    f"{s['official_rmse']:.6f}", f"{pte_kb:.2f}",
                    f"{lat_p50:.3f}", f"{lat_p95:.3f}", emit, int(qat_converged),
                    *[f"{v:.6f}" for v in s["r"]]])


# ----------------------------------------------------------------------
def lower_to_pte(model, in_rows, window, out_path, qat_loader=None,
                 qat_epochs=0, qat_lr=1e-4, batch_size=32, criterion=None,
                 supervise_from=None):
    """FP32 module -> QAT -> convert -> XNNPACK partition -> .pte on disk.

    Returns (size in KB, qat_converged). qat_converged is False when no QAT
    ran (probe / calibration only) or when the QAT loss did not move.
    """
    from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from executorch.exir import to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner)

    remove_dropout(model)
    model.train()

    example = torch.randn(batch_size, in_rows, window)
    batch_dim = torch.export.Dim("batch", min=1, max=batch_size)
    exported = torch.export.export(model, (example,),
                                   dynamic_shapes=({0: batch_dim},))

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(
        is_qat=True, is_per_channel=True))

    qat_model = prepare_qat_pt2e(exported.module(), quantizer)
    allow_exported_model_train_eval(qat_model)
    qat_model.train()

    converged = False
    if qat_epochs and qat_loader is not None:
        # The graph module's `forward` IS model's dense path, so QAT
        # fine-tunes on exactly the same objective as FP32 training.
        qat_model, _, converged = qat_finetune(
            qat_model, qat_loader, criterion, supervise_from,
            epochs=qat_epochs, lr=qat_lr)
    elif qat_loader is not None:
        # PTQ: calibrate observers on real training windows, no weight
        # updates. This isolates the cost of quantisation from the extra
        # fine-tuning that QAT also performs.
        from torchao.quantization.pt2e import move_exported_model_to_eval
        move_exported_model_to_eval(qat_model)
        with torch.no_grad():
            for i, (xb, _) in enumerate(qat_loader):
                qat_model(xb)
                if i + 1 >= 50:
                    break
        print("      PTQ: observers calibrated on 50 training batches")
    else:
        with torch.no_grad():
            qat_model(torch.randn(2, in_rows, window))   # probe only: any ranges

    qat_model.eval()
    quantized = convert_pt2e(qat_model)

    static_example = torch.randn(1, in_rows, window)
    quantized_ep = torch.export.export(quantized, (static_example,))
    edge = to_edge(quantized_ep).to_backend(XnnpackPartitioner())
    program = edge.to_executorch()

    with open(out_path, "wb") as fh:
        fh.write(program.buffer)
    return os.path.getsize(out_path) / 1024.0, converged


def try_load(pte_path, in_rows, window):
    """Load the .pte and run one inference. This is where node 45 blew up."""
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(pte_path).load_method("forward")
    return tuple(method.execute((torch.randn(1, in_rows, window),))[0].shape)


# ----------------------------------------------------------------------
def pte_predict_region(pte_path, x, lo, hi, window, emit):
    """Walk the exported INT8 graph across a region, chunk by chunk.

    Every emitted sample sees `receptive_field` real input samples, so this is
    the same function the FP32 path computes — no zero-padded warm-up is
    involved anywhere inside [lo, hi) as long as lo >= receptive_field, which
    holds for the test recording (it starts 400 s into the session).

    Returns (hi - lo, 5) and the per-invocation latencies in ms.
    """
    import time
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(pte_path).load_method("forward")
    out = np.empty((hi - lo, 5), dtype=np.float64)
    lat = []
    t = lo
    while t < hi:
        e = min(emit, hi - t)
        start = t + e - window
        assert start >= 0, "region starts before one full receptive field"
        chunk = x[:, start:t + e].unsqueeze(0).contiguous()
        t0 = time.perf_counter()
        y = method.execute((chunk,))[0]
        lat.append((time.perf_counter() - t0) * 1000.0)
        out[t - lo:t - lo + e] = y[0, :, -e:].T.detach().cpu().numpy()
        t += e
    return out, lat


# ----------------------------------------------------------------------
def probe(data_dir, preset, emit, modes=("pad", "slice"),
          spatials=("split", "grouped")):
    print("=" * 76)
    print("BACKEND PROBE — which lowering survives XNNPACK?")
    print("=" * 76)

    shapes = []
    for sub in P.SUBJECTS:
        path = P.prep_path(sub, data_dir)
        if not os.path.exists(path):
            print(f"  {sub}: missing {path}, skipping")
            continue
        d = torch.load(path, weights_only=False)
        shapes.append((sub, int(d["n_electrodes"]), int(d["n_bands"])))
    if not shapes:
        print("  No prepared data found. Run decoder/prep.py first.")
        return

    ok = []
    for causal in modes:
        for spatial in spatials:
            for sub, n_el, n_bands in shapes:
                in_rows = n_el * n_bands
                m = build_model(n_el, n_bands=n_bands, preset=preset)
                m.set_causal_mode(causal).set_spatial_mode(spatial)
                w = export_window(m, emit)
                tag = f"causal={causal:<5} spatial={spatial:<7} {sub}"
                with tempfile.TemporaryDirectory() as td:
                    pte = os.path.join(td, "probe.pte")
                    try:
                        kb, _ = lower_to_pte(m, in_rows, w, pte)
                        shp = try_load(pte, in_rows, w)
                        print(f"  {tag}: OK   {kb:6.2f} KB  out {shp}")
                        ok.append((causal, spatial))
                    except Exception as exc:                      # noqa: BLE001
                        print(f"  {tag}: FAIL {type(exc).__name__}: "
                              f"{str(exc)[:110]}")

    print("\n" + "=" * 76)
    if ok:
        c, s = ok[0]
        print(f"Working combinations: {sorted(set(ok))}")
        print(f"Export with:  python export_int8.py --causal {c} --spatial {s}")
    else:
        print("Nothing lowered. The remaining suspects, in order:")
        print("  * the grouped BatchNorm folding — try use_gate=False to drop")
        print("    the sigmoid, then re-probe, to isolate the channel gate;")
        print("  * the depthwise output smoother — build with smooth_kernel=1")
        print("    and re-probe (it costs ~nothing to test and the model still")
        print("    trains, it just loses the learned post-filter);")
        print("  * the dynamic batch dimension — re-export with a static")
        print("    batch of 1, which is all a device ever uses anyway.")
    print("=" * 76)


# ----------------------------------------------------------------------
def run(args):
    script_dir = args.ckpt_dir
    os.makedirs(script_dir, exist_ok=True)
    criterion = T.CorrMSELoss()

    for sub in args.subjects:
        if not os.path.exists(P.prep_path(sub, args.data_dir)):
            print(f"\n[{sub}] no prep file. Skipping.")
            continue
        bundle = T.subject_bundle(sub, args.data_dir)
        d, x = bundle["d"], bundle["x"]
        n_el, n_bands = bundle["n_el"], bundle["n_bands"]
        in_rows = n_el * n_bands

        for seed in args.seeds:
            ck_path = P.ckpt_path(script_dir, sub, seed, args.tag)
            if not os.path.exists(ck_path):
                print(f"\n[{sub} s{seed}] no checkpoint. Skipping.")
                continue

            print(f"\n{'=' * 76}\n{sub} | seed {seed} | causal={args.causal} "
                  f"spatial={args.spatial} emit={args.emit}\n{'=' * 76}")
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            # [FIX-7] restore architecture deltas (no-gate, no-smooth, ...)
            model = build_model(n_el, n_bands=n_bands,
                                preset=ck.get("preset", "tiny"),
                                **ck.get("trunk_kw", {}))
            model.load_state_dict(ck["state_dict"])
            model.set_causal_mode(args.causal).set_spatial_mode(args.spatial)
            model.eval()

            w = export_window(model, args.emit)
            rep = mac_report(model, w)
            # The exported graph is specialised to `w`, not P.WINDOW. QAT must
            # see windows of exactly that width or the stem's view() fails.
            # v3.1: bundle["train"] is a list of interleaved intervals.
            qat_ds = P.SegmentDataset(x, d["dg"], bundle["train"],
                window=w, step=T.TRAIN_STEP,
                delay=P.DELAY, augment=True,
                n_electrodes=n_el, n_bands=n_bands)
            loader = DataLoader(qat_ds, batch_size=args.batch_size,
                shuffle=True, drop_last=True)
            print(f"  export window {w} samples ({w * 10} ms) -> {args.emit} "
                  f"labels/call | {rep['macs_per_sample'] * w / 1e6:.2f} MMAC "
                  f"per call, {rep['macs_per_sample'] / 1e3:.1f} kMAC/label")

            # Confirm the reloaded model still reproduces its recorded
            # validation score before spending time on QAT. If this drifts,
            # the state dict and the architecture have diverged.
            val = P.score_regions(model, x, d, bundle["val"])["official_r"]
            rec = ck.get("val_official_r", float("nan"))
            print(f"  FP32 val official r {val:.4f} (checkpoint recorded {rec:.4f})")
            if rec == rec and abs(val - rec) > 1e-3:
                print("  WARNING: mismatch — check before publishing.")

            fp32_test = None
            if args.score:
                fp32_test = P.score_region(model, x, d, bundle["test"])
                P.print_fingers("FP32  [official test]", fp32_test)

            out_path = P.pte_path(script_dir, sub, seed, args.pte_tag or args.tag)
            T.set_seed(seed + 2000)       # QAT shuffling + augmentation
            try:
                kb, converged = lower_to_pte(model, in_rows, w, out_path,
                                  qat_loader=loader, qat_epochs=args.qat_epochs,
                                  qat_lr=args.qat_lr, batch_size=args.batch_size,
                                  criterion=criterion,
                                  supervise_from=model.receptive_field)
                shp = try_load(out_path, in_rows, w)
                print(f"  saved {os.path.basename(out_path)} ({kb:.2f} KB), "
                      f"loaded OK, output {shp}")
            except Exception as exc:                              # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                print("  Run `python decoder/export_int8.py --probe` to isolate.")
                continue

            if args.score:
                lo, hi = bundle["test"]
                pred, lat = pte_predict_region(out_path, x, lo, hi, w, args.emit)
                s = P.official_score(pred, d["dg_raw"], lo, P.DECIM * lo,
                                     d["dg_raw"].shape[0], mode="zoh",
                                     dg_mean=d["dg_mean"].numpy().ravel(),
                                     dg_std=d["dg_std"].numpy().ravel())
                P.print_fingers("INT8  [official test]", s)
                lat = sorted(lat[5:]) or sorted(lat)
                p50 = lat[len(lat) // 2]
                p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
                print(f"      quant gap {fp32_test['official_r'] - s['official_r']:+.4f} "
                      f"| latency/call median {p50:.2f} ms p95 {p95:.2f} ms "
                      f"for {args.emit} labels (THIS host, not the board)")
                if fp32_test["official_r"] - s["official_r"] > 0.03:
                    print("      gap > 0.03: inspect the depthwise weight ranges "
                          "before blaming the architecture.")
                # [FIX-16] the artefact and the latency claim cannot drift apart
                worst = (args.emit - 1) * 1000.0 / P.FS + p50
                if worst > args.latency_budget_ms:
                    print(f"      WARNING: worst-case label age {worst:.0f} ms "
                          f"exceeds the {args.latency_budget_ms:.0f} ms budget "
                          f"at emit={args.emit}.")
                record_int8(args.int8_csv, sub, seed, s, fp32_test["official_r"],
                            kb, p50, p95, args.emit, converged)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--causal", default="pad", choices=["pad", "slice"])
    p.add_argument("--spatial", default="grouped", choices=["split", "grouped"])
    p.add_argument("--emit", type=int, default=10,         # [FIX-16]
                   help="labels produced per invocation (amortisation factor)")
    p.add_argument("--latency-budget-ms", type=float, default=100.0)
    p.add_argument("--int8-csv", default="int8_results.csv")   # [FIX-15]
    p.add_argument("--data-dir", default=P.DATA_DIR)
    p.add_argument("--ckpt-dir", default=P.CKPT_DIR,
                   help="checkpoints are read from, and .pte files written to, here")
    p.add_argument("--subjects", nargs="*", default=P.SUBJECTS)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--tag", default="flexnet")
    p.add_argument("--pte-tag", default=None,
                   help="name the .pte files with this tag instead of --tag, "
                        "so a PTQ export does not overwrite the QAT artefacts")
    p.add_argument("--preset", default="tiny")
    p.add_argument("--qat-epochs", type=int, default=30)      # [FIX-14]
    p.add_argument("--qat-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    p.add_argument("--score", action="store_true",
                   help="measure the FP32->INT8 gap on the official test set")
    a = p.parse_args()

    if a.probe:
        probe(a.data_dir, a.preset, a.emit)
    else:
        run(a)
