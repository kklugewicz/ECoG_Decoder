"""
train.py — trains ECoGFlexNet for BCI Competition IV ds4.

DESIGN, UNCHANGED FROM v3.0
===========================
1. Model selection uses the PUBLISHED metric (`protocol.official_score`),
   not a smoothed/decimated proxy.
2. Cross-subject pretraining of the shape-independent trunk, then per-subject
   fine-tuning. `--no-pretrain` is the ablation.
3. Dense supervision that is actually causal, starting at the receptive field
   so every supervised timestep has real history rather than zero padding.
4. Loss weighted toward the scored fingers; ring kept as a reduced-weight
   auxiliary task because it is mechanically coupled to middle and little.
5. SWA over the last epochs, accepted only if validation agrees.

v3.1 CHANGES (audit fixes)
==========================
[FIX-5]  SELECTION ON INTERLEAVED BLOCKS, NOT ONE TAIL.
         Validation now calls `protocol.score_regions` over four held-out
         blocks spread through the recording. Same held-out fraction, same
         training cost, far less selection variance. v3.0's single 80 s tail
         produced a validation ranking that was uncorrelated with the test
         ranking for sub1 and inverted for sub2 — with three seeds that is
         indistinguishable from picking at random.

[FIX-6]  DEFAULT FIVE SEEDS, AND THE VALIDATION->TEST RELATIONSHIP IS SAVED.
         Per-subject seed spread was +-0.04 r against a claimed margin of
         ~0.05 over the competition winner. Three seeds cannot separate
         those. Each checkpoint now stores its per-block validation scores so
         `score.py` can report the val->test correlation directly instead
         of leaving the reader to assume selection worked.

[FIX-7]  CHECKPOINTS CARRY THEIR ARCHITECTURE.
         v3.0 stored only `preset`, so a model trained with `--no-gate` could
         not be reloaded by `score.py` without hand-editing. The full
         trunk keyword set is now saved and restored.

[FIX-8]  ARCHITECTURE ABLATION FLAGS: --no-gate, --no-smooth, --skip-last.
         The paper claims the causal channel gate fixes a look-ahead leak
         present in model_v2's ECA layer. That claim needs a row.

Usage:
    python decoder/prep.py
    python decoder/train.py                       # pretrain + per-subject finetune
    python decoder/train.py --seeds 0 --subjects sub1
    python decoder/train.py --no-pretrain         # ablation
    python decoder/score.py
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model, build_multi, mac_report
import protocol as P

# ======================================================================
PRESET = "tiny"
EPOCHS = 160
PRETRAIN_EPOCHS = 60
BATCH_SIZE = 24
TRAIN_STEP = 50            # processed samples between training windows (0.5 s)
LR = 3e-3
PRETRAIN_LR = 3e-3
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 5
MIN_LR = 1e-5
MSE_WEIGHT = 0.25
RING_WEIGHT = 0.4
SWA_EPOCHS = 25
GRAD_CLIP = 5.0
VAL_EVERY = 5
SEEDS = [0, 1, 2, 3, 4]    # [FIX-6]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)


def trunk_overrides(args):
    """[FIX-7]/[FIX-8] the architecture deltas that must survive a save/load."""
    kw = {}
    if getattr(args, "no_gate", False):
        kw["use_gate"] = False
    if getattr(args, "no_smooth", False):
        kw["smooth_kernel"] = 1
    if getattr(args, "skip_last", False):
        kw["skip_mode"] = "last"
    if getattr(args, "eca_gate", False):
        kw["gate_kind"] = "eca"
    return kw


# ======================================================================
# Loss
# ======================================================================
class CorrMSELoss(nn.Module):
    """(1 - Pearson r) + MSE, per finger, pooled over every supervised point.

    Pearson rather than CCC: the metric is Pearson, and CCC additionally
    penalises mean and scale offsets that the metric is invariant to. The MSE
    term is kept only to anchor scale so RMSE stays reportable.
    """

    def __init__(self, mse_weight=MSE_WEIGHT, ring_weight=RING_WEIGHT):
        super().__init__()
        self.mse_weight = mse_weight
        w = torch.ones(5)
        w[P.EXCLUDED_FINGER_IDX] = ring_weight
        self.register_buffer("w", w / w.sum())

    def forward(self, pred, true):
        pred = pred.reshape(-1, pred.shape[-1]).float()
        true = true.reshape(-1, true.shape[-1]).float()
        vx = pred - pred.mean(0, keepdim=True)
        vy = true - true.mean(0, keepdim=True)
        r = (vx * vy).sum(0) / (vx.pow(2).sum(0).sqrt() * vy.pow(2).sum(0).sqrt() + 1e-8)
        corr = (self.w * (1.0 - r)).sum()
        mse = (self.w * ((pred - true) ** 2).mean(0)).sum()
        return self.mse_weight * mse + (1.0 - self.mse_weight) * corr


def lr_at(epoch, total, base):
    if epoch < WARMUP_EPOCHS:
        return base * (epoch + 1) / WARMUP_EPOCHS
    p = (epoch - WARMUP_EPOCHS) / max(1, total - WARMUP_EPOCHS)
    return MIN_LR + 0.5 * (base - MIN_LR) * (1.0 + np.cos(np.pi * p))


def clone_state(m):
    return {k: v.detach().clone().cpu() for k, v in m.state_dict().items()}


def average_states(states):
    out = {}
    for k in states[0]:
        v0 = states[0][k]
        out[k] = (torch.stack([s[k].float() for s in states]).mean(0).to(v0.dtype)
                  if v0.is_floating_point() else v0.clone())
    return out


def reestimate_bn(model, loader, device, max_batches=30):
    """Averaged weights describe a model whose BatchNorm running statistics
    were never measured. Recompute them with a cumulative average."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        return
    saved = [(m.momentum, m.training) for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            model(xb.to(device))
            if i + 1 >= max_batches:
                break
    for m, (mom, tr) in zip(bns, saved):
        m.momentum = mom
        m.train(tr)


# ======================================================================
# Data plumbing
# ======================================================================
def subject_bundle(sub, data_dir, augment=True, train_step=TRAIN_STEP,
                   include_val_in_train=False):
    d = P.load_prep(sub, data_dir)
    x = P.normalised_features(d)
    train_iv, val_iv, test_iv = P.split_bounds(d)
    fit_iv = sorted(train_iv + val_iv) if include_val_in_train else train_iv
    n_el, n_bands = int(d["n_electrodes"]), int(d["n_bands"])
    ds = P.SegmentDataset(x, d["dg"], fit_iv, P.WINDOW, train_step,
                          P.DELAY, augment=augment,
                          n_electrodes=n_el, n_bands=n_bands)
    ds_clean = P.SegmentDataset(x, d["dg"], fit_iv, P.WINDOW, train_step,
                                P.DELAY, augment=False,
                                n_electrodes=n_el, n_bands=n_bands)
    return dict(d=d, x=x, ds=ds, ds_clean=ds_clean, n_el=n_el, n_bands=n_bands,
                val=val_iv, test=test_iv, train=fit_iv)


def run_epoch(forward, loader, opt, crit, device, supervise_from):
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        out = forward(xb).transpose(1, 2)              # (B, T, 5)
        loss = crit(out[:, supervise_from:], yb[:, supervise_from:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in opt.param_groups[0]["params"] if p.requires_grad], GRAD_CLIP)
        opt.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


# ======================================================================
# Stage 1 — cross-subject trunk
# ======================================================================
def pretrain(bundles, seed, device, preset, epochs=PRETRAIN_EPOCHS, tkw=None):
    print(f"\n{'=' * 72}\nSTAGE 1 — cross-subject trunk pretraining "
          f"({', '.join(bundles)}) | seed {seed}\n{'=' * 72}")
    set_seed(seed)
    counts = {s: b["n_el"] for s, b in bundles.items()}
    n_bands = next(iter(bundles.values()))["n_bands"]
    model = build_multi(counts, n_bands=n_bands, preset=preset,
                        **(tkw or {})).to(device)
    crit = CorrMSELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR,
                            weight_decay=WEIGHT_DECAY)
    sup = model.trunk.receptive_field

    loaders = {s: DataLoader(b["ds"], batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True) for s, b in bundles.items()}
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep, epochs, PRETRAIN_LR)
        model.train()
        # Round-robin over subjects so every optimizer step sees one
        # recording and consecutive steps see different ones. Batching
        # subjects together is not possible: the input row count differs.
        its = {s: iter(l) for s, l in loaders.items()}
        total, n = 0.0, 0
        while its:
            for s in list(its):
                try:
                    xb, yb = next(its[s])
                except StopIteration:
                    del its[s]
                    continue
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(xb, s).transpose(1, 2)
                loss = crit(out[:, sup:], yb[:, sup:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                total += float(loss.detach())
                n += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    pretrain epoch {ep + 1:>3}/{epochs} | "
                  f"lr {lr_at(ep, epochs, PRETRAIN_LR):.2e} | loss {total / max(n, 1):.4f}")
    return model


# ======================================================================
# Stage 2 — per-subject
# ======================================================================
def train_subject(sub, bundle, seed, device, preset, script_dir,
                  init_from=None, epochs=EPOCHS, lr=LR, tag="flexnet",
                  fixed_epochs=None, tkw=None):
    tkw = tkw or {}
    print(f"\n{'=' * 72}\n{sub} | seed {seed} | preset {preset} | "
          f"{'finetune from shared trunk' if init_from else 'from scratch'}"
          f"{' | ' + json.dumps(tkw) if tkw else ''}"
          f"\n{'=' * 72}")
    set_seed(seed + 1000)

    model = build_model(bundle["n_el"], n_bands=bundle["n_bands"],
                        preset=preset, **tkw)
    if init_from is not None:
        src = init_from.single(sub)
        model.load_state_dict(copy.deepcopy(src.state_dict()))
    model = model.to(device)

    rep = mac_report(model, P.WINDOW)
    print(f"  {bundle['n_el']} electrodes x {bundle['n_bands']} bands = "
          f"{bundle['n_el'] * bundle['n_bands']} input rows")
    print(f"  {rep['params']:,} params | RF {rep['receptive_field']} steps "
          f"({rep['receptive_field'] * 10} ms) | "
          f"{rep['macs_per_sample'] / 1e3:.1f} kMAC/emitted sample "
          f"({rep['sustained_streaming_mmac_s']:.2f} MMAC/s @100 Hz)")
    print(f"  {len(bundle['ds'])} train windows x "
          f"{P.WINDOW - rep['receptive_field']} supervised steps each | "
          f"{len(bundle['val'])} held-out validation blocks "
          f"({P.interval_len(bundle['val'])} samples)")

    loader = DataLoader(bundle["ds"], batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True)
    bn_loader = DataLoader(bundle["ds_clean"], batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=True)
    crit = CorrMSELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sup = model.receptive_field

    best_r, best_state, best_epoch = -np.inf, clone_state(model), -1
    best_blocks = []
    swa, t0 = [], time.time()

    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep, epochs, lr)
        model.train()
        loss = run_epoch(model, loader, opt, crit, device, sup)

        due = (ep + 1) % VAL_EVERY == 0 or ep == epochs - 1 or ep == 0
        if fixed_epochs is None and due:
            # [FIX-5] averaged over interleaved held-out blocks
            s = P.score_regions(model, bundle["x"], bundle["d"], bundle["val"],
                                device=device)
            if s["official_r"] > best_r:
                best_r, best_epoch = s["official_r"], ep
                best_state = clone_state(model)
                best_blocks = list(s["blocks"])
            print(f"    epoch {ep + 1:>3}/{epochs} | lr {lr_at(ep, epochs, lr):.2e} | "
                  f"loss {loss:.4f} | val official r {s['official_r']:.4f} "
                  f"[{' '.join(f'{b:.3f}' for b in s['blocks'])}] "
                  f"| best {best_r:.4f} (ep {best_epoch + 1})")
        elif (ep + 1) % 20 == 0:
            print(f"    epoch {ep + 1:>3}/{epochs} | loss {loss:.4f}")

        if ep >= epochs - SWA_EPOCHS:
            swa.append(clone_state(model))

        if fixed_epochs is not None:
            best_state, best_epoch = clone_state(model), epochs - 1
            s = P.score_regions(model, bundle["x"], bundle["d"], bundle["val"],
                                device=device)
            best_r, best_blocks = s["official_r"], list(s["blocks"])
            print(f"    fixed budget {epochs} epochs, final weights | "
                f"val official r {best_r:.4f} (recorded, NOT used for selection)")

    # -- weight averaging, accepted only if validation agrees ------------
    if fixed_epochs is None and len(swa) >= 2:
        swa_model = build_model(bundle["n_el"], n_bands=bundle["n_bands"],
                                preset=preset, **tkw).to(device)
        swa_model.load_state_dict(average_states(swa))
        reestimate_bn(swa_model, bn_loader, device)
        s = P.score_regions(swa_model, bundle["x"], bundle["d"], bundle["val"],
                            device=device)
        print(f"    SWA over last {len(swa)} epochs: val r {s['official_r']:.4f} "
              f"vs best single {best_r:.4f}")
        if s["official_r"] > best_r:
            best_state, best_r = clone_state(swa_model), s["official_r"]
            best_blocks = list(s["blocks"])
            print("    -> SWA accepted")

    model.load_state_dict(best_state)
    model.to("cpu").eval()

    out = P.ckpt_path(script_dir, sub, seed, tag)
    torch.save({
        "state_dict": clone_state(model),
        "n_electrodes": bundle["n_el"], "n_bands": bundle["n_bands"],
        "preset": preset,
        "trunk_kw": tkw,                       # [FIX-7]
        "window": P.WINDOW, "delay": P.DELAY, "fs": P.FS,
        "seed": seed, "subject": sub, "val_official_r": best_r,
        "val_blocks": best_blocks,             # [FIX-6]
        "val_n_blocks": len(bundle["val"]),
        "best_epoch": best_epoch, "pretrained": init_from is not None,
        "params": rep["params"], "macs_per_sample": rep["macs_per_sample"],
        "receptive_field": rep["receptive_field"],
        "prep_config": bundle["d"].get("config", {}),
    }, out)
    print(f"  saved {os.path.basename(out)} | val official r {best_r:.4f} "
          f"| {time.time() - t0:.0f}s")
    return best_r, best_epoch


# ======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=P.DATA_DIR)
    p.add_argument("--ckpt-dir", default=P.CKPT_DIR,
                   help="where checkpoints are written")
    p.add_argument("--subjects", nargs="*", default=P.SUBJECTS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--preset", default=PRESET, choices=["nano", "tiny", "base"])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--fixed-epochs", type=int, default=None,
                help="train every run for exactly N epochs and keep the "
                    "final weights: no per-run selection, no SWA")
    p.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--no-gate", action="store_true",
                   help="ablation: drop the causal channel gate")
    p.add_argument("--no-smooth", action="store_true",
                   help="ablation: drop the learned causal output FIR")
    p.add_argument("--skip-last", action="store_true",
                   help="ablation: last-block features instead of skip sum")
    p.add_argument("--eca-gate", action="store_true",
                   help="ablation: model_v2's time-pooling ECA gate (leaks "
                        "look-ahead under dense supervision)")
    p.add_argument("--refit", action="store_true",
                   help="after selection, retrain on train+val for the chosen "
                        "epoch count and save as *_refit")
    p.add_argument("--tag", default="flexnet")
    a = p.parse_args()

    script_dir = a.ckpt_dir
    os.makedirs(script_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkw = trunk_overrides(a)
    print(f"data '{a.data_dir}' | device {device} | preset {a.preset} | "
          f"seeds {a.seeds} | pretrain {'off' if a.no_pretrain else 'on'}"
          f"{' | ' + json.dumps(tkw) if tkw else ''}")

    missing = [s for s in a.subjects if not os.path.exists(P.prep_path(s, a.data_dir))]
    if missing:
        raise SystemExit(f"missing {missing} in '{a.data_dir}' — run prep.py first")

    bundles = {s: subject_bundle(s, a.data_dir) for s in a.subjects}
    summary = {}

    for seed in a.seeds:
        shared = None
        if not a.no_pretrain and len(bundles) > 1:
            shared = pretrain(bundles, seed, device, a.preset,
                              a.pretrain_epochs, tkw=tkw)
        for sub in a.subjects:
            r, best_ep = train_subject(
                sub, bundles[sub], seed, device, a.preset, script_dir,
                init_from=shared, epochs=a.fixed_epochs or a.epochs,
                lr=FINETUNE_LR if shared is not None else LR, tag=a.tag,
                tkw=tkw, fixed_epochs=a.fixed_epochs)
            summary.setdefault(sub, []).append(r)

            if a.refit:
                rb = subject_bundle(sub, a.data_dir, include_val_in_train=True)
                train_subject(sub, rb, seed, device, a.preset, script_dir,
                              init_from=shared, epochs=best_ep + 1,
                              lr=FINETUNE_LR if shared is not None else LR,
                              tag=a.tag + "_refit", fixed_epochs=best_ep + 1,
                              tkw=tkw)

    print(f"\n{'=' * 72}\nVALIDATION official r (selection metric, NOT the "
          f"published number)\n{'=' * 72}")
    for sub, v in summary.items():
        m, s = P.mean_std(v)
        print(f"  {sub:<8} {m:.4f} ± {s:.4f}   " + "  ".join(f"{x:.4f}" for x in v))
    print("\nNow run: python decoder/score.py --tag " + a.tag)


if __name__ == "__main__":
    main()
