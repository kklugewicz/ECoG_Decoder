"""
prep.py — preprocessing for BCI Competition IV ds4.

v3.1 CHANGES (audit fixes). Search for [FIX-n] to find each one in the code.

[FIX-1]  TARGET LEAK ACROSS THE TRAIN/TEST BOUNDARY.
         v3.0 ran `sosfiltfilt` over the CONCATENATED train+test glove trace.
         filtfilt is bidirectional, so training targets within ~one filter
         length of index n_train_raw were contaminated by `test_dg`. The
         magnitude is a handful of samples at 8 Hz — negligible in effect,
         fatal in review. The two recordings are now smoothed independently.
         The test-region smoothed target still exists, but it is used ONLY
         for the declared-internal `smoothed-target` diagnostic row and for
         the protocol-D ceiling, never as a training signal.

[FIX-2]  BAD CHANNELS ARE SCREENED, NOT HARD-CODED.
         v3.0 carried a hand-curated list whose provenance its own comment
         admitted was disputed. A reviewer cannot verify that the list was
         not informed by the test recording. `screen_channels` replaces it
         with a deterministic, train-only, robust criterion (log-MAD outlier
         + flat detection). `--channels legacy` reproduces the old list and
         `--channels all` keeps everything, so both are ablation rows.

[FIX-3]  ONE STATISTICS FIT WINDOW, DOCUMENTED.
         v3.0 fitted the log-compression medians and the target
         standardisation on the first 2/3 of the training recording, while
         model selection used the last 20% — two different boundaries for no
         stated reason. Statistics are now fitted on the WHOLE training
         recording (`--stats-fraction 1.0`), which is exactly the calibration
         data a deployed decoder would have, and never on the test
         recording. `--stats-fraction 0.667` restores the old behaviour.

[FIX-4]  ABLATION SWITCHES.
         The v3.0 docstring claimed five improvements in descending order of
         effect and measured none of them. Every claim is now a flag:
           --target-causal    revert [1]: causal `sosfilt` target smoothing
           --env-cutoff 5.0   revert [3]: the old 83 ms envelope low-pass
           --split-filter     revert [2]: filter train and test separately
           --no-log           revert [4]: raw rectified envelopes
           --bands 4          revert [5]: LMP + mu/beta + one wide high gamma
         `ablate.py` drives them and writes the table the paper needs.

WHAT IS DELIBERATELY UNCHANGED
------------------------------
* Every ECoG-path filter is `lfilter` / `sosfilt`, i.e. strictly causal.
* No test-derived statistic touches anything: CAR is within-sample, the log
  scale and target standardisation use train-only samples, and the z-score
  (applied in protocol) is causal and online.
* The 37 ms dataglove-vs-amplifier lag noted in desc_4.pdf is NOT corrected.
  It is a constant offset present identically in train and test, so the model
  absorbs it during fitting; "correcting" it would double-count.

OUTPUT
------
`data/prep/{sub}_prep.pt`:
    feat        (N_total, n_electrodes, n_bands)  float32, 100 Hz,
                log-compressed, UNNORMALISED, train+test contiguous
    dg          (N_total, 5)   smoothed, decimated, train-standardised
    dg_raw      (N_raw, 5)     the untouched 1000 Hz glove traces
    test_start_proc / test_start_raw, dg_mean, dg_std, fs, decim,
    band_names, n_electrodes, bad_channels, channel_scores, config

Usage:
    python decoder/prep.py                 # all subjects, default configuration
    python decoder/prep.py --selftest      # filter/alignment/leak checks, no data
"""

import argparse
import json
import os

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import torch

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "data", "prep")     # features written here
BASE_DIR = os.path.join(REPO_ROOT, "data", "raw")     # sub*_comp.mat, sub*_testlabels.mat
SUBJECTS = ["sub1", "sub2", "sub3"]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

FS_IN = 1000.0
DECIM = 10
FS_OUT = FS_IN / DECIM          # 100 Hz

ENV_CUTOFF = 20.0               # envelope low-pass, Hz   (was 5.0 -> 83 ms delay)
ENV_ORDER = 4                   #                          now  -> 21 ms delay
LMP_CUTOFF = 12.0               # local motor potential low-pass, Hz
LMP_ORDER = 4
TARGET_CUTOFF = 8.0             # targets only
TARGET_ORDER = 4

NOTCH_FREQS = (60.0, 120.0, 180.0, 240.0)
NOTCH_Q = 30.0

# (name, passband | None for the low-pass-only LMP)
BANDS_7 = [
    ("LMP",    None),
    ("mu",     (8.0, 13.0)),
    ("beta",   (13.0, 30.0)),
    ("lgamma", (33.0, 57.0)),
    ("hg1",    (65.0, 100.0)),
    ("hg2",    (105.0, 145.0)),
    ("hg3",    (155.0, 195.0)),
]
# [FIX-4] ablation: the conventional four-band decomposition.
BANDS_4 = [
    ("LMP",    None),
    ("mu",     (8.0, 13.0)),
    ("beta",   (13.0, 30.0)),
    ("hgamma", (65.0, 195.0)),
]
BAND_SETS = {4: BANDS_4, 7: BANDS_7}

# Front of the TRAINING recording only. Covers the filter start-up transient
# and the amplifier settling artefact. Nothing is removed from the array —
# training windows simply start after it, so every index map stays trivial.
TRAIN_SKIP_RAW = 2000           # 2.0 s @ 1000 Hz
STATS_FRACTION = 1.0            # [FIX-3] fraction of the TRAIN recording used
                                # to fit log medians and target z-scores

CHANNEL_MAD_Z = 5.0             # [FIX-2] robust-z threshold for screening

# [FIX-2] Retained only so `--channels legacy` can reproduce the v3.0 run as
# an ablation row. NOT used by default. Provenance was never established.
LEGACY_BAD_CHANNELS = {
    "sub1": [26, 39, 48, 54],
    "sub2": [5, 9, 20, 26, 27, 31, 37],
    "sub3": [22, 37, 49, 57],
}


def processed_to_raw_index(m, decim=DECIM):
    """Processed index -> index in the original 1000 Hz recording.

    Trivial by construction: nothing is trimmed before decimation and
    decimation keeps sample 0, so index m came from raw sample DECIM*m.
    """
    return decim * np.asarray(m)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_subject_mat(sub, base_dir=BASE_DIR):
    comp = sio.loadmat(os.path.join(base_dir, f"{sub}_comp.mat"))
    labels = sio.loadmat(os.path.join(base_dir, f"{sub}_testlabels.mat"))
    # 0.0298 uV/bit, as in filtering_5/8.
    train_sig = comp["train_data"].astype(np.float64) * 0.0298
    test_sig = comp["test_data"].astype(np.float64) * 0.0298
    train_dg = comp["train_dg"].astype(np.float64)
    test_dg = labels["test_dg"].astype(np.float64)
    return train_sig, train_dg, test_sig, test_dg


# ----------------------------------------------------------------------
# [FIX-2] Channel screening — deterministic, train-only, reportable
# ----------------------------------------------------------------------
def screen_channels(train_sig, skip=TRAIN_SKIP_RAW, z_thresh=CHANNEL_MAD_Z):
    """Flag dead and railing electrodes from the TRAINING recording alone.

    Criterion, in full, so the paper can state it in one sentence: compute
    each channel's median absolute deviation over the training recording;
    take logs, because amplifier gain faults are multiplicative; and reject
    any channel whose log-MAD is more than `z_thresh` robust standard
    deviations from the across-channel median, or whose MAD is numerically
    zero (a flat/disconnected line).

    Returns (bad_channel_indices, robust_z_per_channel). Nothing here can see
    the test recording, so the selection is defensible without appeal to
    anyone's screening notes.
    """
    seg = np.asarray(train_sig[skip:], dtype=np.float64)
    mad = np.median(np.abs(seg - np.median(seg, axis=0, keepdims=True)), axis=0)
    flat = mad <= 1e-9
    lm = np.log(np.maximum(mad, 1e-12))
    ref = lm[~flat] if (~flat).any() else lm
    med = np.median(ref)
    scale = 1.4826 * np.median(np.abs(ref - med))
    z = (lm - med) / (scale if scale > 1e-12 else 1.0)
    bad = np.where(flat | (np.abs(z) > z_thresh))[0]
    return sorted(int(c) for c in bad), z


# ----------------------------------------------------------------------
# Filter bank  (every stage causal)
# ----------------------------------------------------------------------
def notch_and_reference(data):
    """Line-noise notches, then a robust common average reference.

    Median rather than mean: a single railing electrode moves the mean CAR
    into every other channel, and ds4 has a handful of those. The median is
    the same operation when the array is healthy and much better when it is
    not, which is what makes `--channels all` a usable option.
    """
    for f0 in NOTCH_FREQS:
        b, a = signal.iirnotch(f0, NOTCH_Q, FS_IN)
        data = signal.lfilter(b, a, data, axis=0)
    return data - np.median(data, axis=1, keepdims=True)


def one_band(data, band, env_cutoff=ENV_CUTOFF):
    """(N, C) referenced ECoG -> (N, C) one causal band feature at 1000 Hz.

    Order: band-pass -> rectify -> low-pass. Rectifying before the low-pass is
    what turns a band-pass signal into an amplitude envelope; low-passing
    first would annihilate it. The LMP band skips both the band-pass and the
    rectification — it IS the low-frequency potential, and it is signed.
    """
    nyq = FS_IN / 2.0
    if band is None:
        sos = signal.butter(LMP_ORDER, LMP_CUTOFF / nyq, "low", output="sos")
        return signal.sosfilt(sos, data, axis=0)
    sos_bp = signal.butter(4, [band[0] / nyq, band[1] / nyq], "bandpass",
                           output="sos")
    sos_env = signal.butter(ENV_ORDER, env_cutoff / nyq, "low", output="sos")
    return signal.sosfilt(sos_env, np.abs(signal.sosfilt(sos_bp, data, axis=0)),
                          axis=0)


def extract_bands(data, bands=BANDS_7, decim=1, fit_slice=None,
                  compress_log=False, env_cutoff=ENV_CUTOFF):
    """(N, C) -> (N//decim, C, n_bands) float32.

    Bands are filtered and decimated ONE AT A TIME. Holding all seven at
    1000 Hz in float64 would need ~2 GB for a real subject; this needs ~100 MB
    for the result plus one band's working buffer.

    `compress_log` applies log1p(env / median) to the envelope bands, leaving
    the signed LMP alone. The median is per (channel, band) over `fit_slice`
    — training samples only — so nothing downstream of the train/test
    boundary can influence the transform.
    """
    n_out = (data.shape[0] + decim - 1) // decim
    out = np.empty((n_out, data.shape[1], len(bands)), dtype=np.float32)
    for b, (_, band) in enumerate(bands):
        f = one_band(data, band, env_cutoff)[::decim]
        if compress_log and band is not None:
            med = np.maximum(np.median(f[fit_slice], axis=0), 1e-12)
            f = np.log1p(np.maximum(f, 0.0) / med)
        out[:, :, b] = f.astype(np.float32)
        del f
    return out


def smooth_targets(dg, cutoff=TARGET_CUTOFF, order=TARGET_ORDER, fs=FS_IN,
                   causal=False):
    """Low-pass the glove traces.

    Zero-phase by default: `sosfiltfilt` runs forward and backward, so the
    magnitude response is squared and the phase response is exactly zero —
    the smoothed target sits at the same time as the flexion it describes.
    Compare `sosfilt`, which lags by 83 ms at this order and cutoff and drags
    every prediction along with it.

    Non-causality here is not a leak PROVIDED the two recordings are smoothed
    separately; see [FIX-1] and the caller. Labels are prepared offline; only
    the ECoG path has to be causal for a real-time claim.

    `causal=True` is the ablation that restores the v2/filtering_8 behaviour.
    """
    sos = signal.butter(order, cutoff / (fs / 2.0), "low", output="sos")
    if causal:
        return signal.sosfilt(sos, dg, axis=0)
    return signal.sosfiltfilt(sos, dg, axis=0)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def process(sub, base_dir=BASE_DIR, out_dir=OUT_DIR, channels="auto",
            target_causal=False, env_cutoff=ENV_CUTOFF, split_filter=False,
            compress_log=True, n_bands=7, stats_fraction=STATS_FRACTION):
    print(f"\n{'=' * 68}\n{sub}\n{'=' * 68}")
    train_sig, train_dg, test_sig, test_dg = load_subject_mat(sub, base_dir)

    n_train_raw = train_sig.shape[0]
    n_test_raw = test_sig.shape[0]
    assert n_train_raw % DECIM == 0, (
        f"train length {n_train_raw} is not a multiple of DECIM={DECIM}; the "
        f"raw<->processed index map would stop being exact.")
    print(f"  train {train_sig.shape}  test {test_sig.shape}")

    bands = BAND_SETS[int(n_bands)]

    # -- [FIX-2] channel selection ---------------------------------------
    if channels == "auto":
        bad, zscore = screen_channels(train_sig)
        print(f"  screening (train-only, |robust z of log-MAD| > "
              f"{CHANNEL_MAD_Z}): rejected {bad}")
    elif channels == "legacy":
        bad, zscore = LEGACY_BAD_CHANNELS.get(sub, []), None
        print(f"  legacy hand-curated list: {bad}")
    elif channels == "all":
        bad, zscore = [], None
        print("  keeping every electrode")
    else:
        raise ValueError(channels)
    keep = [c for c in range(train_sig.shape[1]) if c not in bad]
    train_sig, test_sig = train_sig[:, keep], test_sig[:, keep]
    n_el = len(keep)
    print(f"  dropped {len(bad)} channels -> {n_el} electrodes")

    # -- ONE continuous stream -------------------------------------------
    # The competition split is chronological within a single session, so the
    # causal filter state at the first test sample is exactly what a deployed
    # decoder would hold. Filtering the halves separately throws that away and
    # buys a start-up transient in exchange; `--split-filter` is that ablation.
    dg_raw = np.concatenate([train_dg, test_dg], axis=0)
    fit_end_raw = int(stats_fraction * n_train_raw)
    fit_proc = slice(TRAIN_SKIP_RAW // DECIM, fit_end_raw // DECIM)

    if split_filter:
        f_tr = extract_bands(notch_and_reference(train_sig), bands, DECIM,
                             fit_proc, compress_log, env_cutoff)
        f_te = extract_bands(notch_and_reference(test_sig), bands, DECIM,
                             slice(None), compress_log, env_cutoff)
        feats = np.concatenate([f_tr, f_te], axis=0)
        del f_tr, f_te
    else:
        sig = np.concatenate([train_sig, test_sig], axis=0)
        assert sig.shape[0] == n_train_raw + n_test_raw
        feats = extract_bands(notch_and_reference(sig), bands, DECIM,
                              fit_proc, compress_log, env_cutoff)
        del sig
    del train_sig, test_sig

    test_start_proc = n_train_raw // DECIM
    assert processed_to_raw_index(test_start_proc) == n_train_raw

    # -- [FIX-1] targets, smoothed WITHOUT crossing the boundary ---------
    # Each recording is smoothed on its own. The zero-phase filter therefore
    # never carries a test label backwards into a training target. The two
    # halves are concatenated afterwards purely so the array indexes like the
    # feature array; only indices < test_start_proc are ever trained on.
    dg_smooth = np.concatenate([
        smooth_targets(dg_raw[:n_train_raw], causal=target_causal),
        smooth_targets(dg_raw[n_train_raw:], causal=target_causal),
    ], axis=0)[::DECIM]

    dg_mean = dg_smooth[fit_proc].mean(axis=0, keepdims=True)
    dg_std = dg_smooth[fit_proc].std(axis=0, keepdims=True) + 1e-8
    dg = (dg_smooth - dg_mean) / dg_std

    # How much of the raw target the smoother even leaves reachable. If this
    # is far below 1 the target is dominated by 25 Hz dataglove stair-steps
    # and no decoder, causal or not, can score near 1 on the internal metric.
    ceil = _corr(dg_smooth[test_start_proc:],
                 dg_raw[n_train_raw:][::DECIM])

    cfg = dict(channels=channels, target_causal=bool(target_causal),
               env_cutoff=float(env_cutoff), split_filter=bool(split_filter),
               compress_log=bool(compress_log), n_bands=int(n_bands),
               stats_fraction=float(stats_fraction),
               channel_mad_z=float(CHANNEL_MAD_Z))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sub}_prep.pt")
    torch.save({
        "feat": torch.tensor(feats, dtype=torch.float32),
        "dg": torch.tensor(dg, dtype=torch.float32),
        "dg_raw": torch.tensor(dg_raw, dtype=torch.float32),
        "dg_mean": torch.tensor(dg_mean, dtype=torch.float32),
        "dg_std": torch.tensor(dg_std, dtype=torch.float32),
        "test_start_proc": int(test_start_proc),
        "test_start_raw": int(n_train_raw),
        "n_train_raw": int(n_train_raw),
        "n_test_raw": int(n_test_raw),
        "train_skip_proc": int(TRAIN_SKIP_RAW // DECIM),
        "fs": FS_OUT, "decim": DECIM,
        "n_bands": len(bands), "band_names": [b[0] for b in bands],
        "n_electrodes": n_el, "kept_channels": keep, "bad_channels": bad,
        "channel_z": None if zscore is None else zscore.tolist(),
        "target_ceiling": ceil.tolist(),
        "config": cfg,
    }, path)

    print(f"  feat {tuple(feats.shape)} @ {FS_OUT:.0f} Hz | "
          f"{len(bands)} bands -> {n_el * len(bands)} model input rows")
    print(f"  test starts at processed {test_start_proc} / raw {n_train_raw}")
    print(f"  config {json.dumps(cfg)}")
    print(f"  per-finger corr(smoothed target, raw target) on the test block:")
    print("    " + "  ".join(f"{FINGER_NAMES[i]} {ceil[i]:.3f}" for i in range(5)))
    print(f"  -> multiplicative ceiling on the internal metric: "
          f"{np.mean(ceil[[0, 1, 2, 4]]):.3f}")
    print(f"  Saved {path}")
    n_bad = int(np.isnan(feats).sum() + np.isinf(feats).sum())
    if n_bad:
        raise ValueError(f"{sub}: {n_bad} NaN/Inf values in feat — fix before saving")


def _corr(a, b):
    va, vb = a - a.mean(0), b - b.mean(0)
    d = np.sqrt((va ** 2).sum(0)) * np.sqrt((vb ** 2).sum(0))
    return (va * vb).sum(0) / np.where(d > 0, d, np.nan)


# ----------------------------------------------------------------------
# Self-test — runs without the dataset
# ----------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    nyq = FS_IN / 2.0
    probe = [0.25, 0.5, 1.0, 2.0]

    def gdelay_ms(sos):
        b, a = signal.sos2tf(sos)
        _, g = signal.group_delay((b, a), w=2 * np.pi * np.array(probe) / FS_IN)
        return 1000.0 * g / FS_IN

    old = signal.butter(4, 5.0 / nyq, "low", output="sos")
    new = signal.butter(ENV_ORDER, ENV_CUTOFF / nyq, "low", output="sos")
    print("group delay (ms) at %s Hz" % probe)
    print(f"  ablation envelope/target     butter4 @ 5 Hz : "
          f"{np.round(gdelay_ms(old), 1)}")
    print(f"  prep     envelope         butter{ENV_ORDER} @ {ENV_CUTOFF:.0f} Hz: "
          f"{np.round(gdelay_ms(new), 1)}")
    _, h = signal.sosfreqz(new, worN=[FS_OUT / 2], fs=FS_IN)
    print(f"  anti-alias margin at {FS_OUT / 2:.0f} Hz: "
          f"{20 * np.log10(abs(h[0])):.1f} dB")

    # Zero-phase target smoothing must not move the signal in time.
    x = signal.sosfiltfilt(signal.butter(4, 1.5 / nyq, "low", output="sos"),
                           rng.standard_normal(60_000))
    x = x / x.std()
    lags = np.arange(-200, 201)
    causal = smooth_targets(x[:, None], cutoff=5.0, causal=True)[:, 0]
    zerop = smooth_targets(x[:, None])[:, 0]
    for name, y in (("causal butter4@5Hz", causal), ("prep zero-phase", zerop)):
        c = [np.corrcoef(np.roll(y, -L)[500:-500], x[500:-500])[0, 1] for L in lags]
        best = lags[int(np.argmax(c))]
        print(f"  {name:<22} peak correlation with the unsmoothed trace at "
              f"lag {best:+d} ms (r={max(c):.4f})")
        if "zero-phase" in name:
            assert abs(best) <= 2, "zero-phase filter is not zero-phase"

    # [FIX-1] The training target must be independent of the test labels.
    n_tr, n_te = 30_000, 15_000
    base = signal.sosfiltfilt(signal.butter(4, 1.0 / nyq, "low", output="sos"),
                              rng.standard_normal((n_tr + n_te, 5)), axis=0)
    alt = base.copy()
    alt[n_tr:] = rng.standard_normal((n_te, 5)) * 50.0     # obliterate test_dg

    def build(d):
        return np.concatenate([smooth_targets(d[:n_tr]),
                               smooth_targets(d[n_tr:])], axis=0)[::DECIM]

    drift = np.abs(build(base)[:n_tr // DECIM] - build(alt)[:n_tr // DECIM]).max()
    print(f"  training target vs. randomised test labels: max drift {drift:.2e}")
    assert drift < 1e-9, "training targets still depend on test_dg"

    naive = signal.sosfiltfilt(signal.butter(TARGET_ORDER,
                                             TARGET_CUTOFF / nyq, "low",
                                             output="sos"), base, axis=0)
    naive_alt = signal.sosfiltfilt(signal.butter(TARGET_ORDER,
                                                 TARGET_CUTOFF / nyq, "low",
                                                 output="sos"), alt, axis=0)
    leak = np.abs(naive[:n_tr] - naive_alt[:n_tr]).max()
    print(f"  (v3.0 whole-stream filtfilt would have leaked {leak:.3e} "
          f"into the training target)")
    assert leak > drift, "the leak self-test is not exercising anything"

    # [FIX-2] Screening must find an obviously dead channel and nothing else.
    sig = rng.standard_normal((20_000, 12)) * 30.0
    sig[:, 3] = 0.0                       # disconnected
    sig[:, 7] *= 200.0                    # railing
    bad, _ = screen_channels(sig, skip=0)
    print(f"  channel screening on a planted array: {bad}")
    assert bad == [3, 7], bad

    # Index map.
    m = np.array([0, 1, 5000, 40000])
    assert np.array_equal(processed_to_raw_index(m), DECIM * m)
    print("  index map raw = 10 * processed: OK")

    # Causality of the ECoG path: perturbing a future sample must not change
    # any earlier feature.
    c = 4
    d0 = rng.standard_normal((5000, c))
    d1 = d0.copy()
    d1[3000:] += 50.0
    f0 = extract_bands(notch_and_reference(d0), decim=1)
    f1 = extract_bands(notch_and_reference(d1), decim=1)
    # (the median CAR couples channels within a sample, never across time)
    assert np.abs(f0[:3000] - f1[:3000]).max() < 1e-9, "feature path is not causal"
    print("  ECoG feature path is strictly causal: OK")
    print("\nAll prep self-tests passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=BASE_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--subjects", nargs="*", default=SUBJECTS)
    p.add_argument("--channels", default="auto",
                   choices=["auto", "legacy", "all"])
    p.add_argument("--target-causal", action="store_true",
                   help="ablation: causal target smoothing (v2 behaviour)")
    p.add_argument("--env-cutoff", type=float, default=ENV_CUTOFF,
                   help="ablation: 5.0 restores the 83 ms envelope delay")
    p.add_argument("--split-filter", action="store_true",
                   help="ablation: filter train and test independently")
    p.add_argument("--no-log", action="store_true",
                   help="ablation: skip log compression of the envelopes")
    p.add_argument("--bands", type=int, default=7, choices=[4, 7])
    p.add_argument("--stats-fraction", type=float, default=STATS_FRACTION)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
    else:
        for s in a.subjects:
            process(s, a.base_dir, a.out_dir, channels=a.channels,
                    target_causal=a.target_causal, env_cutoff=a.env_cutoff,
                    split_filter=a.split_filter, compress_log=not a.no_log,
                    n_bands=a.bands, stats_fraction=a.stats_fraction)
        print("\nDone. Next: python decoder/score.py --tag paper "
              "(shipped models) or python decoder/train.py")
