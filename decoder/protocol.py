"""
protocol.py — the single definition of "the score", shared by training,
evaluation and export so they cannot drift apart.

THE METRIC, VERBATIM FROM desc_4.pdf SECTION V
----------------------------------------------
    "the received variable eval_dg will be compared with the actual finger
     positions in test_dg ... We will calculate the correlation coefficient r
     between the actual and the predicted finger flexions for each subject
     and finger. We will not calculate the correlation coefficient for the
     4th (ring) finger ... The final score will be calculated as the
     arithmetic mean of the 12 correlation coefficients."

So: `eval_dg` is 200,000 x 5 at 1000 Hz, the comparison is against the stored
`test_dg` with nothing done to it, and the average is over 4 fingers x 3
subjects. `official_score` below implements exactly that and is the ONLY
scoring function in this package. Validation during training calls it too, so
model selection optimises the published number rather than a proxy.

v3.1 CHANGE — [FIX-5] INTERLEAVED-BLOCK VALIDATION
--------------------------------------------------
v3.0 selected on the last 20% of the training recording: a single contiguous
block of ~80 s, evaluated 33 times per run, and additionally used to accept or
reject SWA. On ds4 that is far too little data for that much selection
pressure, and it showed — across seeds the validation ranking was uncorrelated
with, and for one subject inverted relative to, the test ranking.

`split_bounds` now returns LISTS of intervals. The same 20% is held out, but
as `VAL_BLOCKS` evenly spaced blocks rather than one tail, and validation
scores each block with the competition metric and averages. Same training
cost, much lower selection variance, and the estimate is no longer a claim
about one arbitrary 80 s window.

Training windows are generated strictly inside the training intervals, so no
validation TARGET is ever differentiated. A training window's causal input
history may overlap an earlier validation block — unavoidable when features
are one continuous stream, harmless because the labels there are unused, and
stated plainly in the paper.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Default locations, relative to the repository root, so the scripts work from
# any working directory. Every script also takes --data-dir (and the ones that
# touch models take --ckpt-dir) to point somewhere else.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
DATA_DIR = os.path.join(REPO_ROOT, "data", "prep")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")
SUBJECTS = ["sub1", "sub2", "sub3"]

FS = 100.0
DECIM = 10
WINDOW = 600             # 6.0 s training window
STEP = 100               # 1.0 s between training windows
DELAY = 0                # emit y[t] at input t. >0 = forecast, costs r.

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
SCORED_FINGER_IDX = [0, 1, 2, 4]     # ring (index 3) excluded, per desc_4.pdf
EXCLUDED_FINGER_IDX = 3

VAL_FRACTION = 0.20      # of the training recording
VAL_BLOCKS = 4           # [FIX-5] spread over the recording, not one tail
EPS = 1e-6


def prep_path(sub, data_dir=DATA_DIR):
    return os.path.join(data_dir, f"{sub}_prep.pt")


def ckpt_path(ckpt_dir, sub, seed, tag="flexnet"):
    return os.path.join(ckpt_dir, f"{tag}_{sub}_s{seed}.pt")


def pte_path(ckpt_dir, sub, seed, tag="flexnet"):
    return os.path.join(ckpt_dir, f"{tag}_{sub}_s{seed}_int8.pte")


# ----------------------------------------------------------------------
# Online normalisation
# ----------------------------------------------------------------------
class StreamingZScore:
    """Causal running z-score. Output at sample k uses only samples 1..k.

    Vectorised cumulative-sum form of the Welford recursion, verified equal
    to 1e-10 against the scalar recursion.
    """

    def __init__(self, eps=EPS):
        self.eps, self.k0, self.sum0, self.sumsq0 = eps, 0, None, None

    def __call__(self, data):
        x = np.asarray(data, dtype=np.float64) if not torch.is_tensor(data) \
            else data.detach().cpu().numpy().astype(np.float64)
        flat = x.reshape(x.shape[0], -1)
        if self.sum0 is None:
            self.sum0 = np.zeros(flat.shape[1])
            self.sumsq0 = np.zeros(flat.shape[1])

        csum = self.sum0 + np.cumsum(flat, axis=0)
        csumsq = self.sumsq0 + np.cumsum(flat ** 2, axis=0)
        k = (self.k0 + np.arange(1, flat.shape[0] + 1)).astype(np.float64)[:, None]

        mean = csum / k
        m2 = np.maximum(csumsq - k * mean ** 2, 0.0)
        std = np.sqrt(np.where(k > 1, m2 / np.maximum(k - 1, 1.0), 0.0))
        out = (flat - mean) / (std + self.eps)
        out[k[:, 0] <= 1] = 0.0

        self.k0 = int(k[-1, 0])
        self.sum0, self.sumsq0 = csum[-1].copy(), csumsq[-1].copy()
        return torch.tensor(out.reshape(x.shape), dtype=torch.float32)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_prep(sub, data_dir=DATA_DIR):
    d = torch.load(prep_path(sub, data_dir), weights_only=False)
    for k in ("feat", "dg", "dg_raw", "test_start_proc"):
        if k not in d:
            raise ValueError(f"{prep_path(sub, data_dir)} is missing '{k}'. "
                             f"Re-run prep.py.")
    return d


def normalised_features(d):
    """(rows, N) model-ready input for the ENTIRE recording, z-scored online.

    One pass, train straight into test: no test sample ever influences the
    normalisation of an earlier sample, and the statistics arriving at the
    first test sample are the ones a deployed decoder would hold.
    """
    z = StreamingZScore()(d["feat"])                     # (N, C, B)
    return z.permute(2, 1, 0).reshape(-1, z.shape[0])    # (B*C, N), band-major


class SegmentDataset(Dataset):
    """Contiguous segments with a target at every timestep.

    `intervals` is a list of [lo, hi) processed-index ranges. Windows never
    straddle an interval boundary, so a validation block's targets cannot
    enter the loss.

    x : (n_bands*n_electrodes, window)
    y : (window, 5)      y[t] is the target for model output timestep t
    """

    def __init__(self, x, dg, intervals, window=WINDOW, step=STEP, delay=DELAY,
                 augment=False, noise=0.05, chan_drop=0.1, gain_jitter=0.1,
                 n_electrodes=None, n_bands=None):
        self.x, self.dg = x, dg
        self.window, self.delay = window, delay
        self.augment = augment
        self.noise, self.chan_drop, self.gain_jitter = noise, chan_drop, gain_jitter
        self.n_electrodes, self.n_bands = n_electrodes, n_bands
        if intervals and np.isscalar(intervals[0]):      # (lo, hi) tolerated
            intervals = [tuple(intervals)]
        self.starts = []
        for lo, hi in intervals:
            hi = min(hi, dg.shape[0] - delay)
            if hi - lo < window:
                continue
            self.starts.extend(range(lo, hi - window + 1, step))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        x = self.x[:, s:s + self.window]
        y = self.dg[s + self.delay:s + self.window + self.delay]
        if self.augment:
            x = self._aug(x)
        return x, y

    def _aug(self, x):
        """Failure modes that actually occur in a chronic ECoG recording:
        an electrode dies (drops across all of its bands at once), impedance
        drifts (per-electrode gain), and everything else (additive noise)."""
        c, b = self.n_electrodes, self.n_bands
        x = x.clone()
        if self.chan_drop > 0:
            keep = (torch.rand(c) > self.chan_drop).float()
            x = x * keep.repeat(b).unsqueeze(1)
        if self.gain_jitter > 0:
            x = x * (1.0 + self.gain_jitter * torch.randn(c)).repeat(b).unsqueeze(1)
        if self.noise > 0:
            x = x + torch.randn_like(x) * self.noise
        return x


def split_bounds(d, val_fraction=VAL_FRACTION, n_blocks=VAL_BLOCKS):
    """[FIX-5] Interleaved-block split of the TRAINING recording.

    Returns (train_intervals, val_intervals, test_interval).

    The recording is cut into `n_blocks` equal chunks; the LAST
    `val_fraction` of each chunk is held out. So the validation set is the
    same size as v3.0's single tail but samples four different parts of the
    session, which is what kills the selection variance.
    """
    test_start = int(d["test_start_proc"])
    lo = int(d.get("train_skip_proc", 200))
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    edges = np.linspace(lo, test_start, n_blocks + 1).astype(int)

    train, val = [], []
    for i in range(n_blocks):
        a, b = int(edges[i]), int(edges[i + 1])
        cut = int(b - val_fraction * (b - a))
        if cut > a:
            train.append((a, cut))
        if b > cut:
            val.append((cut, b))
    return train, val, (test_start, int(d["feat"].shape[0]))


def interval_len(intervals):
    return int(sum(hi - lo for lo, hi in intervals))


# ----------------------------------------------------------------------
# THE metric
# ----------------------------------------------------------------------
def upsample(pred_proc, proc_lo, n_raw, decim=DECIM, mode="zoh"):
    """100 Hz predictions -> a 1000 Hz trace, the shape the competition wants.

    zoh    hold each value for `decim` samples. Strictly causal, and what a
           device emitting at 100 Hz actually puts on the wire.
    linear interpolate toward the next value, i.e. read 10 ms ahead. Legal for
           a competition submission, not for a latency claim.
    """
    pred = np.asarray(pred_proc, dtype=np.float64)
    raw_idx = decim * (proc_lo + np.arange(pred.shape[0]))
    grid = np.arange(n_raw)
    out = np.empty((n_raw, pred.shape[1]))
    if mode == "zoh":
        pos = np.searchsorted(raw_idx, grid, side="right") - 1
        pos = np.clip(pos, 0, len(raw_idx) - 1)
        out = pred[pos]
        out[grid < raw_idx[0]] = np.nan
    else:
        for f in range(pred.shape[1]):
            out[:, f] = np.interp(grid, raw_idx, pred[:, f],
                                  left=np.nan, right=pred[-1, f])
    return out


def official_score(pred_proc, dg_raw, proc_lo, raw_lo, raw_hi,
                   decim=DECIM, mode="zoh", dg_mean=None, dg_std=None):
    """The competition criterion.

    pred_proc : (M, 5) predictions at 100 Hz, standardised units, the first of
                which is attributed to processed index `proc_lo`
    dg_raw    : (N_raw, 5) the untouched 1000 Hz glove traces
    raw_lo/hi : the scored region. For the real submission this is the entire
                test recording, [test_start_raw, N_raw).

    Correlation is affine-invariant, so predictions are scored in whatever
    units they arrive in. RMSE is not, so it is reported in glove units when
    dg_mean/dg_std are supplied.
    """
    dg_raw = np.asarray(dg_raw, dtype=np.float64)
    trace = upsample(pred_proc, proc_lo, dg_raw.shape[0], decim, mode)

    sl = slice(raw_lo, raw_hi)
    p, t = trace[sl], dg_raw[sl]
    ok = ~np.isnan(p[:, 0])
    coverage = float(ok.mean())
    if coverage < 1.0:
        # A deployed decoder emits its prior before its first prediction; in
        # standardised units that is 0. Anything data-dependent would leak.
        p = p.copy()
        p[~ok] = 0.0

    vp, vt = p - p.mean(0), t - t.mean(0)
    den = np.sqrt((vp ** 2).sum(0)) * np.sqrt((vt ** 2).sum(0))
    r = (vp * vt).sum(0) / np.where(den > 0, den, np.nan)

    if dg_mean is not None:
        pg = p * np.asarray(dg_std).reshape(1, -1) + np.asarray(dg_mean).reshape(1, -1)
        rmse = np.sqrt(((pg - t) ** 2).mean(0))
        units = "glove"
    else:
        rmse = np.sqrt(((p - t) ** 2).mean(0))
        units = "std"

    return {"r": r, "rmse": rmse, "units": units, "coverage": coverage,
            "official_r": float(np.nanmean(r[SCORED_FINGER_IDX])),
            "official_rmse": float(np.nanmean(rmse[SCORED_FINGER_IDX])),
            "n_scored": int(raw_hi - raw_lo)}


@torch.no_grad()
def predict_range(model, x, lo, hi, chunk=20_000, device=torch.device("cpu")):
    """Predictions for processed indices [lo, hi), each with a full receptive
    field of real history. Identical to running the model over the entire
    recording and slicing, but only pays for what is needed."""
    from model import predict_dense
    start = max(0, lo - (model.receptive_field - 1))
    y = predict_dense(model, x[:, start:hi], chunk=chunk, device=device)
    return y[lo - start:]


@torch.no_grad()
def score_region(model, x, d, region, mode="zoh", chunk=20_000,
                 device=torch.device("cpu")):
    """Score one processed region with the competition metric."""
    lo, hi = region
    pred = predict_range(model, x, lo, hi, chunk=chunk, device=device).numpy()
    return official_score(pred, d["dg_raw"], lo,
                          DECIM * lo, min(DECIM * hi, d["dg_raw"].shape[0]),
                          mode=mode,
                          dg_mean=d["dg_mean"].numpy().ravel(),
                          dg_std=d["dg_std"].numpy().ravel())


@torch.no_grad()
def score_regions(model, x, d, regions, mode="zoh", chunk=20_000,
                  device=torch.device("cpu")):
    """[FIX-5] Score several held-out blocks and average the competition
    metric over them. Each block is scored independently — concatenating
    disjoint blocks before correlating would let a between-block mean
    difference masquerade as predictive power."""
    if regions and np.isscalar(regions[0]):
        regions = [tuple(regions)]
    per = [score_region(model, x, d, rg, mode, chunk, device) for rg in regions]
    w = np.array([p["n_scored"] for p in per], dtype=np.float64)
    w = w / w.sum()
    return {
        "official_r": float(np.dot(w, [p["official_r"] for p in per])),
        "official_rmse": float(np.dot(w, [p["official_rmse"] for p in per])),
        "r": np.nansum(np.stack([p["r"] for p in per], 0) * w[:, None], axis=0),
        "rmse": np.nansum(np.stack([p["rmse"] for p in per], 0) * w[:, None], axis=0),
        "units": per[0]["units"], "coverage": float(np.dot(w, [p["coverage"] for p in per])),
        "n_scored": int(sum(p["n_scored"] for p in per)),
        "blocks": [p["official_r"] for p in per],
    }


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------
def print_fingers(label, s):
    cells = "".join(f"{s['r'][i]:>8.4f}" for i in range(5))
    print(f"    {label:<22}{cells} | official r {s['official_r']:>7.4f} "
          f"| RMSE {s['official_rmse']:>6.4f} {s['units']}")


def mean_std(v):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(v) / n
    return (m, 0.0) if n < 2 else (m, (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Alignment and metric self-tests. No dataset, no model needed.
    rng = np.random.default_rng(0)
    n_raw, decim = 20_000, 10
    n_proc = n_raw // decim

    # A perfect decoder scored through the real path must return r == 1.
    t = np.cumsum(rng.standard_normal((n_raw, 5)), axis=0)
    import scipy.signal as sg
    t = sg.sosfiltfilt(sg.butter(4, 0.01, output="sos"), t, axis=0)
    pred = t[::decim]
    s = official_score(pred, t, 0, 0, n_raw, decim, "zoh")
    print(f"perfect decoder, zoh    : official r = {s['official_r']:.5f} "
          f"(coverage {s['coverage']:.3f})")
    assert s["official_r"] > 0.999

    s = official_score(pred, t, 0, 0, n_raw, decim, "linear")
    print(f"perfect decoder, linear : official r = {s['official_r']:.5f}")
    assert s["official_r"] > 0.9999

    # An offset prediction must be penalised, and by the right amount.
    for lag_proc in (1, 5, 10):
        lagged = np.roll(pred, lag_proc, axis=0)
        s = official_score(lagged, t, 0, 0, n_raw, decim, "zoh")
        print(f"  prediction lagged {10 * lag_proc:>3} ms -> r = {s['official_r']:.4f}")

    # Partial coverage must be handled without imputing anything data-derived.
    s = official_score(pred[50:], t, 50, 0, n_raw, decim, "zoh")
    print(f"partial coverage        : official r = {s['official_r']:.4f}, "
          f"coverage {s['coverage']:.3f}")
    assert 0.9 < s["coverage"] < 1.0

    # Ring finger must not be in the average.
    bad = pred.copy()
    bad[:, EXCLUDED_FINGER_IDX] = rng.standard_normal(n_proc)
    s = official_score(bad, t, 0, 0, n_raw, decim, "zoh")
    print(f"ring finger randomised  : official r = {s['official_r']:.5f} "
          f"(unchanged => finger 4 correctly excluded)")
    assert s["official_r"] > 0.999

    # [FIX-5] the interleaved split must hold out the right amount, must not
    # overlap, must stay inside the training recording, and must produce
    # training windows that never touch a validation target.
    d = {"test_start_proc": 40_000, "train_skip_proc": 200,
         "feat": torch.zeros(60_000, 1, 1)}
    tr, va, te = split_bounds(d)
    assert len(va) == VAL_BLOCKS and te == (40_000, 60_000)
    n_tr, n_va = interval_len(tr), interval_len(va)
    frac = n_va / (n_tr + n_va)
    print(f"\ninterleaved split       : {len(tr)} train / {len(va)} val blocks, "
          f"val fraction {frac:.3f}")
    assert abs(frac - VAL_FRACTION) < 0.01
    flat = sorted(tr + va)
    for (a1, b1), (a2, b2) in zip(flat, flat[1:]):
        assert b1 <= a2, "split intervals overlap"
    assert flat[0][0] >= 200 and flat[-1][1] <= 40_000

    ds = SegmentDataset(torch.zeros(3, 60_000), torch.zeros(60_000, 5), tr,
                        window=WINDOW, step=STEP, n_electrodes=3, n_bands=1)
    val_idx = np.concatenate([np.arange(a, b) for a, b in va])
    for s0 in ds.starts:
        touched = np.arange(s0, s0 + WINDOW)
        assert not np.intersect1d(touched, val_idx).size, \
            "a training window covers validation targets"
    print(f"                          {len(ds)} training windows, none of "
          f"which supervise a held-out target")

    # Streaming z-score sanity.
    z = StreamingZScore()
    a, b = torch.randn(500, 4, 3) * 3 + 2, torch.randn(300, 4, 3) * 3 + 2
    za, zb = z(a), z(b)
    assert abs(float(zb[-1].mean())) < 5 and za.shape == a.shape
    print("\nAll protocol self-tests passed.")
