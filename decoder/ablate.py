"""
ablate.py — leave-one-out ablation for ECoGFlexNet.

Each row re-runs the pipeline with exactly ONE component reverted and reports
the PAIRED change in the competition metric.

    ROW              REVERTS                                        STAGE
    reference        nothing                                        —
    causal-target    zero-phase -> causal target smoothing          prep
    env-5hz          20 Hz -> 5 Hz envelope low-pass (83 ms delay)  prep
    split-filter     continuous -> independently filtered halves    prep
    no-log           log-compressed -> raw rectified envelopes      prep
    bands-4          7 bands -> conventional 4                      prep
    channels-legacy  screened -> hand-curated bad-channel list      prep
    channels-all     screened -> keep every electrode               prep
    no-pretrain      cross-subject trunk -> per-subject from scratch train
    no-gate          causal channel gate removed                    train
    no-smooth        learned causal output FIR removed              train
    skip-last        skip-sum aggregation -> last block only        train
    preset-nano      tiny -> nano                                   train
    preset-base      tiny -> base                                   train

WHAT WENT WRONG BEFORE, AND WHAT PREVENTS IT NOW
================================================
`score.py --csv` opens its output with mode "w", so every invocation
TRUNCATED it. Running `--seeds 3 4` after `--seeds 0 1 2` therefore deleted
seeds 0-2 from the reference CSV, and the paired comparison quietly fell back
to the two surviving seeds — while arms scored earlier still carried seeds
0-2. The result was a "paired" table whose two sides came from different
seeds, which manufactures effects out of nothing: `channels-all`, which is
identical to `reference` whenever screening rejects no electrodes, appeared to
shift the score by +0.016.

Three changes make that impossible rather than merely unlikely:

  1. THE SEEDS USED FOR SCORING COME FROM DISK, NOT FROM --seeds.
     `available_seeds()` globs the checkpoints a row actually has and scores
     all of them. --seeds now controls only what gets TRAINED, so a row can
     never be scored on a subset of the models it owns.

  2. SCORE OUTPUT IS MERGED, NEVER OVERWRITTEN.
     score writes a temporary part-file which is merged into the row's CSV
     keyed on (subject, seed, row). Re-scoring updates cells; it cannot
     delete them.

  3. THE MANIFEST IS KEYED PER SEED.
     Previously the key was the whole seed list, so "reference|0,1,2" and
     "reference|3,4" were unrelated entries and neither knew about the other.
     Each (row, seed) is now tracked on its own, so extending a row to more
     seeds trains exactly the missing ones.

The report prints the number of paired cells per row and warns whenever a row
and the reference do not share every cell.

SELECTION IS DISABLED BY DEFAULT
================================
A diagnostic run showed validation cannot rank seeds: per-finger sampling SE
(0.05-0.18) exceeds the across-seed spread on nearly every finger, and pinky's
val->test rank agreement was 0.22, below the 0.50 coin flip. With per-run
best-epoch selection on, that noise lands on BOTH sides of every paired
difference. This script passes `--fixed-epochs`; `--select` restores the old
behaviour if you want it as its own row.

RUN A NULL CONTROL FIRST
========================
`channels-all` is the built-in control: when screening rejects nothing it is
the reference under another name, so its Δ must be exactly 0.000 on every
cell. `--determinism` is the stronger version — it trains one configuration
twice under different tags and compares. If either is non-zero, nothing else
in the table is interpretable.

Usage:
    python decoder/ablate.py --plan
    python decoder/ablate.py --determinism --subjects sub1 --seeds 0
    python decoder/ablate.py --rows reference --seeds 0 1 2 3 4 --epochs 40
    python decoder/ablate.py --rows causal-target env-5hz --seeds 0 1 2 3 4 --epochs 40
    python decoder/ablate.py --collect --latex paper/
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

SUBJECTS = ["sub1", "sub2", "sub3"]
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "prep")
DEFAULT_BASE_DIR = os.path.join(REPO_ROOT, "data", "raw")
# Ablation checkpoints are kept apart from the shipped paper models.
DEFAULT_CKPT_DIR = os.path.join(REPO_ROOT, "runs", "checkpoints")

ROWS = {
    "reference":       ([], [], "Full pipeline"),
    "causal-target":   (["--target-causal"], [],
                        "Causal target smoothing (83\\,ms group delay)"),
    "env-5hz":         (["--env-cutoff", "5.0"], [],
                        "5\\,Hz envelope low-pass"),
    "split-filter":    (["--split-filter"], [],
                        "Train/test filtered independently"),
    "no-log":          (["--no-log"], [], "No log compression"),
    "bands-4":         (["--bands", "4"], [], "Four bands instead of seven"),
    "channels-legacy": (["--channels", "legacy"], [],
                        "Hand-curated bad-channel list"),
    "channels-all":    (["--channels", "all"], [], "No channel rejection"),
    "no-pretrain":     ([], ["--no-pretrain"], "No cross-subject pretraining"),
    "no-gate":         ([], ["--no-gate"], "No causal channel gate"),
    "no-smooth":       ([], ["--no-smooth"], "No learned output FIR"),
    "skip-last":       ([], ["--skip-last"], "No skip aggregation"),
    "preset-nano":     ([], ["--preset", "nano"], "\\texttt{nano} preset"),
    "preset-base":     ([], ["--preset", "base"], "\\texttt{base} preset"),
}
ORDER = list(ROWS)
REFERENCE = "reference"
NULL_ROW = "channels-all"          # must return exactly 0.000


# ======================================================================
# Paths and caching
# ======================================================================
def prep_key(name):
    """Rows sharing prep flags share a prep directory, so every train-only row
    reuses the reference's preprocessing instead of rebuilding a copy."""
    flags = ROWS[name][0]
    if not flags:
        return "ref"
    return hashlib.sha1(" ".join(flags).encode()).hexdigest()[:8]


def prep_dir_for(name, base_data):
    k = prep_key(name)
    return base_data if k == "ref" else f"{base_data}_{k}"


TAG_SCOPE = ""   # set from --tag-scope; empty keeps the original abl_<row> names


def tag_for(name):
    """Checkpoint tag. Checkpoints live next to the scripts, NOT in --out-dir,
    so two runs with different --epochs but no scope would overwrite each
    other's checkpoints and score each other's seeds."""
    return f"abl_{TAG_SCOPE}_{name}" if TAG_SCOPE else f"abl_{name}"


def available_seeds(name, subjects, ckpt_dir=DEFAULT_CKPT_DIR):
    """Seeds for which EVERY requested subject has a checkpoint on disk.

    Scoring is driven by this rather than by --seeds, so a later run with a
    different --seeds cannot cause a row to be scored on a subset of the
    models it actually has.
    """
    tag, seeds = tag_for(name), None
    for sub in subjects:
        found = set()
        for p in glob.glob(os.path.join(ckpt_dir, f"{tag}_{sub}_s*.pt")):
            m = re.search(r"_s(\d+)\.pt$", os.path.basename(p))
            if m:
                found.add(int(m.group(1)))
        seeds = found if seeds is None else (seeds & found)
    return sorted(seeds or [])


def manifest_path(out_dir):
    return os.path.join(out_dir, "manifest.json")


def load_manifest(out_dir):
    p = manifest_path(out_dir)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"prep": {}, "train": {}}


def save_manifest(out_dir, man):
    with open(manifest_path(out_dir), "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)


def script(name):
    """Absolute path to a sibling script.

    Child processes deliberately inherit the user's cwd, so relative data
    paths on the command line resolve where they were typed rather than where
    these scripts happen to live.
    """
    return os.path.join(HERE, name)


def _q(t):
    return f'"{t}"' if " " in t else t


def run(cmd, dry):
    print("  $ " + " ".join(_q(c) for c in cmd))
    if dry:
        return True
    if subprocess.run(cmd).returncode != 0:
        print("  !! command failed")
        return False
    return True


# ======================================================================
# CSV accumulation
# ======================================================================
def read_cells(path):
    """{(subject, seed): official_r} from a row's score CSV."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("row") != "official":
                continue
            try:
                out[(r["subject"], int(r["seed"]))] = float(r["official_r"])
            except (ValueError, KeyError, TypeError):
                continue
    return out


def merge_csv(dst, src):
    """Fold `src` into `dst`, keyed on (subject, seed, row).

    score.py truncates its --csv on every call, so without this a second
    invocation with different seeds silently deletes the first invocation's
    cells. Merging makes re-scoring idempotent and additive.
    """
    rows, fields = {}, None
    for path in (dst, src):
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            fields = fields or rd.fieldnames
            for r in rd:
                rows[(r.get("subject"), r.get("seed"), r.get("row"))] = r
    if not rows:
        return 0
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows.values())
    return len(rows)


def nan_cells(path):
    return [k for k, v in read_cells(path).items() if math.isnan(v)]


# ======================================================================
# Execution
# ======================================================================
def execute(name, a, man):
    prep_flags, train_flags, desc = ROWS[name]
    data_dir = prep_dir_for(name, a.data_dir)
    tag = tag_for(name)
    print(f"\n{'=' * 78}\nABLATION: {name} — {desc}\n{'=' * 78}")

    # -- prep (runs for EVERY row, reference included) -------------------
    pk = prep_key(name)
    have_prep = all(os.path.exists(os.path.join(data_dir, f"{s}_prep.pt"))
                    for s in a.subjects)
    if not have_prep or man["prep"].get(pk) != "done":
        if not run([PY, script("prep.py"), "--out-dir", data_dir,
                    "--base-dir", a.base_dir, "--subjects", *a.subjects,
                    *prep_flags], a.dry_run):
            return False
        if not a.dry_run:
            man["prep"][pk] = "done"
            save_manifest(a.out_dir, man)
    else:
        print(f"  prep {data_dir} already built (key {pk}) — reusing")

    # -- train only the seeds that are missing ---------------------------
    todo = [s for s in a.seeds
            if a.force or man["train"].get(f"{name}|{s}") != "done"]
    if todo:
        cmd = [PY, script("train.py"), "--data-dir", data_dir,
               "--ckpt-dir", a.ckpt_dir, "--tag", tag,
               "--subjects", *a.subjects,
               "--seeds", *[str(s) for s in todo], *train_flags]
        cmd += (["--epochs", str(a.epochs)] if a.select
                else ["--fixed-epochs", str(a.epochs)])
        if not run(cmd, a.dry_run):
            return False
        if not a.dry_run:
            for s in todo:
                man["train"][f"{name}|{s}"] = "done"
            save_manifest(a.out_dir, man)
    else:
        print(f"  train already done for {name} seeds "
              f"{','.join(str(s) for s in a.seeds)} — skipping")

    # -- score EVERY checkpoint this row has, and merge ------------------
    seeds = a.seeds if a.dry_run else available_seeds(name, a.subjects, a.ckpt_dir)
    if not seeds:
        print(f"  no checkpoints found for {tag} — nothing to score")
        return False
    csv_path = os.path.join(a.out_dir, f"{name}.csv")
    part = os.path.join(a.out_dir, f"{name}.part.csv")
    if not run([PY, script("score.py"), "--data-dir", data_dir,
                "--ckpt-dir", a.ckpt_dir,
                "--tag", tag, "--subjects", *a.subjects,
                "--seeds", *[str(s) for s in seeds], "--csv", part], a.dry_run):
        return False
    if a.dry_run:
        return True

    n = merge_csv(csv_path, part)
    if os.path.exists(part):
        os.remove(part)
    cells = read_cells(csv_path)
    print(f"  merged -> {os.path.basename(csv_path)}: {n} rows, "
          f"{len(cells)} scored cells, seeds {sorted({s for _, s in cells})}")

    bad = nan_cells(csv_path)
    if bad:
        print(f"  !! {len(bad)} NaN cells {bad[:4]} — NOT marking done; "
              f"investigate before trusting this row")
        for s in a.seeds:
            man["train"].pop(f"{name}|{s}", None)
        save_manifest(a.out_dir, man)
        return False
    return True


# ======================================================================
# Paired statistics
# ======================================================================
def paired_stats(arm, ref):
    shared = sorted(set(arm) & set(ref))
    if not shared:
        return None
    d = np.array([arm[k] - ref[k] for k in shared])
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    per_sub = {}
    for sub in sorted({s for s, _ in shared}):
        per_sub[sub] = float(np.mean([arm[k] - ref[k]
                                      for k in shared if k[0] == sub]))
    return dict(n=n, n_arm=len(arm), n_ref=len(ref), mean=mean, se=se,
                per_subject=per_sub,
                n_negative=int((d < 0).sum()), n_positive=int((d > 0).sum()),
                max_abs=float(np.abs(d).max()),
                abs_mean=float(np.mean([arm[k] for k in shared])),
                t=(mean / se) if (se == se and se > 0) else float("nan"))


def collect(out_dir):
    cells = {n: read_cells(os.path.join(out_dir, f"{n}.csv")) for n in ORDER}
    return {n: c for n, c in cells.items() if c}


def report(cells, latex_dir=None, out_dir=None):
    if REFERENCE not in cells:
        print(f"\nNo '{REFERENCE}' row. Every delta is relative to it:\n"
              f"  python decoder/ablate.py --rows {REFERENCE}")
        return {}

    ref = cells[REFERENCE]
    ref_seeds = sorted({s for _, s in ref})
    print(f"\n{'=' * 100}")
    print("ABLATION — paired change in official r vs the full pipeline")
    print(f"reference: {len(ref)} cells, seeds {ref_seeds}, "
          f"mean r {np.mean(list(ref.values())):.4f}")
    print("=" * 100)
    print(f"{'row':<17}{'mean r':>9}{'delta':>9}{'SE':>8}{'t':>7}{'n':>4}"
          f"{'sign':>8}   {'per-subject delta':<26}")
    print("-" * 100)

    res, warnings = {}, []
    for name in ORDER:
        if name not in cells or name == REFERENCE:
            continue
        st = paired_stats(cells[name], ref)
        if st is None:
            warnings.append(f"{name}: shares no cells with the reference")
            continue
        res[name] = st
        if st["n"] < max(st["n_arm"], st["n_ref"]):
            warnings.append(
                f"{name}: paired on {st['n']} cells, but the row has "
                f"{st['n_arm']} and the reference {st['n_ref']} — unmatched "
                f"cells were DROPPED. Score the missing seeds before "
                f"quoting this row.")
        sign = f"{st['n_negative']}-/{st['n_positive']}+"
        ps = "  ".join(f"{s[-1]}:{v:+.3f}" for s, v in st["per_subject"].items())
        print(f"{name:<17}{st['abs_mean']:>9.4f}{st['mean']:>+9.4f}"
              f"{st['se']:>8.4f}{st['t']:>7.1f}{st['n']:>4}{sign:>8}"
              f"   {ps:<26}")

    print("-" * 100)
    print("delta < 0 means removing that component HURT, i.e. it was earning")
    print("its place. |t| > ~2 is the usual bar; below that, report the row as")
    print("measured-and-null rather than dropping it. A mean that hides two")
    print("signs in the per-subject column is not one effect.")

    if NULL_ROW in res:
        st = res[NULL_ROW]
        print(f"\n{'=' * 100}\nNULL CONTROL\n{'=' * 100}")
        if st["max_abs"] < 1e-9:
            print(f"  {NULL_ROW}: delta = 0.000000 across all {st['n']} cells.")
            print("  The pipeline is deterministic and the pairing is sound.")
        else:
            print(f"  {NULL_ROW}: delta = {st['mean']:+.4f}, largest "
                  f"single-cell difference {st['max_abs']:.4f}.")
            print("  This row must be EXACTLY zero whenever channel screening")
            print("  rejects no electrodes. It is not. Either screening did")
            print("  reject something (check the prep log), or training is")
            print("  nondeterministic — in which case no delta above is")
            print("  interpretable until that is fixed. Run --determinism.")

    if warnings:
        print(f"\n{'=' * 100}\nWARNINGS\n{'=' * 100}")
        for w in warnings:
            print(f"  * {w}")

    missing = [n for n in ORDER if n not in cells]
    if missing:
        print(f"\nnot yet run: {', '.join(missing)}")

    if latex_dir:
        write_latex(latex_dir, ref, res)
    if out_dir:
        dump_csv(out_dir, res)
    return res


def write_latex(latex_dir, ref, res):
    os.makedirs(latex_dir, exist_ok=True)
    path = os.path.join(latex_dir, "tab_ablation.tex")
    rm = np.mean(list(ref.values()))
    rs = np.std(list(ref.values()), ddof=1) if len(ref) > 1 else float("nan")
    with open(path, "w") as f:
        f.write("% AUTO-GENERATED by ablate.py. Do not edit.\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Configuration & Mean $r$ & $\\Delta r$ & $t$ \\\\\n\\midrule\n")
        f.write(f"\\textbf{{Full pipeline}} & \\textbf{{{rm:.3f} $\\pm$ "
                f"{rs:.3f}}} & --- & --- \\\\\n\\midrule\n")
        for name in ORDER:
            if name not in res:
                continue
            st = res[name]
            f.write(f"\\quad {ROWS[name][2]} & {st['abs_mean']:.3f} & "
                    f"${st['mean']:+.3f} \\pm {st['se']:.3f}$ & "
                    f"${st['t']:.1f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {path}")


def dump_csv(out_dir, res):
    path = os.path.join(out_dir, "paired_deltas.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "n_paired", "n_arm", "n_ref", "mean_r", "delta",
                    "se", "t", "n_neg", "n_pos", "max_abs"]
                   + [f"delta_{s}" for s in SUBJECTS])
        for name, st in res.items():
            w.writerow([name, st["n"], st["n_arm"], st["n_ref"],
                        f"{st['abs_mean']:.4f}", f"{st['mean']:.4f}",
                        f"{st['se']:.4f}", f"{st['t']:.2f}",
                        st["n_negative"], st["n_positive"],
                        f"{st['max_abs']:.6f}"]
                       + [f"{st['per_subject'].get(s, float('nan')):.4f}"
                          for s in SUBJECTS])
    print(f"wrote {path}")


# ======================================================================
# Determinism probe
# ======================================================================
def determinism(a):
    """Train the SAME configuration twice under different tags and compare.

    This is the check that should precede any ablation. If two identical runs
    disagree by more than rounding, the paired SE is a fiction and every delta
    in the table is noise of unknown size.
    """
    print(f"{'=' * 78}\nDETERMINISM PROBE\n{'=' * 78}")
    got = {}
    for tag in ("det_a", "det_b"):
        cmd = [PY, script("train.py"), "--data-dir", a.data_dir,
               "--ckpt-dir", a.ckpt_dir,
               "--tag", tag, "--subjects", *a.subjects,
               "--seeds", *[str(s) for s in a.seeds]]
        cmd += (["--epochs", str(a.epochs)] if a.select
                else ["--fixed-epochs", str(a.epochs)])
        if not run(cmd, a.dry_run):
            return
        out = os.path.join(a.out_dir, f"{tag}.csv")
        if not run([PY, script("score.py"), "--data-dir", a.data_dir,
                    "--ckpt-dir", a.ckpt_dir,
                    "--tag", tag, "--subjects", *a.subjects,
                    "--seeds", *[str(s) for s in a.seeds],
                    "--csv", out], a.dry_run):
            return
        got[tag] = read_cells(out)
    if a.dry_run:
        return

    shared = sorted(set(got["det_a"]) & set(got["det_b"]))
    print(f"\n{'cell':<16}{'run A':>10}{'run B':>10}{'diff':>13}")
    print("-" * 49)
    worst = 0.0
    for k in shared:
        d = got["det_a"][k] - got["det_b"][k]
        worst = max(worst, abs(d))
        print(f"{k[0] + ' s' + str(k[1]):<16}{got['det_a'][k]:>10.4f}"
              f"{got['det_b'][k]:>10.4f}{d:>+13.6f}")
    print("-" * 49)
    if worst < 1e-6:
        print(f"DETERMINISTIC (largest difference {worst:.2e}).")
        print("Paired deltas measure the ablation and nothing else.")
    else:
        print(f"NONDETERMINISTIC — largest difference {worst:.4f}.")
        print("That is the true noise floor of every row, and it is added to")
        print("BOTH sides of each paired comparison, so any delta smaller than")
        print("it is unmeasurable. Likely causes, in order:")
        print("  * torch.use_deterministic_algorithms(False) in set_seed()")
        print("  * DataLoader shuffling without a seeded generator")
        print("  * augmentation RNG drawn from global state, not a per-run one")
        print("  * cuDNN autotuning, if this ran on GPU")


# ======================================================================
def preflight(a):
    bad = []
    for s_ in ("prep.py", "train.py", "score.py"):
        if not os.path.exists(script(s_)):
            bad.append(f"{s_} is not next to ablate.py ({HERE})")
    missing = [s for s in a.subjects
               if not os.path.exists(os.path.join(a.base_dir, f"{s}_comp.mat"))]
    if missing:
        bad.append(f"no {{sub}}_comp.mat for {missing} under --base-dir "
                   f"'{a.base_dir}'\n      looked in : "
                   f"{os.path.abspath(a.base_dir)}\n      cwd       : "
                   f"{os.getcwd()}")
    if not a.select and os.path.exists(script("train.py")):
        try:
            if "--fixed-epochs" not in open(script("train.py"),
                                            encoding="utf-8").read():
                bad.append("train.py has no --fixed-epochs flag; add it or "
                           "pass --select")
        except OSError:
            pass
    return bad


def plan(a, rows):
    print(f"{'=' * 78}\nPLAN\n{'=' * 78}")
    keys = {}
    for name in rows:
        keys.setdefault(prep_key(name), []).append(name)
    print(f"  {len(rows)} rows, {len(keys)} prep director"
          f"{'y' if len(keys) == 1 else 'ies'}:")
    for k, names in keys.items():
        print(f"    {prep_dir_for(names[0], a.data_dir):<28} <- "
              f"{', '.join(names)}")
    print()
    for name in rows:
        have = available_seeds(name, a.subjects)
        todo = [s for s in a.seeds if s not in have]
        print(f"    {name:<17} on disk {have or '[]'}  ->  train {todo or '[]'}")
    print(f"\n  selection: "
          f"{'per-run best epoch' if a.select else f'OFF, fixed {a.epochs} epochs'}")
    print("  scoring uses the seeds found on disk, not --seeds")
    if NULL_ROW not in rows:
        print(f"\n  NOTE: '{NULL_ROW}' is not in this run. It is the null")
        print("  control and must come out at exactly 0.000 — run it once.")


def selftest():
    rng = np.random.default_rng(0)
    subs, seeds = SUBJECTS, range(5)
    base = {"sub1": 0.50, "sub2": 0.44, "sub3": 0.65}
    ref = {(s, k): base[s] + 0.03 * rng.standard_normal()
           for s in subs for k in seeds}
    arm = {k: v - 0.02 + 0.005 * rng.standard_normal() for k, v in ref.items()}

    st = paired_stats(arm, ref)
    pooled = np.std(list(arm.values()), ddof=1)
    print(f"  true effect -0.020 -> paired delta {st['mean']:+.4f} "
          f"+- {st['se']:.4f} (t {st['t']:.1f}, n {st['n']})")
    print(f"  pooled std of the arm alone {pooled:.4f} = "
          f"{pooled / st['se']:.0f}x the paired SE")
    assert abs(st["mean"] + 0.020) < 0.005 and abs(st["t"]) > 5

    assert paired_stats(dict(ref), ref)["max_abs"] == 0.0
    print("  identical arm -> max_abs exactly 0.0 (the null-control test)")

    partial = {k: v for k, v in arm.items() if k[1] < 3}
    stp = paired_stats(partial, ref)
    assert stp["n"] == 9 and stp["n_ref"] == 15
    print(f"  unequal seeds: paired on {stp['n']} of {stp['n_ref']} — "
          f"reported, so the drop cannot pass unnoticed")

    with tempfile.TemporaryDirectory() as td:
        dst, p1 = os.path.join(td, "r.csv"), os.path.join(td, "p.csv")

        def write(path, seeds_):
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["subject", "seed", "row", "official_r"])
                for s in seeds_:
                    w.writerow(["sub1", s, "official", 0.5 + 0.01 * s])

        write(p1, [0, 1, 2])
        merge_csv(dst, p1)
        write(p1, [3, 4])
        merge_csv(dst, p1)
        cells = read_cells(dst)
        assert sorted(s for _, s in cells) == [0, 1, 2, 3, 4], cells
        print(f"  merge_csv: 0-2 then 3-4 -> {len(cells)} cells retained "
              f"(plain overwrite would have left 2)")

    assert prep_key("no-gate") == prep_key("reference") == "ref"
    print("  prep dedup: train-only rows reuse the reference prep")
    print("\n  All ablate self-tests passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR,
                    help="where ablation checkpoints are written and read")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "runs", "ablation"))
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--latex", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag-scope", default="",
                    help="prefix for checkpoint tags, e.g. e80; use a distinct "
                         "value for every --out-dir that trains models")
    a = ap.parse_args()
    global TAG_SCOPE
    TAG_SCOPE = a.tag_scope

    if a.selftest:
        selftest()
        return

    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = a.rows or ORDER
    unknown = [r for r in rows if r not in ROWS]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; choose from {ORDER}")

    if a.plan:
        plan(a, rows)
        return

    problems = preflight(a) if not a.collect else []
    if problems:
        print("PREFLIGHT FAILED\n")
        for p_ in problems:
            print(f"  * {p_}")
        raise SystemExit(1)

    if a.determinism:
        determinism(a)
        return

    if not a.collect:
        rows = ([REFERENCE] if REFERENCE in rows else []) + \
               [r for r in rows if r != REFERENCE]
        man = load_manifest(a.out_dir)
        failed = [n for n in rows if not execute(n, a, man)]
        if failed:
            print(f"\nFAILED ROWS: {', '.join(failed)}")
        if a.dry_run:
            return

    report(collect(a.out_dir), a.latex, a.out_dir)


if __name__ == "__main__":
    main()
