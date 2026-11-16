"""
board_bench.py — runs ON the board (Radxa ZERO 3 in the paper; any Linux
machine with the ExecuTorch runtime works). INT8 decoder study.

  accuracy  every emit-10 model walked over the full test recording, back to
            back; predictions saved for scoring on the host with
            decoder/protocol.py (so the metric code is identical), per-call
            latency recorded.
  sweep     per-call latency at emit 1/5/10/25/50, seed-0 model per subject.
  paced     full test recording replayed in real time at 100 Hz: a batch of
            E samples becomes available every E/fs seconds; each call starts
            when its last sample arrives (or when the previous call finishes,
            if it overran). Records compute time, label age, deadline misses.
  soak      continuous back-to-back inference for N seconds, logging SoC
            temperature and CPU frequency once a second (throttle check).

Temperature and current CPU frequency are sampled in every mode (NaN where
the kernel does not expose them).

Inputs come from board/dump_inputs.py (run on the host); the .pte files are
the ones in checkpoints/. See board/README.md.
"""
import argparse, glob, json, os, threading, time
import numpy as np
import torch
from executorch.runtime import Runtime

FS = 100.0
RT = Runtime.get()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def soc_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except OSError:
        return float("nan")


def cpu_mhz():
    try:
        return int(open("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq").read()) / 1000.0
    except OSError:
        return float("nan")


class Sampler(threading.Thread):
    """Temperature + frequency once a second in the background."""
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop_flag = [], False
    def run(self):
        t0 = time.monotonic()
        while not self.stop_flag:
            self.rows.append((round(time.monotonic() - t0, 2), soc_temp(), cpu_mhz()))
            time.sleep(1.0)
    def stop(self):
        self.stop_flag = True
        self.join()
        temps = [r[1] for r in self.rows] or [float("nan")]
        mhz = [r[2] for r in self.rows] or [float("nan")]
        return dict(temp_start=temps[0], temp_max=max(temps), temp_end=temps[-1],
                    mhz_min=min(mhz), mhz_max=max(mhz), trace=self.rows)


def governor():
    try:
        return open("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read().strip()
    except OSError:
        return "n/a"


def load(pte):
    return RT.load_program(pte).load_method("forward")


RF = 284   # receptive field of the tiny preset (same for every subject)


def window_of(emit):
    """Exported window = RF + E - 1 (decoder/export_int8.py)."""
    return RF + emit - 1


def stats(ms):
    a = np.sort(np.asarray(ms))
    pick = lambda q: float(a[min(len(a) - 1, int(q * len(a)))])
    return dict(n=len(a), median=pick(0.5), p95=pick(0.95), p99=pick(0.99),
                max=float(a[-1]), mean=float(a.mean()))


def walk(method, x, off, n_test, window, emit, paced=False, spin=False):
    """Run the decoder across the test region. Returns preds (n_test,5) and
    per-call records. Mirrors export_int8.pte_predict_region exactly."""
    preds = np.empty((n_test, 5), dtype=np.float32)
    recs = []
    period = emit / FS
    t_start = time.monotonic()
    busy_until = t_start
    t = 0
    k = 0
    while t < n_test:
        e = min(emit, n_test - t)
        s = off + t + e - window
        chunk = torch.from_numpy(np.ascontiguousarray(x[:, s:off + t + e][None]))
        if paced:
            arrive = t_start + (k + 1) * period     # last sample of batch k lands
            if spin:
                # busy-wait: the core never enters deep idle between batches
                while time.monotonic() < arrive:
                    pass
            else:
                now = time.monotonic()
                if now < arrive:
                    time.sleep(arrive - now)
            begin = max(time.monotonic(), arrive)
        else:
            begin = time.monotonic()
        y = method.execute((chunk,))[0]
        end = time.monotonic()
        preds[t:t + e] = y[0, :, -e:].T.numpy()
        rec = dict(compute_ms=(end - begin) * 1000.0)
        if paced:
            # oldest label in the batch was sampled (e-1)/fs before arrival
            rec["label_age_ms"] = ((e - 1) / FS + (end - arrive)) * 1000.0
            rec["late_ms"] = (end - (arrive + period)) * 1000.0   # >0: next batch waiting
        recs.append(rec)
        t += e
        k += 1
    return preds, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["accuracy", "sweep", "paced", "soak"])
    ap.add_argument("--models", default=os.path.join(REPO, "checkpoints"),
                    help="folder with the .pte files")
    ap.add_argument("--tag", default="paper_ptq",
                    help="model file prefix: <tag>_<sub>_s<seed>_int8.pte")
    ap.add_argument("--inputs", default=os.path.join(HERE, "inputs"),
                    help="folder written by dump_inputs.py")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--soak-s", type=int, default=600)
    ap.add_argument("--budget-ms", type=float, default=100.0)
    ap.add_argument("--spin", action="store_true",
                    help="paced mode: busy-wait between batches instead of sleeping")
    ap.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.inputs, "meta.json")))
    X = {s: np.load(os.path.join(a.inputs, f"{s}_x.npy")) for s in meta}
    print(f"temp {soc_temp():.1f} C, cpu {cpu_mhz():.0f} MHz, governor {governor()}")
    report = {}

    if a.mode == "accuracy":
        for sub, m in meta.items():
            for seed in range(5):
                pte = os.path.join(a.models, f"{a.tag}_{sub}_s{seed}_int8.pte")
                meth = load(pte)
                emit = 10
                w = window_of(emit)
                smp = Sampler(); smp.start()
                preds, recs = walk(meth, X[sub], m["offset"], m["n_test"], w, emit)
                th = smp.stop()
                np.save(os.path.join(a.out, f"preds_{sub}_s{seed}.npy"), preds)
                st = stats([r["compute_ms"] for r in recs][5:])
                report[f"{sub}_s{seed}"] = dict(emit=emit, window=w, latency=st,
                                                temp_max=th["temp_max"], mhz_min=th["mhz_min"])
                print(f"{sub} s{seed}: emit {emit} | median {st['median']:.2f} ms p95 "
                      f"{st['p95']:.2f} p99 {st['p99']:.2f} | temp max {th['temp_max']:.1f} C "
                      f"| min {th['mhz_min']:.0f} MHz", flush=True)

    elif a.mode == "sweep":
        for sub, m in meta.items():
            for emit in (1, 5, 10, 25, 50):
                name = (f"{a.tag}_{sub}_s0_int8.pte" if emit == 10
                        else f"{a.tag}_e{emit}_{sub}_s0_int8.pte")
                meth = load(os.path.join(a.models, name))
                w = window_of(emit)
                n = min(m["n_test"], 3000 * emit)       # >= ~3000 calls, or all
                smp = Sampler(); smp.start()
                _, recs = walk(meth, X[sub], m["offset"], n, w, emit)
                th = smp.stop()
                st = stats([r["compute_ms"] for r in recs][10:])
                report[f"{sub}_e{emit}"] = dict(emit=emit, window=w, latency=st,
                                                per_label_ms=st["median"] / emit,
                                                temp_max=th["temp_max"], mhz_min=th["mhz_min"])
                print(f"{sub} emit {emit:>2} (window {w}): median {st['median']:.2f} ms "
                      f"p95 {st['p95']:.2f} p99 {st['p99']:.2f} max {st['max']:.2f} | "
                      f"{st['median'] / emit:.3f} ms/label | max {th['temp_max']:.1f} C, "
                      f"min {th['mhz_min']:.0f} MHz", flush=True)

    elif a.mode == "paced":
        tag = "spin" if a.spin else "sleep"
        for sub, m in meta.items():
            if a.subjects and sub not in a.subjects:
                continue
            meth = load(os.path.join(a.models, f"{a.tag}_{sub}_s0_int8.pte"))
            emit = 10; w = window_of(emit)
            smp = Sampler(); smp.start()
            t0 = time.monotonic()
            _, recs = walk(meth, X[sub], m["offset"], m["n_test"], w, emit,
                           paced=True, spin=a.spin)
            wall = time.monotonic() - t0
            th = smp.stop()
            age = [r["label_age_ms"] for r in recs]
            late = [r["late_ms"] for r in recs]
            report[sub] = dict(emit=emit, calls=len(recs), wall_s=wall,
                               compute=stats([r["compute_ms"] for r in recs]),
                               label_age=stats(age),
                               budget_misses=int(sum(x > a.budget_ms for x in age)),
                               overruns=int(sum(x > 0 for x in late)),
                               wait=tag, thermal=th)
            # persist per subject immediately: a crash must not lose earlier subjects
            np.save(os.path.join(a.out, f"paced_{tag}_{sub}_age_ms.npy"), np.asarray(age))
            json.dump(report[sub], open(os.path.join(a.out, f"paced_{tag}_{sub}.json"), "w"), indent=1)
            if hasattr(os, "sync"):
                os.sync()
            print(f"{sub}: {len(recs)} calls over {wall:.1f} s (real time "
                  f"{m['n_test'] / FS:.1f} s) | label age median "
                  f"{np.median(age):.1f} ms max {max(age):.1f} | misses >{a.budget_ms:.0f} ms: "
                  f"{report[sub]['budget_misses']} | overruns {report[sub]['overruns']} | "
                  f"temp {th['temp_start']:.1f}->{th['temp_max']:.1f} C, min "
                  f"{th['mhz_min']:.0f} MHz", flush=True)

    elif a.mode == "soak":
        sub = "sub3"
        m = meta[sub]
        meth = load(os.path.join(a.models, f"{a.tag}_{sub}_s0_int8.pte"))
        emit = 10; w = window_of(emit)
        smp = Sampler(); smp.start()
        t0, calls, lat = time.monotonic(), 0, []
        while time.monotonic() - t0 < a.soak_s:
            _, recs = walk(meth, X[sub], m["offset"], m["n_test"], w, emit)
            lat += [r["compute_ms"] for r in recs]
            calls += len(recs)
            el = time.monotonic() - t0
            print(f"  {el:6.0f} s | {calls} calls | temp {soc_temp():.1f} C | "
                  f"{cpu_mhz():.0f} MHz | last-pass median "
                  f"{np.median([r['compute_ms'] for r in recs]):.2f} ms", flush=True)
        th = smp.stop()
        n = len(lat)
        report = dict(seconds=a.soak_s, calls=calls, thermal=th,
                      first_minute=stats(lat[: n // 10]), last_minute=stats(lat[-n // 10:]))
        print(f"soak: temp {th['temp_start']:.1f} -> max {th['temp_max']:.1f} C | "
              f"CPU {th['mhz_min']:.0f}-{th['mhz_max']:.0f} MHz | median first 10% "
              f"{report['first_minute']['median']:.2f} ms, last 10% "
              f"{report['last_minute']['median']:.2f} ms")

    json.dump(report, open(os.path.join(a.out, f"{a.mode}.json"), "w"), indent=1)
    print(f"wrote {a.out}/{a.mode}.json")


if __name__ == "__main__":
    main()
