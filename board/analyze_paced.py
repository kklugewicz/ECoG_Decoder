"""Label-age distributions from the board's real-time replay (per-batch .npy).

label age = time from the OLDEST sample in a batch arriving to the decoder
emitting that batch: (E-1)/fs batching delay + wake-up/scheduling delay +
compute. With E=10 at 100 Hz the batching part alone is 90 ms.

    python board/analyze_paced.py                          # board/results
    python board/analyze_paced.py --results results/board  # the paper's run

Writes paced_summary.json into the results folder.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET, BATCH_MS = 100.0, 90.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    R = ap.parse_args().results

    print(f"{'wait':<6}{'subj':<6}{'n':>6}{'median':>8}{'p95':>8}{'p99':>8}{'p99.9':>8}"
          f"{'max':>8}{'>100ms':>9}{'  overhead p99 (age-90-compute)':>8}")
    summary = {}
    for tag in ("sleep", "spin"):
        allv = []
        for sub in ("sub1", "sub2", "sub3"):
            p = os.path.join(R, f"paced_{tag}_{sub}_age_ms.npy")
            if not os.path.exists(p):
                continue
            a = np.load(p)
            j = json.load(open(os.path.join(R, f"paced_{tag}_{sub}.json")))
            allv.append(a)
            q = np.percentile(a, [50, 95, 99, 99.9])
            miss = int((a > BUDGET).sum())
            comp99 = j["compute"]["p99"]
            print(f"{tag:<6}{sub:<6}{len(a):>6}{q[0]:>8.1f}{q[1]:>8.1f}{q[2]:>8.1f}{q[3]:>8.1f}"
                  f"{a.max():>8.1f}{miss:>5} ({100 * miss / len(a):.2f}%)"
                  f"   compute p99 {comp99:.1f} ms, overruns {j['overruns']}, "
                  f"temp max {j['thermal']['temp_max']:.0f} C, min {j['thermal']['mhz_min']:.0f} MHz")
        if allv:
            a = np.concatenate(allv)
            q = np.percentile(a, [50, 95, 99, 99.9])
            miss = int((a > BUDGET).sum())
            summary[tag] = dict(n=len(a), median=q[0], p95=q[1], p99=q[2], p999=q[3],
                                max=float(a.max()), misses=miss, miss_pct=100 * miss / len(a))
            print(f"{tag:<6}{'ALL':<6}{len(a):>6}{q[0]:>8.1f}{q[1]:>8.1f}{q[2]:>8.1f}{q[3]:>8.1f}"
                  f"{a.max():>8.1f}{miss:>5} ({100 * miss / len(a):.2f}%)")
            # smallest budget that 99% / 99.9% of labels meet
            print(f"        budget met by 99% of labels: {q[2]:.0f} ms | by 99.9%: {q[3]:.0f} ms\n")
    json.dump(summary, open(os.path.join(R, "paced_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
