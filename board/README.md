# On-board benchmark

`board_bench.py` runs the INT8 models with the ExecuTorch runtime on the board
itself. Everything else here runs on the host. The paper's run (Radxa ZERO 3,
RK3566, 4 x Cortex-A55 at 1.8 GHz) is in `results/board/`.

| Mode | What it measures | Time |
|---|---|---|
| `accuracy` | all 15 deployed models over the full test recordings; saves the predictions for scoring on the host | 4 min |
| `sweep` | call time at 1, 5, 10, 25 and 50 labels per call (seed-0 models) | 4 min |
| `soak` | 10 minutes back to back, logging SoC temperature and CPU clock | 10 min |
| `paced` | the test recordings replayed in real time at 100 Hz: label age and deadline misses (`--spin` busy-waits between batches instead of sleeping) | 11 min each |

## 1. On the host: make the inputs

```bash
python decoder/prep.py            # if not done yet
python board/dump_inputs.py       # writes board/inputs/ (about 100 MB)
```

## 2. On the board: set up and run

Any 64-bit Linux board works. Clone this repository on it (or copy `board/`,
`checkpoints/paper_ptq_*` and `board/inputs/`), then:

```bash
python3 -m venv ~/et && ~/et/bin/pip install -r requirements-export.txt
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
PY=~/et/bin/python bash board/run_board.sh      # about 45 minutes
```

For the paper the board had a heatsink and a fan (without them an earlier run
reached 83 C, close to the throttle point) and the desktop session was idle.
`run_board.sh` waits for the SoC to cool to 50 C before each stage and stops
the benchmark if it passes 95 C. Masking suspend matters for the paced runs:
the desktop's idle suspend interrupted them before it was masked. Undo it
afterwards with `sudo systemctl unmask ...`.

## 3. Back on the host: score it

Copy `board/results/` from the board to the same place on the host, then

```bash
python board/score_board.py          # board predictions vs the host INT8 scores
python board/analyze_paced.py        # label-age percentiles -> paced_summary.json
python board/write_board_numbers.py --results board/results --out board/results
```

Each script takes `--results results/board` to re-analyse the paper's run
instead.
