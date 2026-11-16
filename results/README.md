# results/

Every raw result the paper reports, as written by the scripts. The tables,
macros and figure data in `figures/` are generated from these files only.

| Folder / file | Contents | Written by |
|---|---|---|
| `main/main_results.csv` | FP32 test scores of the 15 shipped models (`official` is the paper's metric; the other rows are the evaluation-protocol ladder) | `decoder/score.py --tag paper --csv` |
| `main/ensemble_results.csv` | the 5-seed FP32 ensemble per subject | `decoder/score.py --ensemble-csv` |
| `main/int8_ptq_results.csv` | deployed INT8 models (post-training quantisation), scored on the host | `decoder/export_int8.py --qat-epochs 0 --score` |
| `main/int8_qat_results.csv` | INT8 after 30 epochs of quantisation-aware training | `decoder/export_int8.py --score` |
| `main/eca_gate_leak.csv` | the time-pooling (ECA) gate scored densely, on training windows, and as exported | `decoder/score_gate_leak.py --tag abl_eca-gate` |
| `ablation_e40/` | leave-one-out ablation, 40 epochs, 5 seeds x 3 subjects per row; `paired_deltas.csv` has the paired differences | `decoder/ablate.py --epochs 40` |
| `ablation_e80/` | the same grid at 80 epochs (replication) | `decoder/ablate.py --epochs 80` |
| `reference_e120/` | reference and causal-target rows at 120 epochs (training-budget check) | `decoder/ablate.py --epochs 120` |
| `board/accuracy.json`, `preds_*.npy` | per-call latency and the predictions of all 15 models on the board | `board/board_bench.py accuracy` |
| `board/board_int8_scores.csv` | those predictions scored, next to the host INT8 scores | `board/score_board.py` |
| `board/sweep.json` | call time vs labels per call | `board/board_bench.py sweep` |
| `board/deploy_board.csv` | compute, latency and MACs vs labels per call, calibrated on the sweep | `decoder/deploy_sweep.py --csv` |
| `board/soak.json` | 10-minute sustained run with temperature and clock | `board/board_bench.py soak` |
| `board/paced_*` | real-time replay at 100 Hz, sleep-wait and spin-wait; `paced_summary.json` pools the subjects | `board/board_bench.py paced`, `board/analyze_paced.py` |
| `board/*.log` | console output of the board runs | |
