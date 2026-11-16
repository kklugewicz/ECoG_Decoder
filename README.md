# ECoG_Decoder

Code, trained models and raw results for the paper *A 67 kB causal network
for finger-flexion decoding from ECoG: on-device latency, ablations and a
look-ahead leak* (in preparation).

A causal temporal convolutional network (25,369 parameters) decodes the
flexion of all five fingers from electrocorticography, sample by sample, on
[BCI Competition IV data set 4](https://www.bbci.de/competition/iv/) (three
subjects). Exported to INT8 with ExecuTorch it is a 66.8 kB file that runs in
real time on a Radxa ZERO 3 single-board computer.

| Official competition metric (test set) | Sub. 1 | Sub. 2 | Sub. 3 | Mean |
|---|---|---|---|---|
| FP32, mean of 5 seeds | 0.511 | 0.451 | 0.636 | **0.533** |
| INT8, on the board | 0.510 | 0.451 | 0.648 | **0.536** |
| 5-seed FP32 ensemble | 0.588 | 0.504 | 0.679 | 0.590 |

The metric is the competition's: Pearson r between prediction and data glove
at 1 kHz over the whole test recording, averaged over thumb, index, middle and
little finger (the ring finger is not scored). On the board, 99.9% of label
batches are emitted within 99.2 ms of their oldest sample arriving
(6,000 batches replayed at 100 Hz).

## Quick start

Python 3.10 or later, CPU only. Developed on Windows, tested on Linux.

```bash
git clone https://github.com/kklugewicz/ECoG_Decoder.git
cd ECoG_Decoder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # about 15 s, no data needed
```

The tests run the self-checks built into the code (causality, streaming
equivalence, filter alignment, and the metric itself: a perfect decoder must
score r = 1) and load all 15 shipped models.

## Reproduce the main result (about 5 minutes)

```bash
python scripts/get_data.py --agree     # downloads data set 4 (about 220 MB) into data/raw
python decoder/prep.py                 # features into data/prep (about 1 min, 330 MB)
python decoder/score.py --tag paper    # scores the 15 shipped models: mean r 0.533
pytest                                 # now also checks every score against results/main
```

The data are not included here. `--agree` means you accept the BCI
Competition IV terms: cite the data set paper (Schalk et al. 2007, see
[Citation](#citation)) and credit the group that recorded it. If the download
is blocked, `scripts/get_data.py` prints the two links to fetch by hand.

## Train from scratch

```bash
python decoder/train.py --seeds 0 1 2 3 4    # writes checkpoints/flexnet_<sub>_s<seed>.pt
python decoder/score.py                      # scores them (tag flexnet)
```

Each seed is a cross-subject pretraining stage (about 8 minutes) followed by
a fine-tune per subject (about 7 minutes each), so about half an hour per seed
with four CPU threads and about two and a half hours for all five. Training on
the CPU is bit-deterministic on a given machine: the same retrain run twice
gave byte-identical checkpoints. On a different CPU or PyTorch build the
floating-point arithmetic differs in the last bits, and over a full run that
moves the result about as much as a change of seed does. A retrain of seed 0
on a Linux machine scored a mean r of 0.544 (0.519, 0.475 and 0.636 for the
three subjects) against 0.552 for the shipped seed 0. The five shipped seeds
range from 0.509 to 0.552, and each subject's retrained score falls inside
that subject's five-seed range. Compare a retrain with the five-seed mean,
not with one shipped model.

The leave-one-out ablation is `decoder/ablate.py`; `--plan` prints what it
will run and `--help` lists the options. Its checkpoints go to
`runs/`, away from the shipped models.

## INT8 export with ExecuTorch

```bash
pip install -r requirements-export.txt       # torch 2.12.1, torchao 0.17.0, executorch 1.3.1
python decoder/export_int8.py --tag paper --pte-tag my_ptq --qat-epochs 0 --score --int8-csv my_ptq.csv
```

This is the post-training quantisation that produced the deployed models:
per-channel symmetric INT8 weights, observers calibrated on 50 training
batches, XNNPACK backend, 10 labels per call. `--qat-epochs 30` (the default)
gives the quantisation-aware variant the paper compares it with. A re-export
on another machine lands within about 1e-4 of the shipped model's score; the
shipped `.pte` files are the ones that ran on the board. The version pins
matter: the quantisation and lowering APIs change between releases.

## Run it on a board

See [board/README.md](board/README.md). Any Linux machine with the ExecuTorch
runtime can run the benchmark; the paper used a Radxa ZERO 3 (RK3566,
4 x Cortex-A55).

## Regenerate every table, figure and number in the paper

Each command reads raw results and writes the LaTeX macros and tables the
paper inputs (into `figures/tex/`) or the per-figure CSVs (into
`figures/data/`). After `decoder/prep.py`:

```bash
python decoder/score.py --tag paper --pte-tag paper_ptq --emit 10 \
    --int8-csv results/main/int8_ptq_results.csv --csv results/main/main_results.csv \
    --ensemble-csv results/main/ensemble_results.csv --latex figures/tex
python decoder/deploy_sweep.py --electrodes 62 --calib 1:3.84 --calib 5:3.86 \
    --calib 10:4.08 --calib 25:4.12 --calib 50:4.44 \
    --csv results/board/deploy_board.csv --latex figures/tex
python board/score_board.py --results results/board
python board/analyze_paced.py --results results/board
python board/write_board_numbers.py --results results/board
python figures/paper_tables.py
python figures/export_figure_data.py
```

`git diff --stat figures/tex figures/data` should then show nothing changed.
The CSVs in `results/` can change in the last digits on another machine
(floating point), the rounded numbers do not. The `--calib` values are the board's median call times for subject 1 (from
`results/board/sweep.log`). Figures 3 to 10 are drawn by MATLAB (R2019b or
later, or GNU Octave 8) from `figures/data/`: run `make_all_figures` (and
`make_ieee_figures` for the 4-page version) in `figures/matlab/`; the PDFs
land in `figures/out/`. Figures 1 and 2 are diagrams.

## What is where

```
decoder/      prep.py           filter-bank features, causal and online (notch, 7 bands, envelopes)
              protocol.py       the split, normalisation and the official metric, shared by everything
              model.py          the causal TCN (ECoGFlexNet)
              train.py          cross-subject pretraining + per-subject fine-tune
              score.py          the number that goes in the paper, plus the evaluation-protocol ladder
              ablate.py         leave-one-out ablation with paired statistics
              export_int8.py    PTQ/QAT INT8 export to ExecuTorch (XNNPACK) and host-side INT8 scoring
              score_gate_leak.py  the look-ahead leak of a time-pooling (ECA) gate, scored three ways
              deploy_sweep.py   labels-per-call trade-off: compute, latency, MAC count
checkpoints/  paper_<sub>_s<seed>.pt          the 15 FP32 models in the paper
              paper_ptq_<sub>_s<seed>_int8.pte  the 15 deployed INT8 models
              paper_ptq_e<E>_<sub>_s0_int8.pte  seed 0 exported at E = 1, 5, 25, 50 (latency sweep)
board/        benchmark that runs on the board, and the host-side scripts that score it
figures/      paper_tables.py, export_figure_data.py, matlab/, and their outputs (tex/, data/)
results/      every raw result the paper reports (see results/README.md)
scripts/      get_data.py
tests/        pytest suite
```

## Citation

If you use this code, please cite the paper (reference to follow) and the data
set:

- G. Schalk, J. Kubánek, K. J. Miller et al., "Decoding two-dimensional
  movement trajectories using electrocorticographic signals in humans,"
  *J. Neural Eng.* 4(3):264-275, 2007.
- K. J. Miller and G. Schalk, "Prediction of finger flexion: 4th
  Brain-Computer Interface data competition," BCI Competition IV, data set 4
  description, 2008.

## License

MIT, see [LICENSE](LICENSE). The data set has its own terms (above).
