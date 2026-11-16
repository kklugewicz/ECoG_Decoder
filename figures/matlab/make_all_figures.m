function make_all_figures(datadir, outdir)
%MAKE_ALL_FIGURES  Build every data figure of the paper from the CSV files.
%
%   1. Refresh the data (only needed after new runs), from the repo root:
%        python figures/export_figure_data.py
%   2. In MATLAB, from this folder:
%        make_all_figures
%      PDFs (vector) and 300-dpi PNGs land in figures/out.
%
%   Figures 1 (system overview) and 2 (architecture) are diagrams and are
%   not generated from data. Restyle every figure at once in fig_style.m.
%   Runs in MATLAB R2019b+ and in GNU Octave 8+.
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(datadir); datadir = fullfile(here, '..', 'data'); end
if nargin < 2 || isempty(outdir);  outdir  = fullfile(here, '..', 'out'); end
if ~exist(outdir, 'dir'); mkdir(outdir); end
addpath(here);

figs = {@fig03_traces, @fig04_accuracy_vs_memory, @fig05_ablation, @fig06_gate_leak, ...
        @fig07_quant_budget, @fig08_realtime, @fig09_batching, @fig10_val_vs_test};
for k = 1:numel(figs)
    fprintf('%s\n', func2str(figs{k}));
    figs{k}(datadir, outdir);
end
fprintf('Done: %d figures in %s\n', numel(figs), outdir);
end
