function make_ieee_figures(datadir, outdir)
%MAKE_IEEE_FIGURES  Build the three column-width figures of the 4-page IEEE
%   version of the paper.
%
%   In MATLAB, from this folder:
%       make_ieee_figures
%   PDFs (vector) and 300-dpi PNGs land in figures/out. Each figure is
%   8.8 cm wide, one IEEE column, with 8 pt text. Restyle all of them in
%   fig_style.m.
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(datadir); datadir = fullfile(here, '..', 'data'); end
if nargin < 2 || isempty(outdir);  outdir  = fullfile(here, '..', 'out'); end
if ~exist(outdir, 'dir'); mkdir(outdir); end
addpath(here);

figs = {@ieee_fig1_accuracy_memory, @ieee_fig2_label_age, @ieee_fig3_gate_leak};
for k = 1:numel(figs)
    fprintf('%s\n', func2str(figs{k}));
    figs{k}(datadir, outdir);
end
fprintf('Done: %d figures in %s\n', numel(figs), outdir);
end
