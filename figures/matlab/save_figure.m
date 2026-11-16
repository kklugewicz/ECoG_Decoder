function save_figure(fig, outdir, name)
%SAVE_FIGURE  Vector PDF (for LaTeX) plus a 300-dpi PNG preview.
if ~exist(outdir, 'dir'); mkdir(outdir); end
pdf = fullfile(outdir, [name '.pdf']);
png = fullfile(outdir, [name '.png']);
if exist('OCTAVE_VERSION', 'builtin')
    print(fig, pdf, '-dpdf');
    print(fig, png, '-dpng', '-r300');
elseif exist('exportgraphics', 'file')          % MATLAB R2020a and later
    exportgraphics(fig, pdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(fig, png, 'Resolution', 300, 'BackgroundColor', 'white');
else                                             % older MATLAB
    print(fig, pdf, '-dpdf', '-painters');
    print(fig, png, '-dpng', '-r300');
end
close(fig);
fprintf('  wrote %s\n', pdf);
end
