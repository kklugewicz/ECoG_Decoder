function fig05_ablation(datadir, outdir)
%FIG05_ABLATION  Leave-one-out ablation as a forest plot.
%   Dots: the 15 paired subject-seed differences at 40 epochs, coloured by
%   subject. Black circle and line: mean and 95% CI at 40 epochs. Open
%   diamond and grey line: the independent 80-epoch replication.
%   Right column: exact sign-flip p, Holm-corrected over the 12 rows.
st = fig_style();
S = read_csv_struct(fullfile(datadir, 'fig05_ablation_summary.csv'));
C = dlmread(fullfile(datadir, 'fig05_ablation_cells.csv'), ',', 1, 0);
n = numel(S.row_id);

W = st.full; H = 10.0;
fig = new_figure(W, H);
left = 6.0; right = 1.7; bottom = 1.2; top = 0.9;
ax = new_axes(fig, [left bottom W - left - right H - bottom - top], st);
xl = [-0.105 0.095];
set(ax, 'XLim', xl, 'YLim', [0.4 n + 0.6], 'YTick', 1:n, ...
        'YTickLabel', flipud(S.label(:)), 'TickLength', [0 0]);
set(ax, 'XTick', -0.10:0.05:0.10);

% alternate row shading
for j = 1:2:n
    patch([xl(1) xl(2) xl(2) xl(1)], [j - 0.5 j - 0.5 j + 0.5 j + 0.5], st.gridc, ...
          'EdgeColor', 'none', 'Parent', ax);
end
plot(ax, [0 0], [0.4 n + 0.6], '-', 'Color', st.muted, 'LineWidth', 0.7);

jit = [-0.10 -0.05 0 0.05 0.10];
for j = 1:n
    y = n + 1 - j;                          % row 1 at the top
    for s = 1:3
        sel = C(:, 1) == S.row_id(j) & C(:, 2) == 40 & C(:, 3) == s;
        d = C(sel, 5); sd = C(sel, 4);
        plot(ax, d, y + 0.18 + jit(sd + 1), 'o', 'MarkerSize', st.ms - 1, ...
             'MarkerEdgeColor', st.sub(s, :), 'MarkerFaceColor', 'w', 'LineWidth', 0.6);
    end
    plot(ax, [S.ci40_lo(j) S.ci40_hi(j)], [y + 0.18 y + 0.18], '-', 'Color', st.ink, 'LineWidth', 1.0);
    plot(ax, S.mean40(j), y + 0.18, 'o', 'MarkerSize', st.ms + 1, ...
         'MarkerFaceColor', st.ink, 'MarkerEdgeColor', st.ink);
    plot(ax, [S.ci80_lo(j) S.ci80_hi(j)], [y - 0.22 y - 0.22], '-', 'Color', st.muted, 'LineWidth', 1.0);
    plot(ax, S.mean80(j), y - 0.22, 'd', 'MarkerSize', st.ms + 1, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', st.ink2, 'LineWidth', 0.8);
    p = S.pholm40(j);
    if p < 0.001
        ptxt = '<0.001';
    else
        ptxt = sprintf('%.3f', p);
    end
    fw = 'normal'; if p < 0.05; fw = 'bold'; end
    text(xl(2) + 0.006, y, ptxt, 'Parent', ax, 'FontName', st.font, 'FontSize', st.fsSmall, ...
         'Color', st.ink, 'FontWeight', fw, 'VerticalAlignment', 'middle');
end
text(xl(2) + 0.006, n + 0.85, 'p (Holm)', 'Parent', ax, 'FontName', st.font, ...
     'FontSize', st.fsSmall, 'Color', st.ink2, 'VerticalAlignment', 'bottom');
xlabel(ax, 'Change in r when the component is reverted  (< 0: the component helps)', ...
       'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
yl = H - 0.35;
x = legend_item(ov, 0.3, yl, 'open', st.sub(1, :), 'S1', st);
x = legend_item(ov, x - 0.1, yl, 'open', st.sub(2, :), 'S2', st);
x = legend_item(ov, x - 0.1, yl, 'open', st.sub(3, :), 'S3: one model each (40 epochs)', st);
x = legend_item(ov, x, yl, 'marker', st.ink, 'Mean, 95% CI (40 epochs)', st);
legend_item(ov, x, yl, 'open', st.ink2, '80-epoch replication', st, 'd');
save_figure(fig, outdir, 'fig05_ablation');
end
