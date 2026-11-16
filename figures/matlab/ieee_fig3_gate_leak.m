function ieee_fig3_gate_leak(datadir, outdir)
%IEEE_FIG3_GATE_LEAK  Fig. 3 of the 4-page IEEE paper (column width).
%   Paired change in r against the matched model with the pointwise causal
%   gate, for no gate and for a time-pooling (ECA) gate scored three ways:
%   densely over 200-s chunks (the gate pools over the future), over
%   training-length windows, and as exported (causal). Thin lines join each
%   ECA model's three scores, so the drop from dense to exported is visible
%   model by model. Black: mean and 95% CI over the 15 subject-seed models.
st = fig_style();
D = dlmread(fullfile(datadir, 'fig06_gate_leak.csv'), ',', 1, 0);
% columns: subject seed causal_gate no_gate eca_dense eca_trainwin eca_exported
subj = D(:, 1); seed = D(:, 2);
d = D(:, 4:7) - repmat(D(:, 3), 1, 4);          % vs pointwise causal gate
jit = [-0.16 -0.08 0 0.08 0.16] * 0.9;          % spread the five seeds
xs = repmat(1:4, size(D, 1), 1) + repmat(jit(seed + 1)', 1, 4) - 0.12;

W = 8.8; H = 5.5;
fig = new_figure(W, H);
ax = new_axes(fig, [1.8 1.0 W - 1.95 H - 1.65], st);
set(ax, 'XLim', [0.55 4.75], 'XTick', [], 'YLim', [-0.09 0.095]);
plot(ax, [0.55 4.75], [0 0], '-', 'Color', st.muted, 'LineWidth', 0.7);
for i = 1:size(D, 1)                             % one ECA model: dense -> window -> exported
    c = 0.55 * st.sub(subj(i), :) + 0.45;        % lighter shade of the subject colour
    plot(ax, xs(i, 2:4), d(i, 2:4), '-', 'Color', c, 'LineWidth', 0.5);
end
names = {{'No', 'gate'}, {'ECA,', 'dense'}, {'ECA, train', 'window'}, {'ECA,', 'exported'}};
for k = 1:4
    for i = 1:size(D, 1)
        plot(ax, xs(i, k), d(i, k), 'o', 'MarkerSize', st.ms - 1, ...
             'MarkerEdgeColor', st.sub(subj(i), :), 'MarkerFaceColor', 'w', 'LineWidth', 0.6);
    end
    [m, lo, hi] = mean_ci(d(:, k));
    plot(ax, [k + 0.2 k + 0.2], [lo hi], '-', 'Color', st.ink, 'LineWidth', 1.0);
    plot(ax, k + 0.2, m, 'o', 'MarkerSize', st.ms + 1, 'MarkerFaceColor', st.ink, ...
         'MarkerEdgeColor', st.ink);
    text(k + 0.29, m, sprintf('%+.3f', m), 'Parent', ax, 'FontName', st.font, ...
         'FontSize', st.fsSmall, 'Color', st.ink, 'VerticalAlignment', 'middle');
    text_lines(ax, k + 0.05, -0.094, names{k}, st, 'FontSize', st.fs, 'Color', st.ink2, ...
               'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
ylabel(ax, 'Change in r', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
yl = H - 0.3;
x = legend_item(ov, 1.8, yl, 'open', st.sub(1, :), 'S1', st);
x = legend_item(ov, x, yl, 'open', st.sub(2, :), 'S2', st);
x = legend_item(ov, x, yl, 'open', st.sub(3, :), 'S3 (one model each)', st);
legend_item(ov, x, yl, 'marker', st.ink, 'Mean, 95% CI', st);
save_figure(fig, outdir, 'ieee_fig3_gate_leak');
end
