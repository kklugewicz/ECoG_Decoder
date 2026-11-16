function fig06_gate_leak(datadir, outdir)
%FIG06_GATE_LEAK  The look-ahead leak of a time-pooling (ECA) channel gate.
%   (a) Each line is one of 15 ECA models (subject x seed) scored three
%       ways: densely over 200-s chunks (the gate pools over the future),
%       over training-length windows, and as exported (causal).
%   (b) Paired change in r against the matched model with our pointwise
%       causal gate. Dots: single models. Black: mean and 95% CI.
st = fig_style();
D = dlmread(fullfile(datadir, 'fig06_gate_leak.csv'), ',', 1, 0);
% columns: subject seed causal_gate no_gate eca_dense eca_trainwin eca_exported
subj = D(:, 1); seed = D(:, 2);

W = st.full; H = 7.2;
fig = new_figure(W, H);

% ---- (a) slopegraph of absolute r
ax = new_axes(fig, [1.5 1.55 5.4 H - 2.45], st);
for i = 1:size(D, 1)
    plot(ax, 1:3, D(i, 5:7), '-o', 'Color', st.sub(subj(i), :), 'LineWidth', 0.8, ...
         'MarkerSize', st.ms - 1, 'MarkerFaceColor', st.sub(subj(i), :));
end
set(ax, 'XLim', [0.7 3.3], 'XTick', 1:3, 'XTickLabel', {'Dense', 'Train window', 'Exported'}, ...
        'YLim', [0.36 0.70]);
ylabel(ax, 'r on the test recording', 'FontSize', st.fs, 'Color', st.ink);
xlabel(ax, 'How the ECA model is scored', 'FontSize', st.fs, 'Color', st.ink);

% ---- (b) paired change vs the pointwise causal gate
ax2 = new_axes(fig, [8.9 1.55 W - 9.2 H - 2.45], st);
cols = [4 5 6 7];
names = {{'No', 'gate'}, {'ECA', 'dense'}, {'ECA', 'train window'}, {'ECA', 'exported'}};
set(ax2, 'XLim', [0.6 4.75], 'XTick', [], 'YLim', [-0.09 0.095]);
plot(ax2, [0.6 4.75], [0 0], '-', 'Color', st.muted, 'LineWidth', 0.7);
dy = [-0.16 -0.08 0 0.08 0.16];
for k = 1:4
    d = D(:, cols(k)) - D(:, 3);
    for i = 1:numel(d)
        plot(ax2, k - 0.2 + dy(seed(i) + 1) * 0.6, d(i), 'o', 'MarkerSize', st.ms - 1, ...
             'MarkerEdgeColor', st.sub(subj(i), :), 'MarkerFaceColor', 'w', 'LineWidth', 0.6);
    end
    [m, lo, hi] = mean_ci(d);
    plot(ax2, [k + 0.06 k + 0.06], [lo hi], '-', 'Color', st.ink, 'LineWidth', 1.0);
    plot(ax2, k + 0.06, m, 'o', 'MarkerSize', st.ms + 1, 'MarkerFaceColor', st.ink, ...
         'MarkerEdgeColor', st.ink);
    text(k + 0.16, m, sprintf('%+.3f', m), 'Parent', ax2, 'FontName', st.font, ...
         'FontSize', st.fsSmall, 'Color', st.ink, 'VerticalAlignment', 'middle');
    text_lines(ax2, k + 0.05, -0.093, names{k}, st, 'FontSize', st.fs, 'Color', st.ink2, ...
               'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
ylabel(ax2, 'Change in r vs pointwise causal gate', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
panel_letter(ov, 0.15, H - 0.15, 'a', st);
panel_letter(ov, 7.6, H - 0.15, 'b', st);
yl = H - 0.42;
x = legend_item(ov, 1.5, yl, 'marker', st.sub(1, :), 'Subject 1', st);
x = legend_item(ov, x, yl, 'marker', st.sub(2, :), 'Subject 2', st);
legend_item(ov, x, yl, 'marker', st.sub(3, :), 'Subject 3', st);
x = legend_item(ov, 8.9, yl, 'open', st.ink2, 'One model', st);
legend_item(ov, x, yl, 'marker', st.ink, 'Mean, 95% CI (n = 15)', st);
save_figure(fig, outdir, 'fig06_gate_leak');
end
