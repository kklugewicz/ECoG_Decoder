function fig07_quant_budget(datadir, outdir)
%FIG07_QUANT_BUDGET  What INT8 and longer training do to accuracy.
%   (a) Paired change in r against the same model in FP32: post-training
%       INT8 (run on the board) and quantisation-aware training (QAT).
%   (b) Paired change in r against the 40-epoch model at 80 and 120 epochs.
%   Dots: single subject-seed models, coloured by subject. Black: mean, 95% CI.
st = fig_style();
Q = dlmread(fullfile(datadir, 'fig07a_quantisation.csv'), ',', 1, 0);
% columns: subject seed fp32 int8_ptq int8_qat int8_ptq_board
B = dlmread(fullfile(datadir, 'fig07b_training_budget.csv'), ',', 1, 0);
% columns: subject seed epochs_40 epochs_80 epochs_120

W = st.full; H = 7.0;
fig = new_figure(W, H);
yl = [-0.085 0.04];

% ---- (a) quantisation
ax = new_axes(fig, [1.6 1.55 5.6 H - 2.45], st);
dq = {Q(:, 6) - Q(:, 3), Q(:, 5) - Q(:, 3)};
na = {{'INT8 PTQ', 'on the board'}, {'INT8 QAT'}};
set(ax, 'XLim', [0.5 2.6]);
strip(ax, dq, Q(:, 1), Q(:, 2), na, yl, st);
ylabel(ax, 'Change in r vs FP32', 'FontSize', st.fs, 'Color', st.ink);
same = max(abs(Q(:, 6) - Q(:, 4)));
if same == 0
    note = {'Board output equals', 'host PTQ for all 15'};
else
    note = {sprintf('Board vs host PTQ:'), sprintf('max |diff| %.1e', same)};
end
text_lines(ax, 1.04, -0.018, note, st, 'Color', st.ink2, ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');

% ---- (b) training budget
ax2 = new_axes(fig, [9.6 1.55 W - 9.9 H - 2.45], st);
db = {B(:, 4) - B(:, 3), B(:, 5) - B(:, 3)};
nb = {{'80 epochs'}, {'120 epochs'}};
set(ax2, 'XLim', [0.5 2.6]);
strip(ax2, db, B(:, 1), B(:, 2), nb, yl, st);
ylabel(ax2, 'Change in r vs 40 epochs', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
panel_letter(ov, 0.15, H - 0.15, 'a', st);
panel_letter(ov, 8.3, H - 0.15, 'b', st);
yy = H - 0.42;
x = legend_item(ov, 1.6, yy, 'open', st.sub(1, :), 'S1', st);
x = legend_item(ov, x - 0.1, yy, 'open', st.sub(2, :), 'S2', st);
x = legend_item(ov, x - 0.1, yy, 'open', st.sub(3, :), 'S3: one model each', st);
legend_item(ov, 9.6, yy, 'marker', st.ink, 'Mean, 95% CI (n = 15)', st);
save_figure(fig, outdir, 'fig07_quant_budget');
end

function strip(ax, dlist, subj, seed, names, yl, st)
% Dot strip per condition with mean and CI to the right, zero line, labels below.
set(ax, 'XTick', [], 'YLim', yl);
plot(ax, [0.5 numel(dlist) + 0.6], [0 0], '-', 'Color', st.muted, 'LineWidth', 0.7);
dy = [-0.16 -0.08 0 0.08 0.16];
for k = 1:numel(dlist)
    d = dlist{k};
    for i = 1:numel(d)
        plot(ax, k - 0.12 + dy(seed(i) + 1) * 0.6, d(i), 'o', 'MarkerSize', st.ms - 1, ...
             'MarkerEdgeColor', st.sub(subj(i), :), 'MarkerFaceColor', 'w', 'LineWidth', 0.6);
    end
    [m, lo, hi] = mean_ci(d);
    plot(ax, [k + 0.2 k + 0.2], [lo hi], '-', 'Color', st.ink, 'LineWidth', 1.0);
    plot(ax, k + 0.2, m, 'o', 'MarkerSize', st.ms + 1, 'MarkerFaceColor', st.ink, ...
         'MarkerEdgeColor', st.ink);
    text(k + 0.32, m, sprintf('%+.3f', m), 'Parent', ax, 'FontName', st.font, ...
         'FontSize', st.fsSmall, 'Color', st.ink, 'VerticalAlignment', 'middle');
    text_lines(ax, k + 0.1, yl(1) - 0.002, names{k}, st, 'FontSize', st.fs, 'Color', st.ink2, ...
               'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
end
end
