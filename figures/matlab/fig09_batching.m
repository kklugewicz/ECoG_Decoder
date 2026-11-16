function fig09_batching(datadir, outdir)
%FIG09_BATCHING  How the emission batch E trades compute for label latency.
%   (a) Board call time for one inference call against E (log x). Filled:
%       median, open: 99th percentile, coloured by subject.
%   (b) Compute per label against worst-case label latency, one point per E
%       (log-log). Worst case = (E - 1) x 10 ms of waiting + one call.
%       Dashed: the cached streaming form, which needs no batching.
st = fig_style();
A = dlmread(fullfile(datadir, 'fig09a_call_time_vs_E.csv'), ',', 1, 0);
% columns: emit_E subject median_ms p95_ms p99_ms
P = dlmread(fullfile(datadir, 'fig09b_pareto.csv'), ',', 1, 0);
% columns: emit_E window kmac_per_label mmac_per_s max_latency_ms in_100ms_budget
kmacCached = 24.4;          % numbers.tex \KMACcached

W = st.full; H = 7.0;
fig = new_figure(W, H);

% ---- (a) call time vs E
ax = new_axes(fig, [1.5 1.3 6.0 H - 2.2], st);
set(ax, 'XScale', 'log');
off = [0.93 1 1.07];        % small horizontal offsets so subjects do not overlap
for s = 1:3
    r = A(A(:, 2) == s, :);
    plot(ax, r(:, 1) * off(s), r(:, 3), '-o', 'Color', st.sub(s, :), 'LineWidth', 0.9, ...
         'MarkerSize', st.ms, 'MarkerFaceColor', st.sub(s, :));
    plot(ax, r(:, 1) * off(s), r(:, 5), 'o', 'MarkerSize', st.ms, ...
         'MarkerEdgeColor', st.sub(s, :), 'MarkerFaceColor', 'w', 'LineWidth', 0.6);
end
set(ax, 'XLim', [0.8 65], 'XTick', [1 5 10 25 50], 'XTickLabel', {'1', '5', '10', '25', '50'}, ...
        'YLim', [0 10]);
xlabel(ax, 'Emission batch E (labels per call)', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax, 'Call time on the board (ms)', 'FontSize', st.fs, 'Color', st.ink);

% ---- (b) compute vs worst-case latency
ax2 = new_axes(fig, [9.4 1.3 W - 9.7 H - 2.2], st);
set(ax2, 'XScale', 'log', 'YScale', 'log');
plot(ax2, [100 100], [10 2e4], '--', 'Color', st.ink2, 'LineWidth', 0.8);
plot(ax2, [2 1e4], [kmacCached kmacCached], ':', 'Color', st.violet, 'LineWidth', 1.0);
text(9000, kmacCached * 1.12, sprintf('Cached streaming form: %.1f kMAC per label', kmacCached), ...
     'Parent', ax2, 'FontName', st.font, 'FontSize', st.fsSmall, 'Color', st.ink2, ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
plot(ax2, P(:, 5), P(:, 3), '-', 'Color', st.light, 'LineWidth', 0.8);
for i = 1:size(P, 1)
    inb = P(i, 6) == 1;
    if P(i, 1) == 10
        plot(ax2, P(i, 5), P(i, 3), 'o', 'MarkerSize', st.ms + 3, 'MarkerFaceColor', st.blue, ...
             'MarkerEdgeColor', st.blue);
    elseif inb
        plot(ax2, P(i, 5), P(i, 3), 'o', 'MarkerSize', st.ms + 1, 'MarkerFaceColor', st.ink2, ...
             'MarkerEdgeColor', st.ink2);
    else
        plot(ax2, P(i, 5), P(i, 3), 'o', 'MarkerSize', st.ms + 1, 'MarkerFaceColor', 'w', ...
             'MarkerEdgeColor', st.ink2, 'LineWidth', 0.8);
    end
    lbl = sprintf('E = %d', P(i, 1));
    if P(i, 1) == 10; lbl = 'E = 10 (used)'; end
    text(P(i, 5) * 1.18, P(i, 3) * 1.12, lbl, 'Parent', ax2, 'FontName', st.font, ...
         'FontSize', st.fsSmall, 'Color', st.ink, 'VerticalAlignment', 'bottom');
end
text(100 * 1.12, 1.5e4, '100 ms budget', 'Parent', ax2, 'FontName', st.font, ...
     'FontSize', st.fsSmall, 'Color', st.ink2, 'VerticalAlignment', 'top');
set(ax2, 'XLim', [2 1e4], 'YLim', [10 2e4], 'XTick', [10 100 1000], ...
         'XTickLabel', {'10', '100', '1,000'}, 'YTick', [10 100 1000 10000], ...
         'YTickLabel', {'10', '100', '1,000', '10,000'});
xlabel(ax2, 'Worst-case label latency (ms)', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax2, 'Compute per label (kMAC)', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
panel_letter(ov, 0.15, H - 0.15, 'a', st);
panel_letter(ov, 8.3, H - 0.15, 'b', st);
yy = H - 0.42;
x = legend_item(ov, 1.5, yy, 'marker', st.ink2, 'Median', st);
x = legend_item(ov, x, yy, 'open', st.ink2, '99th percentile', st);
x = legend_item(ov, x, yy, 'marker', st.sub(1, :), 'S1', st);
x = legend_item(ov, x - 0.1, yy, 'marker', st.sub(2, :), 'S2', st);
legend_item(ov, x - 0.1, yy, 'marker', st.sub(3, :), 'S3', st);
x = legend_item(ov, 9.4, yy, 'marker', st.ink2, 'Within budget', st);
legend_item(ov, x, yy, 'open', st.ink2, 'Over budget', st);
save_figure(fig, outdir, 'fig09_batching');
end
