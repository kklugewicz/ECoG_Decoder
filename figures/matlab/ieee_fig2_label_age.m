function ieee_fig2_label_age(datadir, outdir)
%IEEE_FIG2_LABEL_AGE  Fig. 2 of the 4-page IEEE paper (column width).
%   Real-time replay on the board (E = 10, 100 Hz input). Fraction of label
%   batches older than x (log scale), for a sleep-paced and a busy-wait
%   (spin) pacing loop. Label age runs from the arrival of a batch's oldest
%   sample to the emission of its labels; 90 ms of it is waiting for the
%   batch to fill. Same data as panel (a) of fig08_realtime.
st = fig_style();
E = dlmread(fullfile(datadir, 'fig08a_label_age_exceedance.csv'), ',', 1, 0);
% columns: label_age_ms p_exceed_sleep p_exceed_spin
R = dlmread(fullfile(datadir, 'fig08a_label_age_raw.csv'), ',', 1, 0);
% columns: mode (1 sleep, 2 spin) subject batch label_age_ms ...

W = 8.8; H = 5.4;
fig = new_figure(W, H);
ax = new_axes(fig, [1.75 1.2 W - 2.0 H - 1.85], st);
set(ax, 'YScale', 'log');
cols = {st.blue, st.orange};
first = true;
for m = 1:2
    p = E(:, 1 + m);
    ok = p > 0;
    plot(ax, E(ok, 1), p(ok), '-', 'Color', cols{m}, 'LineWidth', 1.1);
    rows = R(R(:, 1) == m, :);
    [vmax, i] = max(rows(:, 4));
    first = first && rows(i, 3) == 0;      % was the oldest label the first call?
    plot(ax, min(vmax, 149.5), 1 / size(rows, 1), 'o', 'MarkerSize', st.ms, ...
         'MarkerEdgeColor', cols{m}, 'MarkerFaceColor', 'w', 'LineWidth', 0.8);
end
set(ax, 'XLim', [93 150], 'YLim', [1e-4 1.5], 'YTick', [1e-4 1e-3 1e-2 1e-1 1], ...
        'YTickLabel', {'0.01%', '0.1%', '1%', '10%', '100%'});
plot(ax, [100 100], [1e-4 1.5], '--', 'Color', st.ink2, 'LineWidth', 0.8);
text(101, 0.6, '100 ms budget', 'Parent', ax, 'FontName', st.font, ...
     'FontSize', st.fsSmall, 'Color', st.ink2, 'VerticalAlignment', 'middle');
n1 = sum(R(:, 1) == 1); late1 = sum(R(R(:, 1) == 1, 4) > 100);
n2 = sum(R(:, 1) == 2); late2 = sum(R(R(:, 1) == 2, 4) > 100);
text_lines(ax, 149, 0.04, {'Batches older than 100 ms:', ...
                           sprintf('sleep-paced %d of %s', late1, commas(n1)), ...
                           sprintf('spin-paced %d of %s', late2, commas(n2))}, st, ...
           'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
if first
    olab = 'Oldest: first call after start-up';
else
    olab = 'Oldest batch of each run';
end
text(149, 2.6e-4, olab, 'Parent', ax, 'FontName', st.font, 'FontSize', st.fsSmall, ...
     'Color', st.ink2, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
xlabel(ax, 'Label age (ms)', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax, 'Batches older than x', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
yy = H - 0.3;
x = legend_item(ov, 1.75, yy, 'line', st.blue, 'Sleep-paced loop', st);
legend_item(ov, x, yy, 'line', st.orange, 'Spin-paced loop', st);
save_figure(fig, outdir, 'ieee_fig2_label_age');
end

function s = commas(n)
s = sprintf('%d', n);
k = numel(s) - 3;
while k > 0
    s = [s(1:k) ',' s(k + 1:end)];
    k = k - 3;
end
end
