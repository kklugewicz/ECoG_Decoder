function fig08_realtime(datadir, outdir)
%FIG08_REALTIME  Real-time replay on the board (E = 10, 100 Hz input).
%   (a) Label age: time from the arrival of the oldest sample in a 10-sample
%       batch to the emission of the batch's labels (90 ms of it is waiting
%       for the batch to fill). Plotted as the fraction of batches older than
%       x (log scale) so the tail is visible. 100 ms = budget.
%       Two pacing loops: sleep-based and busy-wait (spin).
%   (b) SoC temperature during the 10-minute soak test.
st = fig_style();
E = dlmread(fullfile(datadir, 'fig08a_label_age_exceedance.csv'), ',', 1, 0);
% columns: label_age_ms p_exceed_sleep p_exceed_spin (0 = blank = none older)
R = dlmread(fullfile(datadir, 'fig08a_label_age_raw.csv'), ',', 1, 0);
T = dlmread(fullfile(datadir, 'fig08b_soak_thermal.csv'), ',', 1, 0);
% columns: time_s soc_temp_c cpu_mhz

W = st.full; H = 6.8;
fig = new_figure(W, H);

% ---- (a) exceedance of label age
ax = new_axes(fig, [1.6 1.3 7.4 H - 2.2], st);
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
    % the oldest label sits where the curve ends (fraction 1/n)
    plot(ax, min(vmax, 149.5), 1 / size(rows, 1), 'o', 'MarkerSize', st.ms, ...
         'MarkerEdgeColor', cols{m}, 'MarkerFaceColor', 'w', 'LineWidth', 0.8);
end
set(ax, 'XLim', [93 150], 'YLim', [1e-4 1.5], 'YTick', [1e-4 1e-3 1e-2 1e-1 1], ...
        'YTickLabel', {'0.01%', '0.1%', '1%', '10%', '100%'});
plot(ax, [100 100], [1e-4 1.5], '--', 'Color', st.ink2, 'LineWidth', 0.8);
text(100.8, 0.6, '100 ms budget', 'Parent', ax, 'FontName', st.font, ...
     'FontSize', st.fsSmall, 'Color', st.ink2, 'VerticalAlignment', 'middle');
n1 = sum(R(:, 1) == 1); late1 = sum(R(R(:, 1) == 1, 4) > 100);
n2 = sum(R(:, 1) == 2); late2 = sum(R(R(:, 1) == 2, 4) > 100);
text_lines(ax, 149, 0.05, {'Batches older than 100 ms:', ...
                           sprintf('sleep-paced %d of %s', late1, commas(n1)), ...
                           sprintf('spin-paced %d of %s', late2, commas(n2))}, st, ...
           'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
if first
    olab = 'Oldest batch: first call after start-up';
else
    olab = 'Oldest batch of each run';
end
text(149, 2.6e-4, olab, 'Parent', ax, 'FontName', st.font, 'FontSize', st.fsSmall, ...
     'Color', st.ink2, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
xlabel(ax, 'Label age (ms)', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax, 'Fraction of batches older than x', 'FontSize', st.fs, 'Color', st.ink);

% ---- (b) soak temperature
ax2 = new_axes(fig, [10.6 1.3 W - 10.9 H - 2.2], st);
plot(ax2, T(:, 1) / 60, T(:, 2), '-', 'Color', st.ink2, 'LineWidth', 1.0);
set(ax2, 'XLim', [0 ceil(max(T(:, 1)) / 60)], 'YLim', [40 52]);
xlabel(ax2, 'Time (min)', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax2, ['SoC temperature (' st.deg 'C)'], 'FontSize', st.fs, 'Color', st.ink);
mhz = unique(T(:, 3));
if numel(mhz) == 1
    msg = {sprintf('CPU held at %.1f GHz', mhz / 1000), 'throughout (no throttling)'};
else
    msg = {sprintf('CPU clock %.1f to %.1f GHz', min(mhz) / 1000, max(mhz) / 1000)};
end
text_lines(ax2, 0.3, 51.4, msg, st, 'Color', st.ink2, 'VerticalAlignment', 'top');

ov = fig_overlay(fig, W, H);
panel_letter(ov, 0.15, H - 0.15, 'a', st);
panel_letter(ov, 9.6, H - 0.15, 'b', st);
yy = H - 0.42;
x = legend_item(ov, 1.6, yy, 'line', st.blue, 'Sleep-paced loop', st);
legend_item(ov, x, yy, 'line', st.orange, 'Spin-paced loop', st);
save_figure(fig, outdir, 'fig08_realtime');
end

function s = commas(n)
s = sprintf('%d', n);
k = numel(s) - 3;
while k > 0
    s = [s(1:k) ',' s(k + 1:end)];
    k = k - 3;
end
end
