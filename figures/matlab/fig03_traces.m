function fig03_traces(datadir, outdir)
%FIG03_TRACES  Decoded (board, INT8) vs data-glove flexion, four scored fingers.
%   Median subject and median seed; 30-s window picked by a fixed rule in
%   export_figure_data.py. Both traces z-scored over the whole test
%   recording for display (Pearson r is unaffected by scaling).
st = fig_style();
D = dlmread(fullfile(datadir, 'fig03_traces.csv'), ',', 1, 0);
R = read_csv_struct(fullfile(datadir, 'fig03_finger_r.csv'));
t = D(:, 1);
names = {'Thumb', 'Index', 'Middle', 'Little'};

W = st.full; H = 10.5;
fig = new_figure(W, H);
left = 1.5; right = 0.3; bottom = 1.15; top = 0.75; gap = 0.25;
ph = (H - bottom - top - 3 * gap) / 4;
for k = 1:4
    y0 = H - top - k * ph - (k - 1) * gap;
    ax = new_axes(fig, [left y0 W - left - right ph], st);
    g = D(:, 1 + k); p = D(:, 5 + k);
    plot(ax, t, g, '-', 'Color', st.ink2, 'LineWidth', 1.1);
    plot(ax, t, p, '-', 'Color', st.blue, 'LineWidth', 0.9);
    lo = min([g; p]); hi = max([g; p]); pad = 0.08 * (hi - lo);
    set(ax, 'XLim', [t(1) t(end)], 'YLim', [lo - pad hi + pad]);
    ylabel(ax, names{k}, 'FontSize', st.fs, 'Color', st.ink);
    text(t(end), hi + pad, sprintf('r = %.2f ', R.r_whole_recording(k)), ...
         'Parent', ax, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
         'FontName', st.font, 'FontSize', st.fsSmall, 'Color', st.ink2);
    if k < 4
        set(ax, 'XTickLabel', []);
    else
        xlabel(ax, 'Time in test recording (s)', 'FontSize', st.fs, 'Color', st.ink);
    end
end
ov = fig_overlay(fig, W, H);
x = legend_item(ov, left, H - 0.35, 'line', st.ink2, 'Data glove', st);
legend_item(ov, x, H - 0.35, 'line', st.blue, 'Decoded on board (INT8)', st);
save_figure(fig, outdir, 'fig03_traces');
end
