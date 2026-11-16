function fig04_accuracy_vs_memory(datadir, outdir)
%FIG04_ACCURACY_VS_MEMORY  Mean r on ds4 against model memory, log scale.
%   Filled markers: measured on embedded hardware. Open: offline only.
%   Prior values are as reported by their authors; protocols differ
%   (fingers averaged, scoring rate, split), so read positions as approximate.
st = fig_style();
S = read_csv_struct(fullfile(datadir, 'fig04_accuracy_vs_memory.csv'));
keep = ~isnan(S.memory_kB);

% short labels and placement: {text, x factor (log x), dy in r, alignment}
lab = containers.Map();
lab('This work; INT8 on board')          = {'This work, INT8 on board', 0.67, 0.026, 'left'};
lab('This work; FP32')                   = {'This work, FP32', 1.2, -0.03, 'left'};
lab('This work; 5-seed INT8 ensemble')   = {'5-seed INT8 ensemble (board)', 1.25, 0.0, 'left'};
lab('Lin et al. RF; deployed (STM32H7)') = {'Lin RF, on STM32H7', 1.25, 0.0, 'left'};
lab('Lin et al. CNN+LSTM')               = {'Lin CNN+LSTM', 1.25, 0.0, 'left'};
lab('Lin et al. LSTM')                   = {'Lin LSTM', 1.25, 0.0, 'left'};
lab('Lin et al. CNN')                    = {'Lin CNN', 1.25, 0.0, 'left'};
lab('Lin et al. ANN')                    = {'Lin ANN', 1.25, 0.0, 'left'};
lab('Lin et al. SVR')                    = {'Lin SVR', 0.8, 0.0, 'right'};
lab('Lin et al. PLS')                    = {'Lin PLS', 1.25, 0.0, 'left'};
lab('FingerFlex (U-Net)')                = {'FingerFlex', 1.25, -0.012, 'left'};
lab('DTCNet')                            = {'DTCNet', 1.25, 0.012, 'left'};

W = st.full; H = 8.2;
fig = new_figure(W, H);
ax = new_axes(fig, [1.5 1.35 W - 1.8 H - 2.45], st);
set(ax, 'XScale', 'log', 'XLim', [30 2e6], 'YLim', [0.2 0.75]);
set(ax, 'XTick', [1e2 1e3 1e4 1e5 1e6], ...
        'XTickLabel', {'100 kB', '1 MB', '10 MB', '100 MB', '1 GB'});
set(ax, 'YGrid', 'on');
try; set(ax, 'GridColor', st.gridc, 'GridAlpha', 1); catch; end

idx = find(keep)';
for i = idx
    name = S.method{i};
    x = S.memory_kB(i); y = S.r_mean(i);
    onboard = S.measured_on_device(i) == 1;
    if strncmp(name, 'This work', 9)
        c = st.blue; mk = 'o';
        if ~isempty(strfind(name, 'ensemble')); mk = 's'; end
    elseif strncmp(name, 'Lin et al.', 10)
        c = st.orange; mk = 'd';
    else
        c = st.muted; mk = '^';
    end
    if strcmp(name, 'DTCNet')       % 550k-790k parameters, FP32
        plot(ax, [2200 3160], [y y], '-', 'Color', c, 'LineWidth', 0.8);
    end
    if onboard
        plot(ax, x, y, mk, 'MarkerFaceColor', c, 'MarkerEdgeColor', c, 'MarkerSize', st.ms + 2);
    else
        plot(ax, x, y, mk, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c, ...
             'MarkerSize', st.ms + 2, 'LineWidth', 0.9);
    end
    if isKey(lab, name)
        L = lab(name);
        text(x * L{2}, y + L{3}, L{1}, 'Parent', ax, 'FontName', st.font, ...
             'FontSize', st.fsSmall, 'Color', st.ink, 'HorizontalAlignment', L{4}, ...
             'VerticalAlignment', 'middle');
    end
end
xlabel(ax, 'Model memory (deployed file or parameters x 4 bytes)', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax, 'Mean r on BCI IV ds4', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
yl = H - 0.35;
x = legend_item(ov, 1.5, yl, 'marker', st.blue, 'This work', st, 'o');
x = legend_item(ov, x, yl, 'marker', st.orange, 'Lin et al. 2025', st, 'd');
x = legend_item(ov, x, yl, 'marker', st.muted, 'Other ds4 decoders', st, '^');
x = legend_item(ov, x + 0.6, yl, 'marker', st.ink2, 'Measured on hardware', st, 'o');
legend_item(ov, x, yl, 'open', st.ink2, 'Offline only', st, 'o');
save_figure(fig, outdir, 'fig04_accuracy_vs_memory');
end
