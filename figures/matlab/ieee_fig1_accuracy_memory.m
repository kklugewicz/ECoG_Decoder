function ieee_fig1_accuracy_memory(datadir, outdir)
%IEEE_FIG1_ACCURACY_MEMORY  Fig. 1 of the 4-page IEEE paper (column width).
%   Mean r on ds4 against model memory, log scale. Same data as
%   fig04_accuracy_vs_memory, laid out for one 8.8 cm column.
%   Filled markers: measured on embedded hardware. Open: offline only.
st = fig_style();
S = read_csv_struct(fullfile(datadir, 'fig04_accuracy_vs_memory.csv'));
keep = ~isnan(S.memory_kB);

% label text and placement: {text, x factor (log x), dy in r, alignment}
lab = containers.Map();
lab('This work; INT8 on board')          = {'This work, INT8', 0.62, -0.034, 'left'};
lab('This work; FP32')                   = {'FP32', 1.22, 0.0, 'left'};
lab('This work; 5-seed INT8 ensemble')   = {'5-seed ensemble', 0.62, 0.034, 'left'};
lab('Lin et al. RF; deployed (STM32H7)') = {'Lin forest (MCU)', 1.25, 0.0, 'left'};
lab('Lin et al. CNN+LSTM')               = {'Lin CNN+LSTM', 1.25, 0.0, 'left'};
lab('Lin et al. LSTM')                   = {'Lin LSTM', 1.25, 0.0, 'left'};
lab('Lin et al. CNN')                    = {'Lin CNN', 1.25, 0.0, 'left'};
lab('Lin et al. ANN')                    = {'Lin ANN', 1.25, 0.0, 'left'};
lab('Lin et al. SVR')                    = {'Lin SVR', 0.8, 0.0, 'right'};
lab('Lin et al. PLS')                    = {'Lin PLS', 1.25, 0.0, 'left'};
lab('FingerFlex (U-Net)')                = {'FingerFlex', 1.3, -0.014, 'left'};
lab('DTCNet')                            = {'DTCNet', 1.3, 0.014, 'left'};

W = 8.8; H = 6.4;
fig = new_figure(W, H);
ax = new_axes(fig, [1.45 1.2 W - 1.75 H - 2.2], st);
set(ax, 'XScale', 'log', 'XLim', [30 6e5], 'YLim', [0.2 0.75]);
set(ax, 'XTick', [1e2 1e3 1e4 1e5], ...
        'XTickLabel', {'100 kB', '1 MB', '10 MB', '100 MB'});
set(ax, 'YTick', 0.2:0.1:0.7, 'YGrid', 'on');
try; set(ax, 'GridColor', st.gridc, 'GridAlpha', 1); catch; end

for i = find(keep)'
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
        plot(ax, x, y, mk, 'MarkerFaceColor', c, 'MarkerEdgeColor', c, 'MarkerSize', st.ms + 1);
    else
        plot(ax, x, y, mk, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c, ...
             'MarkerSize', st.ms + 1, 'LineWidth', 0.9);
    end
    if isKey(lab, name)
        L = lab(name);
        text(x * L{2}, y + L{3}, L{1}, 'Parent', ax, 'FontName', st.font, ...
             'FontSize', st.fsSmall, 'Color', st.ink, 'HorizontalAlignment', L{4}, ...
             'VerticalAlignment', 'middle');
    end
end
xlabel(ax, 'Model memory', 'FontSize', st.fs, 'Color', st.ink);
ylabel(ax, 'Mean r on ds4', 'FontSize', st.fs, 'Color', st.ink);

ov = fig_overlay(fig, W, H);
y1 = H - 0.3; y2 = H - 0.68;
x = legend_item(ov, 1.45, y1, 'marker', st.blue, 'This work', st, 'o');
x = legend_item(ov, x, y1, 'marker', st.orange, 'Lin et al. 2025', st, 'd');
legend_item(ov, x, y1, 'marker', st.muted, 'Other ds4 decoders', st, '^');
x = legend_item(ov, 1.45, y2, 'marker', st.ink2, 'Measured on hardware', st, 'o');
legend_item(ov, x, y2, 'open', st.ink2, 'Offline only', st, 'o');
save_figure(fig, outdir, 'ieee_fig1_accuracy_memory');
end
