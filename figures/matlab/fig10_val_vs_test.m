function fig10_val_vs_test(datadir, outdir)
%FIG10_VAL_VS_TEST  Does the validation split rank models the way test does?
%   One panel per subject; each dot is one trained model (13 configurations:
%   the reference and 12 ablations, x 5 seeds, 40 epochs). rho: Spearman
%   rank correlation within subject.
%   Open circles: the reference configuration.
st = fig_style();
D = dlmread(fullfile(datadir, 'fig10_val_vs_test.csv'), ',', 1, 0);
% columns: subject config_id seed val_r test_r

W = st.full; H = 6.0;
fig = new_figure(W, H);
left = 1.5; gap = 0.9; right = 0.3; bottom = 1.25; top = 0.75;
pw = (W - left - right - 2 * gap) / 3;
for s = 1:3
    r = D(D(:, 1) == s, :);
    ax = new_axes(fig, [left + (s - 1) * (pw + gap) bottom pw H - bottom - top], st);
    other = r(:, 2) ~= 0;
    plot(ax, r(other, 4), r(other, 5), 'o', 'MarkerSize', st.ms - 1, ...
         'MarkerFaceColor', st.sub(s, :), 'MarkerEdgeColor', st.sub(s, :));
    plot(ax, r(~other, 4), r(~other, 5), 'o', 'MarkerSize', st.ms + 1, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', st.ink, 'LineWidth', 0.8);
    xr = [min(r(:, 4)) max(r(:, 4))]; yr = [min(r(:, 5)) max(r(:, 5))];
    px = 0.08 * diff(xr); py = 0.08 * diff(yr);
    set(ax, 'XLim', [xr(1) - px xr(2) + px], 'YLim', [yr(1) - py yr(2) + 3 * py]);
    rho = spearman(r(:, 4), r(:, 5));
    text(xr(1) - px * 0.5, yr(2) + 2.6 * py, sprintf('Subject %d   \\rho = %+.2f', s, rho), ...
         'Parent', ax, 'FontName', st.font, 'FontSize', st.fsSmall, 'Color', st.ink, ...
         'VerticalAlignment', 'top');
    xlabel(ax, 'Validation r', 'FontSize', st.fs, 'Color', st.ink);
    if s == 1
        ylabel(ax, 'Test r', 'FontSize', st.fs, 'Color', st.ink);
    end
end
ov = fig_overlay(fig, W, H);
yy = H - 0.35;
nconf = numel(unique(D(:, 2))); nseed = numel(unique(D(:, 3)));
x = legend_item(ov, left, yy, 'marker', st.ink2, ...
                sprintf('One trained model (%d configurations x %d seeds)', nconf, nseed), st);
legend_item(ov, x, yy, 'open', st.ink, 'Reference configuration', st);
save_figure(fig, outdir, 'fig10_val_vs_test');
end

function rho = spearman(x, y)
% Spearman correlation with average ranks for ties (no toolbox needed).
rho = corrcoef_safe(avg_rank(x), avg_rank(y));
end

function r = avg_rank(v)
v = v(:);
[s, i] = sort(v);
r = zeros(size(v));
k = 1;
n = numel(v);
while k <= n
    j = k;
    while j < n && s(j + 1) == s(k)
        j = j + 1;
    end
    r(i(k:j)) = (k + j) / 2;
    k = j + 1;
end
end

function c = corrcoef_safe(a, b)
a = a - mean(a); b = b - mean(b);
c = sum(a .* b) / sqrt(sum(a .^ 2) * sum(b .^ 2));
end
