function zero_line(ax, orient, value, st, style)
%ZERO_LINE  Reference line across the axes (works where xline/yline do not).
if nargin < 5; style = '-'; end
lim = [get(ax, 'XLim'); get(ax, 'YLim')];
if strcmp(orient, 'x')
    plot(ax, [value value], lim(2, :), style, 'Color', st.muted, 'LineWidth', 0.6);
else
    plot(ax, lim(1, :), [value value], style, 'Color', st.muted, 'LineWidth', 0.6);
end
end
