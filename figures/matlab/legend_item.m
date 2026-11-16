function x_end = legend_item(ov, x, y, kind, color, label, st, varargin)
%LEGEND_ITEM  One legend entry drawn on the overlay at (x, y) cm.
%   kind: 'line' | 'marker' | 'open' (hollow marker) | 'dash'
%   Optional: marker symbol (default 'o'). Returns the x where the text ends
%   (approximate), so entries can be chained left to right.
mk = 'o';
if ~isempty(varargin); mk = varargin{1}; end
switch kind
    case 'line'
        plot(ov, [x x + 0.5], [y y], '-', 'Color', color, 'LineWidth', 1.2);
        tx = x + 0.6;
    case 'dash'
        plot(ov, [x x + 0.5], [y y], '--', 'Color', color, 'LineWidth', 0.8);
        tx = x + 0.6;
    case 'marker'
        plot(ov, x + 0.12, y, mk, 'MarkerFaceColor', color, 'MarkerEdgeColor', color, ...
             'MarkerSize', st.ms + 1);
        tx = x + 0.3;
    case 'open'
        plot(ov, x + 0.12, y, mk, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', color, ...
             'MarkerSize', st.ms + 1, 'LineWidth', 0.8);
        tx = x + 0.3;
end
text(tx, y, label, 'Parent', ov, 'FontName', st.font, 'FontSize', st.fsSmall, ...
     'Color', st.ink, 'VerticalAlignment', 'middle');
x_end = tx + 0.13 * numel(label) + 0.35;   % ~0.13 cm per character at 7 pt
end
