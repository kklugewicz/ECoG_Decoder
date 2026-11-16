function text_lines(ax, x, y, lines, st, varargin)
%TEXT_LINES  Multi-line text in data coordinates, one text object per line.
%   Works the same in MATLAB and Octave (gnuplot cannot take cell-array text).
%   Set the axes limits and scale BEFORE calling: line spacing is computed
%   from them. Options: 'HorizontalAlignment' (left), 'VerticalAlignment'
%   (middle; top = first line at y, bottom = last line at y), 'FontSize'
%   (st.fsSmall), 'Color' (st.ink), 'FontWeight' (normal).
o = struct('HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
           'FontSize', st.fsSmall, 'Color', st.ink, 'FontWeight', 'normal');
for k = 1:2:numel(varargin)
    o.(varargin{k}) = varargin{k + 1};
end
if ischar(lines); lines = {lines}; end
n = numel(lines);
pos = get(ax, 'Position');                       % cm (new_axes sets the units)
yl = get(ax, 'YLim');
line_cm = o.FontSize * 1.2 / 72 * 2.54;
for k = 1:n
    switch o.VerticalAlignment
        case 'top';    off = -(k - 1) * line_cm;
        case 'bottom'; off = (n - k) * line_cm;
        otherwise;     off = ((n - 1) / 2 - (k - 1)) * line_cm;
    end
    if strcmp(get(ax, 'YScale'), 'log')
        yk = y * 10 ^ (off / pos(4) * log10(yl(2) / yl(1)));
    else
        yk = y + off / pos(4) * diff(yl);
    end
    text(x, yk, lines{k}, 'Parent', ax, 'FontName', st.font, 'FontSize', o.FontSize, ...
         'Color', o.Color, 'FontWeight', o.FontWeight, ...
         'HorizontalAlignment', o.HorizontalAlignment, ...
         'VerticalAlignment', o.VerticalAlignment);
end
end
