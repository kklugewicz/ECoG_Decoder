function ax = new_axes(fig, pos_cm, st)
%NEW_AXES  Axes at an exact position in cm [left bottom width height],
%   styled once: thin grey axes, outward ticks, no box.
ax = axes('Parent', fig, 'Units', 'centimeters', 'Position', pos_cm);
hold(ax, 'on');
set(ax, 'FontName', st.font, 'FontSize', st.fs, 'LineWidth', st.axlw, ...
        'TickDir', 'out', 'Box', 'off', 'XColor', st.ink2, 'YColor', st.ink2, ...
        'Layer', 'top', 'Color', 'none');
try
    set(ax, 'TickLength', [0.012 0.012]);
catch
end
end
