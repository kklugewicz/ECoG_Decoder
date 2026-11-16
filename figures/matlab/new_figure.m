function fig = new_figure(width_cm, height_cm)
%NEW_FIGURE  Blank figure of an exact printed size (cm), white background.
fig = figure('Units', 'centimeters', 'Color', 'w', 'Visible', 'off');
pos = get(fig, 'Position');
set(fig, 'Position', [pos(1) pos(2) width_cm height_cm]);
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [width_cm height_cm], ...
         'PaperPosition', [0 0 width_cm height_cm], 'PaperPositionMode', 'manual');
set(fig, 'InvertHardcopy', 'off');
end
