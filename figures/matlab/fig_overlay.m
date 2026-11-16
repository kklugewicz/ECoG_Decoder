function ov = fig_overlay(fig, W, H)
%FIG_OVERLAY  Invisible full-figure axes in cm, for legends and panel
%   letters placed at exact positions (behaves the same in MATLAB and Octave).
ov = axes('Parent', fig, 'Units', 'centimeters', 'Position', [0 0 W H], ...
          'XLim', [0 W], 'YLim', [0 H], 'Visible', 'off');
hold(ov, 'on');
end
