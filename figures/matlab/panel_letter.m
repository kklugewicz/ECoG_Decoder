function panel_letter(ov, x, y, letter, st)
%PANEL_LETTER  Bold panel label ('a', 'b', ...) at (x, y) cm on the overlay.
text(x, y, letter, 'Parent', ov, 'FontName', st.font, 'FontSize', st.fs + 2, ...
     'FontWeight', 'bold', 'Color', st.ink, 'VerticalAlignment', 'top');
end
