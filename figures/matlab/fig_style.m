function st = fig_style()
%FIG_STYLE  Shared look for every paper figure: sizes, fonts and colours.
%   Change a value here and re-run make_all_figures to restyle all figures.
%
%   Colours: subjects 1-3 always use the first three categorical slots
%   (blue, orange, aqua), which stay distinguishable under the common
%   colour-vision deficiencies. Neutral greys carry reference series.

st.font    = 'Arial';
st.fs      = 8;      % axis labels and tick labels (pt)
st.fsSmall = 7;      % annotations
st.lw      = 1.0;    % data lines (pt)
st.axlw    = 0.6;    % axis lines (pt)
st.ms      = 4;      % marker size (pt)

st.full = 16.0;      % full text width (cm)
st.half = 8.0;       % half width (cm)

st.ink    = hex2rgb('#0b0b0b');   % primary text and key marks
st.ink2   = hex2rgb('#52514e');   % secondary text, axes
st.muted  = hex2rgb('#8a8985');   % reference series
st.light  = hex2rgb('#c3c2b7');   % thin context lines
st.gridc  = hex2rgb('#e6e5e1');   % gridlines, shading

st.blue   = hex2rgb('#2a78d6');
st.orange = hex2rgb('#eb6834');
st.aqua   = hex2rgb('#1baf7a');
st.violet = hex2rgb('#6250d6');
st.red    = hex2rgb('#e34948');

st.sub = [st.blue; st.orange; st.aqua];   % subject 1, 2, 3

% degree sign: UTF-16 in MATLAB, UTF-8 bytes in Octave
if exist('OCTAVE_VERSION', 'builtin')
    st.deg = char([194 176]);
else
    st.deg = char(176);
end
end

function rgb = hex2rgb(h)
rgb = sscanf(h(2:end), '%2x%2x%2x')' / 255;
end
