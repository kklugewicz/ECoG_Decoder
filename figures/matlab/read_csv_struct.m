function S = read_csv_struct(path)
%READ_CSV_STRUCT  Read a CSV with a header row into a struct of columns.
%   Numeric columns become double vectors (empty cells -> NaN); any column
%   with text becomes a cell array of strings. Text fields must not contain
%   commas (export_figure_data.py guarantees this). Works in MATLAB and Octave.
txt = fileread(path);
lines = regexp(txt, '\r?\n', 'split');
lines = lines(~cellfun(@(s) isempty(strtrim(s)), lines));
hdr = strsplit(strtrim(lines{1}), ',', 'CollapseDelimiters', false);
n = numel(lines) - 1;
raw = cell(n, numel(hdr));
for i = 1:n
    parts = strsplit(strtrim(lines{i + 1}), ',', 'CollapseDelimiters', false);
    parts(end + 1:numel(hdr)) = {''};
    raw(i, :) = parts(1:numel(hdr));
end
S = struct();
for j = 1:numel(hdr)
    name = regexprep(hdr{j}, '[^A-Za-z0-9_]', '_');
    if ~isletter(name(1)); name = ['c_' name]; end
    col = raw(:, j);
    num = str2double(col);
    if all(isnan(num) == cellfun(@isempty, col))   % numeric (blank = NaN)
        S.(name) = num;
    else
        S.(name) = col;
    end
end
end
