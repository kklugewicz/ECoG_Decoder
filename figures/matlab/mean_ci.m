function [m, lo, hi] = mean_ci(d)
%MEAN_CI  Mean and two-sided 95% t confidence interval of a vector.
%   Uses tinv when the Statistics Toolbox is present; otherwise a table
%   of t quantiles for small n (exact to 4 digits), so no toolbox is needed.
d = d(:);
n = numel(d);
m = mean(d);
if n < 2
    lo = m; hi = m; return;
end
try
    t = tinv(0.975, n - 1);
catch
    tab = [12.7062 4.3027 3.1824 2.7764 2.5706 2.4469 2.3646 2.3060 2.2622 2.2281 ...
           2.2010 2.1788 2.1604 2.1448 2.1314 2.1199 2.1098 2.1009 2.0930 2.0860 ...
           2.0796 2.0739 2.0687 2.0639 2.0595 2.0555 2.0518 2.0484 2.0452 2.0423];
    if n - 1 <= numel(tab)
        t = tab(n - 1);
    else
        t = 1.96;
    end
end
h = t * std(d) / sqrt(n);
lo = m - h; hi = m + h;
end
