%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function [p11, p12] = support_wavelet_init(f0, m, fs)

%% init
p1 = 1 / fs;
p2 = 1 / (f0 / m * (2 * pi));
p3 = round((p2 * 10) / p1);
p3 = p3 + rem(p3, 2);

%% carrier
p4 = ((0:(p3 - 1)) * p1) - (((p3 - 1) * p1) / 2);

%% formulae
p5 = (-1) * ((p4 .^ 2) / ((p2 .^ 2) * 2));
p6 = p4 * (2 * pi * f0);

%% shape
p7 = complex(p5, p6);
p8 = exp(p7) * (1 / sqrt(p2 * sqrt(pi)));
p9 = p8 / (sum(abs(p8)) / 2);

%% split into halves
p10 = round(p3 / 2);
p11 = p9(1:p10);
p12 = p9((p10 + 1):end);

end % end

%-------------------------------------------------------------------------------