%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function test_wavelet()

f = 50;
fs = 1000;

N = 1000;
x = randn(N, 1);

% wavelet init
m = 5;
[a, b] = support_wavelet_init(f, m, fs);
W = support_wavelet_fft(a, b, N)';

% filter data
y = ifft(fft(x) .* W);

plot(x); hold on;
plot(real(y));

end % end

%-------------------------------------------------------------------------------