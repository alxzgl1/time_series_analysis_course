%-------------------------------------------------------------------------------
% Function
% Reference, http://se.mathworks.com/help/signal/ref/pyulear.html
%-------------------------------------------------------------------------------
function L04_ar_power_estimation()

% clear command window
clc;

% parameters
fs = 256;    % Hz, i.e. samples per second
nFFT = 512;  % points, fft resolution = fs / nFFT (Hz)
nDuration = 2;

% time variable
dt = 1 / fs;
t = (0:dt:(nDuration - dt))';

% frequency variable
f = 0:(fs / nFFT):(fs - (fs / nFFT));

% generate signal
f1 = 10; X1 = sin(2 * pi * t * f1);
f2 = 20; X2 = sin(2 * pi * t * f2);
f3 = 40; X3 = sin(2 * pi * t * f3);

x = (X1 + X2 + X3) / 3;

% Autoregressive power spectral density estimate - Yule-Walker method
order = 20; U = pyulear(x, order);

% fft
Y = abs(fft(x, nFFT)); 

% plot signals
subplot(2, 2, [1, 2]);
plot(t, x, 'k'); hold on;
plot(t, X1 + 2.25, 'r');
plot(t, X2 + 4.5, 'Color', [0, 0.5, 0]);
plot(t, X2 + 6.75, 'b');
set(gca, 'XLim', [0, nDuration], 'YLim', [-2, 8], 'FontSize', 8);
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
aTitle = sprintf('Signal mixture  / f1=%d (Hz), f2=%d (Hz), f3=%d (Hz)', f1, f2, f3);
title(aTitle, 'FontSize', 8);

subplot(2, 2, 3);
plot(f, Y, 'Color', [0, 0, 0], 'Marker', '.'); hold on;
set(gca, 'XLim', [0, fs/4], 'FontSize', 8);
set(get(gca, 'XLabel'), 'String', 'f (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
title('FFT spectrum', 'FontSize', 8);

subplot(2, 2, 4);
plot(U, 'Color', [0, 0, 0], 'Marker', '.'); hold on;
set(gca, 'XLim', [0, fs/4], 'FontSize', 8);
set(get(gca, 'XLabel'), 'String', 'f (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
title('AR spectrum', 'FontSize', 8);

end % end 

%-------------------------------------------------------------------------------
