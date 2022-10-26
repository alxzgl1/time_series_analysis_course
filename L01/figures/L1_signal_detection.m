%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L1_signal_detection()

clc;

% parameters
fs = 1000;          % Hz
N = 1000;           % samples
nDuration = N / fs; % seconds

nFFT = 1000;  % points, fft resolution = fs / nFFT (Hz)

% time variable
dt = 1 / fs;        % seconds
t = (0:dt:(nDuration - dt))';

% frequency variable
f = 0:(fs / nFFT):(fs - (fs / nFFT));

% generate signal
f1 = 10; X1 = (1 / f1) * sin(2 * pi * t * f1 + pi / 4);
f2 = 20; X2 = (1 / f2) * sin(2 * pi * t * f2 + pi / 3);
f3 = 30; X3 = (1 / f3) * sin(2 * pi * t * f3 + pi / 2);
X4 = 0.09 * randn(N, 1);
x = X1 + X2 + X3 + X4;

% fft
Y = abs(fft(x, nFFT)); 

% plot signals
subplot(1, 2, 1);
plot(t, x, 'k'); hold on;
plot(t, X1 + 0.5, 'r');
plot(t, X2 + 0.70, 'Color', [0, 0.5, 0]);
plot(t, X3 + 0.85, 'b');
plot(t, X4 + 1.25, 'Color', [0.5, 0.5, 0.5]);

set(gca, 'XLim', [0, nDuration], 'YLim', [-0.4, 1.6], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
title('Signal mixture', 'FontSize', 12);

subplot(1, 2, 2);
plot(f, Y, 'Color', [0, 0, 0], 'Marker', '.'); hold on;
set(gca, 'XLim', [0, fs / 10], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'f (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
title('FFT spectrum', 'FontSize', 12);

end % end 

%-------------------------------------------------------------------------------