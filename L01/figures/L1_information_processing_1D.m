%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L1_information_processing_1D()

clc;

% parameters
fs = 1000;
nDuration = 4; 
N = nDuration * fs;
f0 = 2;

% time variable
dt = 1 / fs;
t = (0:dt:(nDuration - dt))';

% filtered noise
n = randn(N, 1);
[b, a] = butter(4, [8, 12] / (fs / 2));
yn = filtfilt(b, a, [n(end:-1:1); n; n(end:-1:1)]) * 2;
un = abs(hilbert(yn));
yn = yn((N + 1):(2 * N));
un = un((N + 1):(2 * N));

% modulated signal
fc = 10;
M = 0.5;
ys = sin(2 * pi * t * fc) .* (1 + M * cos(2 * pi * t * f0)) / 4;
us = abs(hilbert(ys));

% draw
subplot(2, 2, 1);
plot(t, yn, 'k'); hold on;
plot(t, un, 'Color', 'r'); 
set(gca, 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 2, 2);
plot(t, autocorr(un, length(un) - 1));
set(gca, 'YLim', [-1, 1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);

subplot(2, 2, 3);
plot(t, ys, 'k'); hold on;
plot(t, us, 'Color', 'r'); 
set(gca, 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 2, 4);
plot(t, autocorr(us, length(un) - 1));
set(gca, 'YLim', [-1, 1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);

end % end 

%-------------------------------------------------------------------------------