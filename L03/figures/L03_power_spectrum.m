%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_power_spectrum()

clc;

% sampling parameters
fs = 1000;    % sampling rate, in Hz
T  = 1;      % duration, in seconds
N  = T * fs; % duration, in samples
M = 20;

% time variable
t = linspace(0, T, N);
f = linspace(0, fs/M, (N/M)); % HERE, fs/10

% sin signal
f0 = 5;
x1 = sin(2 * pi * f0 * t);
% chirp signal
f0 = 1;
f1 = 10;
t1 = T;
x2 = chirp(t, f0, t1, f1);
% non-periodic signal
f0 = 10;
x3 = sin(2 * pi * f0 * t);
x3(floor(N/4):end) = 0;
x3 = x3 / max(x3);
% random noise
x4 = randn(1, N);
x4 = x4 / max(x4);

% fft
y1 = abs(fft(x1));
y1 = y1(1:(N/M));
y1 = y1 / max(y1);

y2 = abs(fft(x2));
y2 = y2(1:(N/M));
y2 = y2 / max(y2);

y3 = abs(fft(x3));
y3 = y3(1:(N/M));
y3 = y3 / max(y3);

y4 = abs(fft(x4));
y4 = y4(1:(N/M));
y4 = y4 / max(y4);

% plot

figure(1);

subplot(2, 2, 1);
plot(t, x1, 'k');
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [-1.5, 1.5], 'FontSize', 12);

subplot(2, 2, 2);
plot(f, y1, 'r-.', 'Marker', '.');
set(get(gca, 'XLabel'), 'String', 'frequency (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
set(gca, 'YLim', [0, 1.5], 'XLim', [1, length(y1)], 'FontSize', 12);

subplot(2, 2, 3);
plot(t, x2, 'k');
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [-1.5, 1.5], 'FontSize', 12);

subplot(2, 2, 4);
plot(f, y2, 'r-.', 'Marker', '.');
set(get(gca, 'XLabel'), 'String', 'frequency (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
set(gca, 'YLim', [0, 1.5], 'XLim', [1, length(y2)], 'FontSize', 12);

figure(2);

subplot(2, 2, 1);
plot(t, x3, 'k');
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [-1.5, 1.5], 'FontSize', 12);

subplot(2, 2, 2);
plot(f, y3, 'r-.', 'Marker', '.');
set(get(gca, 'XLabel'), 'String', 'frequency (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
set(gca, 'YLim', [0, 1.5], 'XLim', [1, length(y2)], 'FontSize', 12);

subplot(2, 2, 3);
plot(t, x4, 'k');
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [-1.5, 1.5], 'FontSize', 12);

subplot(2, 2, 4);
plot(f, y4, 'r-.', 'Marker', '.');
set(get(gca, 'XLabel'), 'String', 'frequency (Hz)');
set(get(gca, 'YLabel'), 'String', 'power');
set(gca, 'YLim', [0, 1.5], 'XLim', [1, length(y2)], 'FontSize', 12);

end % end 

%-------------------------------------------------------------------------------