%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L1_information_processing_2D()

clc;

% parameters
fs = 1000;
nDuration = 4; 
N = nDuration * fs;

% time variable
dt = 1 / fs;
t = (0:dt:(nDuration - dt))';

% filtered noise
x = randn(1, N);
y = randn(1, N);

A = [1, 0.7; 0.3, 1];

xy = A * [x; y];

x = xy(1, :)';
y = xy(2, :)';

[b, a] = butter(4, [8, 12] / (fs / 2));
x = filtfilt(b, a, [x(end:-1:1); x; x(end:-1:1)]);
y = filtfilt(b, a, [y(end:-1:1); y; y(end:-1:1)]);

% envelope
X = abs(hilbert(x));
Y = abs(hilbert(y));
X = X((N + 1):(2 * N));
Y = Y((N + 1):(2 * N));

x = x((N + 1):(2 * N));
y = y((N + 1):(2 * N));

% draw
subplot(2, 2, 1);
plot(t, x, 'k'); hold on;
plot(t, X, 'Color', 'r'); 
set(gca, 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 2, 2);
plot(X, Y, '.');
set(gca, 'XLim', [-0.1, 0.4], 'YLim', [-0.1, 0.4], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'amplitude', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 2, 3);
plot(t, y, 'k'); hold on;
plot(t, Y, 'Color', 'r'); 
set(gca, 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 2, 4);
XY = crosscorr(X, Y, length(X) - 1);
plot([-t(end:-1:1); t(1:(end - 1))], XY);
set(gca, 'YLim', [-1, 1], 'XLim', [-nDuration, nDuration], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);



end % end 

%-------------------------------------------------------------------------------