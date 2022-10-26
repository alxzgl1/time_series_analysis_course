%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_amplitude_and_phase()

% clear command window
clc;

% parameters
fs = 1000;    % sampling rate of original signal
T = 1;        % signal duration
N = T * fs;   % number of samples

fc = 30;      % carrier frequency
f0 = 3;       % modulation frequency
M = 0.5;      % modulation factor (depth)

% time variable
t = linspace(0, T, N);

% AM signal
y = (1 + M * cos(2 * pi * t * f0));
u = sin(2 * pi * t * fc);
s = y .* u;

% signal amplitude (envelope) and phase
E = abs(hilbert(s));
P = angle(hilbert(s)) / (2 * pi);

% plot signals
subplot(2, 1, 1);
plot(t, s, 'k'); hold on;
% plot(t, y, 'r');
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.6, 1.6], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 1, 2);
plot(t, E, 'r'); hold on;
plot(t, P, 'b');
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.6, 1.6], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

end % end 

%-------------------------------------------------------------------------------
