%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_periodic_signal()

clc;

% parameters
fs = 1000;    % sampling rate of original signal
T = 1;        % signal duration
N = T * fs;   % number of samples

% time variable
t = linspace(0, T, N);

% signal 1
A1 = 1;
f1 = 5;
phi1 = 0;
y1 = A1 * sin(2 * pi * f1 * t + phi1);

% signal 2
A2 = 1;
f2 = 5;
phi2 = pi / 2;
y2 = A2 * sin(2 * pi * f2 * t + phi2);

% plot signals
subplot(2, 1, 1);
plot(t, y1, 'Color', [0, 0.5, 1], 'LineStyle', '-.'); hold on;
plot(t, y2, 'Color', [1, 0.5, 0], 'LineWidth', 2); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-2, 2], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% subplot(2, 1, 2);
% plot(t, y2, 'k'); 
% set(gca, 'XLim', [t(1), t(end)], 'YLim', [-0.5, 1.5], 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
