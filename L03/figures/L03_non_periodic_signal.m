%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_non_periodic_signal()

clc;

% parameters
fs = 1000;    % sampling rate of original signal
T = 1;        % signal duration
N = T * fs;   % number of samples

% time variable
t = linspace(0, T, N);

% chirp signal
f0 = 1;
f1 = 10;
y = chirp(t, f0, T, f1);

% spike signal
f0 = 10;
u = sin(2 * pi * f0 * t);
u(floor(N/4):end) = 0;

% plot signals
subplot(2, 1, 1);
plot(t, y, 'k'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.5, 1.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 1, 2);
plot(t, u, 'k'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.5, 1.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
