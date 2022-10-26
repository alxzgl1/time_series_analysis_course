%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_AM_FM()

% clear command window
clc;

% parameters
fs = 1000;    % sampling rate of original signal
T = 1;        % signal duration
N = T * fs;   % number of samples

fc = 30;      % carrier frequency
f0 = 3;       % modulation frequency
AM = 0.5;     % modulation factor (depth)
FM = 4;       % modulation factor (depth)

% time variable
t = linspace(0, T, N);

% AM signal
y1 = (1 + AM * cos(2 * pi * t * f0));
u1 = sin(2 * pi * t * fc);
s1 = y1 .* u1;

% FM signal
y2 = cos(2 * pi * t * f0);
s2 = cos(2 * pi * t * fc + FM * sin(2 * pi * t * f0));

% plot signals
subplot(2, 1, 1);
plot(t, s1, 'k'); hold on;
plot(t, y1, 'r--');
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.6, 1.6], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

subplot(2, 1, 2);
plot(t, s2, 'k'); hold on;
plot(t, y2, 'r--'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.6, 1.6], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
