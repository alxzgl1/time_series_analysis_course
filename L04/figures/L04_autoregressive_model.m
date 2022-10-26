%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L04_autoregressive_model()

clc;

% parameters
fs = 1000;    % sampling rate of original signal
T = 1;        % signal duration
N = T * fs;   % number of samples

% time variable
t = linspace(0, T, N);

% signal
f0 = 5;
x = sin(2 * pi * f0 * t) + 0.5 * randn(1, N);

% fit model
p = 2;
m = ar(x(:), p);


% plot signals
subplot(2, 1, 1);
plot(t, x, 'Color', [0, 0.5, 1]); 
% plot(t, y2, 'Color', [1, 0.5, 0], 'LineWidth', 2); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-2, 2], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
