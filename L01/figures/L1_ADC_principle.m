%-------------------------------------------------------------------------------
% Function, L1_ADC_principle()
%-------------------------------------------------------------------------------
function L1_ADC_principle()

% clear command window
clc;

% parameters
N = 1000;     % number of samples
fs = 1000;    % sampling rate of original signal
f = 5;        % frequency of original signal
fq = 25;      % sampling rate 

nDuration = N / fs;

% signal
dt = 1 / fs;
t = (0:dt:(nDuration - dt))';
y = sin(2 * pi * t * f);

% sampling sequence
q = zeros(N, 1);
for n = 1:N
  if rem(n, floor(fs / fq)) == 0 || n == 1
    q(n) = 1;
  end
end

% sampled signal
u = y .* q;

% reconstruction of signal 
w = reconstruct_signal(u);

% plot signals
plot_signals(t, y, q, u, w, N);

end % end

%-------------------------------------------------------------------------------
% Function, reconstruct_signal()
%-------------------------------------------------------------------------------
function w = reconstruct_signal(u)

% signal reconstruction 
x = (-7.99:0.02:8) * pi;
r = sin(x) ./ x;
w = conv(u, r);
w = w((floor(length(x) / 2) + 1):(end - floor(length(x) / 2)));
w = w / max(w);

end % end

%-------------------------------------------------------------------------------
% Function, plot_signals()
%-------------------------------------------------------------------------------
function plot_signals(t, y, q, u, w, N)

% original signal
subplot(2, 2, 1); plot(t, y, 'k');
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.1, 1.1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% sampling sequence
subplot(2, 2, 2); 
for n = 1:N
  plot(t, y, 'k'); hold on; 
  if q(n) ~= 0
    line([t(n), t(n)], [0, 0.3], 'Color', 'r'); 
  end
end
line([t(1), t(end)], [0, 0], 'Color', 'r'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.1, 1.1], 'FontSize', 12); 
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% sampled signal
subplot(2, 2, 3); 
for n = 1:N
  if u(n) ~= 0 
    line([t(n), t(n)], [0, u(n)], 'Color', 'r');
    plot(t(n), u(n), 'Color', 'b', 'Marker', 'o', 'MarkerSize', 3, 'MarkerFaceColor', 'b'); hold on;
  end
end
line([t(1), t(end)], [0, 0], 'Color', 'r');
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.1, 1.1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% reconstructed signal
subplot(2, 2, 4); 
for n = 1:N
  if u(n) ~= 0
    line([t(n), t(n)], [0, u(n)], 'Color', 'r');
    plot(t(n), u(n), 'Color', 'b', 'Marker', 'o', 'MarkerSize', 3, 'MarkerFaceColor', 'b'); hold on;
  end
end
line([t(1), t(end)], [0, 0], 'Color', 'r');
plot(t(1:(end - 1)), w, 'Color', [0.5, 0.5, 0.5]); hold on; 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-1.1, 1.1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

end % end 

%-------------------------------------------------------------------------------
