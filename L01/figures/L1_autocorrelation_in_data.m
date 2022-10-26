%-------------------------------------------------------------------------------
% Function, E1_autocorrelation_in_data()
%-------------------------------------------------------------------------------
function L1_autocorrelation_in_data()

clc;

% parameters
nDuration = 2; % seconds
fs = 1000;   	% sampling rate of original signal
fql = 25;     % sampling frequency
fqh = 100; 	  % sampling frequency

N = nDuration * fs;

% time variable
dt = 1 / fs;
t = (0:dt:(nDuration - dt))';

% signal
y = randn(N, 1);
[b, a] = butter(4, [8, 12] / (fs / 2));
y = filtfilt(b, a, [y(end:-1:1); y; y(end:-1:1)]);
y = y((N + 1):(2 * N));

% get sampled signal - low sampling frequency
ql = zeros(N, 1);
for n = 1:N
  if rem(n, floor(fs / fql)) == 0 || n == 1
    ql(n) = 1;
  end
end
ul = y .* ql;

% get sampled signal - high sampling frequency
qh = zeros(N, 1);
for n = 1:N
  if rem(n, floor(fs / fqh)) == 0 || n == 1
    qh(n) = 1;
  end
end
uh = y .* qh;

% figure 1
figure(1);

% low fq
subplot(2, 2, 1); plot(t, y); hold on;
subplot(2, 2, 1); 
for n = 1:N
  if ul(n) ~= 0
    line([t(n), t(n)], [0, y(n)], 'Color', 'r'); 
  end
end
line([t(1), t(end)], [0, 0], 'Color', 'r'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% high fq
subplot(2, 2, 2); plot(t, y); hold on;
subplot(2, 2, 2); 
for n = 1:N
  if uh(n) ~= 0
    line([t(n), t(n)], [0, y(n)], 'Color', 'r'); 
  end
end
line([t(1), t(end)], [0, 0], 'Color', 'r'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% low fq samples
subplot(2, 2, 3); 
c = [0, 0.5, 1];
for n = 1:N
  if ul(n) ~= 0
    line([t(n), t(n)], [0, y(n)], 'Color', 'r'); 
    plot(t(n), ul(n), 'Color', c, 'Marker', 'o', 'MarkerSize', 2, 'MarkerFaceColor', c); hold on;
  end
end
line([t(1), t(end)], [0, 0], 'Color', 'r'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% high fq samples
subplot(2, 2, 4); 
c = [0, 0.5, 1];
for n = 1:N
  if uh(n) ~= 0
    line([t(n), t(n)], [0, y(n)], 'Color', 'r'); 
    plot(t(n), uh(n), 'Color', c, 'Marker', 'o', 'MarkerSize', 2, 'MarkerFaceColor', c); hold on;
  end
end
line([1, N], [0, 0], 'Color', 'r'); 
set(gca, 'XLim', [t(1), t(end)], 'YLim', [-0.5, 0.5], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'time (s)', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% figure 2
figure(2);

% low fq autocorrelation
ul = ul(ul ~= 0); % remove zeros
aul = autocorr(ul, length(ul) - 1);
subplot(2, 2, 1); plot(aul);
set(gca, 'XLim', [1, length(ul)], 'YLim', [-1, 1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);

% high fq autocorrelation
uh = uh(uh ~= 0); % remove zeros
auh = autocorr(uh, length(uh) - 1);
subplot(2, 2, 2); plot(auh);
set(gca, 'XLim', [1, length(uh)], 'YLim', [-1, 1], 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
