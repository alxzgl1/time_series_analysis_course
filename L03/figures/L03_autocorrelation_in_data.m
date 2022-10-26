%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_autocorrelation_in_data()

clc;

% generate gaussian noise
N = 200;
rng(1);
x = randn(N, 1);

% smooth signal by 4 neighboring points
y = filtfilt(ones(1, 4) / 4, 1, x);

% compute ACF
rx = autocorr(x, length(x) - 1);
rx = rx / max(rx);
rx = [rx(end:-1:1); rx(2:end)]; 
ry = autocorr(y, length(y) - 1);
ry = ry / max(ry);
ry = [ry(end:-1:1); ry(2:end)]; 

% original data
figure(1);

% plot data
subplot(2, 1, 1);
plot(x, 'k'); hold on;
% plot(y, 'r');
xlim([0, N]);
ylim([-3, 3]);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% plot ACF
subplot(2, 1, 2);
plot(rx, 'k'); hold on;
% plot(ry, 'r');
xlim([0, 2*N-1]);
ylim([-0.5, 1.25]);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);

% smoothed data
figure(2);

% plot data
subplot(2, 1, 1);
plot(x, 'k'); hold on;
plot(y, 'r');
xlim([0, N]);
ylim([-3, 3]);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% plot ACF
subplot(2, 1, 2);
plot(rx, 'k'); hold on;
plot(ry, 'r');
xlim([0, 2*N-1]);
ylim([-0.5, 1.25]);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'correlation', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
