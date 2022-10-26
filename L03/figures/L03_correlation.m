%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_correlation()

clc;

% generate data
N = 100;
x = randn(N, 1);
y = randn(N, 1);

x = x / max(x);
y = y / max(y);

u = (x + y) / 2;
z = (x .^ 2 + y .^ 2) / 2;

% correlation coefficient using numpy
rxx = corr(x, x);
rxy = corr(x, y);
rxu = corr(x, u);
rxz = corr(x, z);
fprintf(1, '%1.2f, %1.2f, %1.2f, %1.2f\n', rxx, rxy, rxu, rxz);

% fit
p = polyfit(x, x, 1); fx = p(1) * x + p(2);
p = polyfit(x, y, 1); fy = p(1) * x + p(2);
p = polyfit(x, u, 1); fu = p(1) * x + p(2);
p = polyfit(x, z, 1); fz = p(1) * x + p(2);

draw_plot = 0;
if draw_plot == 1
  % figure 1
  figure(1);

  % plot
  subplot(2, 1, 1);
  plot(x, 'k');
  xlim([1, N]);
  ylim([-1.25, 1.25]);
  set(gca, 'FontSize', 12);
  set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
  set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

  subplot(2, 1, 2);
  plot(y, 'r');
  xlim([1, N]);
  ylim([-1.25, 1.25]);
  set(gca, 'FontSize', 12);
  set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
  set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);
end

% figure 2
L = 1.25 * max([abs(x); abs(y)]);
figure(2);

subplot(2, 2, 1);
plot(x, x, 'r.', 'MarkerSize', 8); hold on;
plot(x, fx, 'k');
xlim([-L, L]);
ylim([-L, L]);
set(gca, 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'x', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'x', 'FontSize', 12);

subplot(2, 2, 2);
plot(x, y, 'r.', 'MarkerSize', 8); hold on;
plot(x, fy, 'k');
xlim([-L, L]);
ylim([-L, L]);
set(gca, 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'x', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'y', 'FontSize', 12);

subplot(2, 2, 3);
plot(x, u, 'r.', 'MarkerSize', 8); hold on;
plot(x, fu, 'k');
xlim([-L, L]);
ylim([-L, L]);
set(gca, 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'x', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'u', 'FontSize', 12);

subplot(2, 2, 4);
plot(x, z, 'r.', 'MarkerSize', 8); hold on;
plot(x, fz, 'k');
xlim([-L, L]);
ylim([-0.5, L]);
set(gca, 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'x', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'u', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
