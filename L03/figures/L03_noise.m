%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_noise()

clc;

% generate distributions
N = 10000;
x0 = randn(N, 1);
x1 = lognrnd(0.5, 0.5, N, 1);

% histogram - normal
xmin = min(x0);
xmax = max(x0);
bx0 = linspace(xmin, xmax, 100);
hx0 = histc(x0, bx0);
hx0 = hx0 / sum(hx0);

% fit data - normal
[mu, sigma] = normfit(x0);
px0 = normpdf(bx0, mu, sigma);
px0 = px0 / sum(px0);

% histogram - lognormal
xmin = min(x1);
xmax = max(x1);
bx1 = linspace(xmin, xmax, 100);
hx1 = histc(x1, bx1);
hx1 = hx1 / sum(hx1);

% fit data - lognormal
p = lognfit(x1);
px1 = lognpdf(bx1, p(1), p(2));
px1 = px1 / sum(px1);

% * figure 1
figure(1);

% plot
subplot(2, 1, 1);
plot(x0, 'k');
xlim([1, N]);
set(gca, 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% plot histogram
subplot(2, 1, 2);
bar(bx0, hx0, 'k'); hold on;
plot(bx0, px0, 'r', 'LineWidth', 2);
xlim([-3.5, 3.5]);
set(gca, 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'amplitude', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'counts', 'FontSize', 12);

% * figure 2
figure(2);

% plot
subplot(2, 1, 1);
plot(x1, 'k');
xlim([1, N]);
set(gca, 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);

% plot histogram
subplot(2, 1, 2);
bar(bx1, hx1, 'k'); hold on;
plot(bx1, px1, 'r', 'LineWidth', 2);
xlim([0, 7]);
set(gca, 'FontSize', 12);
set(get(gca, 'XLabel'), 'String', 'amplitude', 'FontSize', 12);
set(get(gca, 'YLabel'), 'String', 'counts', 'FontSize', 12);

end % end

%-------------------------------------------------------------------------------
