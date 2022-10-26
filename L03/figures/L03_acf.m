%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L03_acf()

clc;

% generate distributions
N = 100;
x = randn(N, 1);

r = autocorr(x, N - 1);
r = [r(end:-1:1); r(2:end)];

% plot(r);
plot(r(1:N/2), r(1:N/2), '.'); hold on;
plot(r(1:N/2), r(N/2:-1:1), '.');


% * figure 1
% figure(1);
% 
% % plot
% subplot(2, 1, 1);
% plot(x0, 'k');
% xlim([1, N]);
% set(gca, 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'samples', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'amplitude', 'FontSize', 12);
% 
% % plot histogram
% subplot(2, 1, 2);
% bar(bx0, hx0, 'k'); hold on;
% plot(bx0, px0, 'r', 'LineWidth', 2);
% xlim([-3.5, 3.5]);
% set(gca, 'FontSize', 12);
% set(get(gca, 'XLabel'), 'String', 'amplitude', 'FontSize', 12);
% set(get(gca, 'YLabel'), 'String', 'counts', 'FontSize', 12);



end % end

%-------------------------------------------------------------------------------
