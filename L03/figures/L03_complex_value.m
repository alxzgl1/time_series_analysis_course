%-------------------------------------------------------------------------------
% Function, L3_harmonic_functions()
%-------------------------------------------------------------------------------
function L03_complex_value()

clc;

% parameters
fs = 1000; % Hz
T = 1; % seconds
N = T * fs;

A = 1;    % amplitude
f = 5;    % frequency

% init
t = linspace(0, T, N);

% cos & sin
y = A * cos(2 * pi * t * f);
u = A * sin(2 * pi * t * f);

% exp = cos + i * sin
w = A * exp(1i * 2 * pi * t * f);

% figure 1
figure(1);

% plot
subplot(2, 1, 1); plot(t, y, 'b'); hold on; plot(t, u, 'r'); 
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [-1.5, 1.5], 'FontSize', 12);
% title('cos & sin', 'FontSize', 8);

subplot(2, 1, 2); plot(t, real(w), 'b'); hold on; plot(t, imag(w), 'r'); 
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [-1.5, 1.5], 'FontSize', 12);
% title('re\{exp\} & im\{exp\}', 'FontSize', 8);

% figure 2
figure(2);

subplot(2, 2, 1);

plot(w, 'Color', [0, 0, 0]); hold on;
set(get(gca, 'XLabel'), 'String', 'cos(t)');
set(get(gca, 'YLabel'), 'String', 'sin(t)');
set(gca, 'YLim', [-1.05, 1.05], 'XLim', [-1.05, 1.05], 'FontSize', 12);
% title('exp({\iti}*t)', 'FontSize', 8);


subplot(2, 2, 3); plot(t, exp(-5 * t), 'Color', [1, 0.5, 0]); hold on;
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');
set(gca, 'YLim', [0, 1], 'FontSize', 12);
% title('exp(-t)', 'FontSize', 8);

end % end 

%-------------------------------------------------------------------------------