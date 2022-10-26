%-------------------------------------------------------------------------------
% Function 
%-------------------------------------------------------------------------------
function L04_ar_power_estimation_pure()

% clear command window
clc;

% parameters
N = 1000;
fs = 500;     % Hz, i.e. samples per second

% time variable
dt = 1 / fs;
t = (0:dt:((N / fs) - dt))';

% generate signal
f1 = 15; X1 = sin(2 * pi * t * f1);
f2 = 45; X2 = sin(2 * pi * t * f2);
x = X1 + X2 + 0.0 * randn(N, 1);

% Yule AR model
order = 8;
[a, sigma] = aryule(x, order);
K = length(a);

% Yule power
U = pyulear(x, order);

% estimate power spectral density from AR coeffs
L = length(U);
F = L * 2 - 1;
Y = zeros(F, 1);
for f = 0:(F - 1)
  t = 0;
  for k = 1:K
    t = t + a(k) * exp(-1i * 2 * pi * k * (f / F));
    Y(f + 1) = Y(f + 1) + (sigma ^ 2) / abs(t) .^ 2;
  end
end
Y = Y(1:L);
f = 0:(fs / (2 * L)):(fs / 2 - (fs / (2 * L)));

% plot
subplot(3, 1, 1); plot(x);
subplot(3, 1, 2); 
plot(f, Y / max(Y)); hold on;
plot(f, U / max(U));

end % end 

%-------------------------------------------------------------------------------
