%-------------------------------------------------------------------------------
% Function 
%-------------------------------------------------------------------------------
function L04_ar_power_estimation_my()

% clear command window
clc;

% parameters
fs = 500; 
T = 1;
N = T * fs;
nFFT = 500; % fft resolution = fs / nFFT (Hz)

% time variable
dt = T / N;
t = (0:dt:(T - dt))'; % linspace(0, T, N)
df = fs / nFFT;
f = (0:df:(fs - df))'; % linspace(0, fs, nFFT)

% generate signal
f1 = 25; X1 = sin(2 * pi * t * f1);
f2 = 50; X2 = sin(2 * pi * t * f2);
x = X1 + X2 + 0.0 * randn(N, 1);

% Yule AR model
order = 12;
a = aryule(x, order);

% Fourier transform of AR coefficients
y = my_fft(a, nFFT); % fft(a, nFFT)

% power spectrum
Y = abs(1 ./ y);

% Yule power
U = pyulear(x, order);

% plot
subplot(2, 1, 1); plot(x);
subplot(2, 1, 2); 
plot(f, Y / max(Y)); hold on;
F = linspace(0, fs/2, length(U));
plot(F, U / max(U));

end % end 

%-------------------------------------------------------------------------------
% Function, my_fft()
%-------------------------------------------------------------------------------
function y = my_fft(x, nFFT)

N = length(x);
% correct length of x
if N < nFFT
  x = [x(:); zeros(nFFT - N, 1)];
end
N = length(x);
y = zeros(nFFT, 1); 
u = zeros(nFFT, 1);
t = (0:(N-1))';
for k = 0:(nFFT - 1)
  % relative frequency
  f = k / nFFT;
  % complex exponent
  y(k + 1) = sum(exp(1i * 2 * pi * t * f) .* x);
  % cos + 1i * sin
  u(k + 1) = sum(cos(2 * pi * t * f) .* x + ...
            1i * sin(2 * pi * t * f) .* x);
end

end % end

%-------------------------------------------------------------------------------
