%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L04_burg_power_estimation()

clc;

A = [1 -2.7607 3.8106 -2.6535 0.9238];
[H,F] = freqz(1, A, [], 1);
plot(F, 20 * log10(abs(H)))
xlabel('Frequency (Hz)')
ylabel('PSD (dB/Hz)')

x = randn(1000, 1);
y = filter(1, A, x);
[Pxx, F] = pburg(y, 4, 1024, 1);

hold on;
plot(F, 10*log10(Pxx))
legend('True Power Spectral Density','pburg PSD Estimate')

end % end

%-------------------------------------------------------------------------------
