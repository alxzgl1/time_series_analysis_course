%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L1_information_processing_OLD()

clc;

% parameters
fs = 1000;
nDuration = 2;
fc = 50;      
fm = 5;        
M = 0.5;      

% time variable
dt = 1 / fs;
t = (0:dt:(nDuration - dt))';

% AM signal
u = (1 + M * cos(2 * pi * t * fm));
y = sin(2 * pi * t * fc) .* u;

% get peaks
% th = 1.0;
% p = get_peaks(y, th);
% p(1:floor(0.5 * length(p))) = 0;
% p = p + u / 10;

% draw
plot(t, y, 'k'); set(gca, 'YLim', [0, 1.5]);
set(get(gca, 'XLabel'), 'String', 'time (s)');
set(get(gca, 'YLabel'), 'String', 'amplitude');

end % end 

%-------------------------------------------------------------------------------
% Function, reconstruct_signal()
%-------------------------------------------------------------------------------
function u = get_peaks(y, th)

N = length(y);
y(y < th) = 0;
u = zeros(N, 1);
for n = 2:(N - 1)
  if y(n - 1) < y(n) && y(n + 1) < y(n)
    u(n) = 1;
  end
end

end % end

%-------------------------------------------------------------------------------