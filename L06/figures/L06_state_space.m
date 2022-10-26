%-------------------------------------------------------------------------------
% Function
%-------------------------------------------------------------------------------
function L06_state_space()

% filter parameters
order = 2;
f = 0.5;

% butterworth filter
[A, B, C, D] = butter(order, f);

% state space
Hd = dfilt.statespace(A, B, C, D);

Hd

end % end

%-------------------------------------------------------------------------------
