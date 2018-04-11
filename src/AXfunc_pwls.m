function [y] = AXfunc_pwls(x,A,At,weight,beta1,beta2)
%%% description: 

% qCGMRF weighting
% r =   [beta1*gradient_qCGMRF(x(1:end/2), weight) ...
%     beta2*gradient_qCGMRF(x(end/2+1:end), weight)];

% quadratic weighting
r =   [beta1*gradient_LS(x(1:end/2), weight) ...
    beta2*gradient_LS(x(end/2+1:end), weight)];


r = r(:);
y = (At*(A*x)) + r;
