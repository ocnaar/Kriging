%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Octavio Narvaez-Aroche                                                  %
% ocnaar@berkeley.edu                                                     %
% Berkeley Center for Control and Identification                          %
% Fall 2016                                                               % 
%                                                                         %
% Compute quadratic regression for sampled points, and the terms of the   % 
% linear minimum-variance estimator that remain fixed during the Kriging  % 
% interpolation of the function f:R->R. A Gaussian function with standard %
% deviation of sigma is used to compute the covariance matrix.            %
%                                                                         %
% Input                                                                   %
% 	x: vector of n samples in R.                                          %
% 	xmin: lower bound for the values in x.                                %
% 	xmax: upper bound for the values in x.                                %
% 	z: n by 1 array with the values for f(x_k).                           %      
% 	sigma: non zero hyperparameter.                                       %
%                                                                         %
% Output                                                                  %
% 	SZ: inverse of covariance matrix.                                     %
% 	V: n by 1 array that remains fixed in Kriging Interpolation.          % 
% 	Aq: vector in R^3 with coefficients of the quadratic surface to fit f.%
% 	cR: condition number for inversion of the covariance matrix.          %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [SZ,V,Aq,cR] = Kriging(x,xmin,xmax,z,sigma)

% Determine number of samples. 
[n,~] = size(x);

% Fit data with a quadratic surface by solving a least squares problem.
xnorm = (x-xmin)/(xmax-xmin);    % Normalize samples. 
phi = QuadFeatures(xnorm);       % Features for quadratic regression.
Aq = phi\z;                      % Coefficients of quadratic surface.  
q = phi*Aq;                      % Estimate for z.     

% Compute covariance matrix RM 
RM = zeros(n,n);
for i=1:n
    for j=1:n
        RM(i,j) = exp((-(xnorm(i)-xnorm(j))^2)/(sigma^2));
    end
end

% Add small weights in the diagonal of RM to improve condition number.
% RM = RM+1e-8*eye(length(RM));

% Condition number of RM
cR = cond(RM);

% Inverse of covariance matrix. 
SZ = RM\eye(length(RM));

% Compute value for V.
V = RM\(z-q);