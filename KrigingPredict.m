%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Octavio Narvaez-Aroche                                                  %
% ocnaar@berkeley.edu                                                     %
% Berkeley Center for Control and Identification                          %
% Fall 2016                                                               % 
%                                                                         %
% Compute a prediction of f(x), and its expected error using a Kriging    % 
% interpolation of the function f:R->R. Use "Kriging.m" first in order    %
% to obtain the input of this function. The standard deviation of the     %  
% covariance function is given by the hyperparameter sigma.               %
%                                                                         %
% Input                                                                   %
% 	x: n by 1 array with the values where the function f is interpolated. %
% 	xmin: lower bound for the values in x.                                %
% 	xmax: upper bound for the values in x.                                %
% 	xdata: vector of N samples in R.                                      %
% 	SZ: inverse of covariance matrix.                                     %
% 	V: N by 1 array that remains fixed in Kriging Interpolation.          % 
% 	Aq: vector in R^3 with coefficients of the quadratic surface to fit f.%
% 	sigma: non zero hyperparameter.                                       %
%                                                                         %
% Output                                                                  %
% 	y: prediction of f(x).                                                %
% 	e: expected error of the prediction.                                  %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [y,e] = KrigingPredict(x,xmin,xmax,xdata,SZ,V,Aq,sigma)

% Normalize data.
xnorm =(x-xmin)/(xmax-xmin);
xdatanorm =(xdata-xmin)/(xmax-xmin);

% Number of required predictions.
[n,~] = size(x);

% Array of predictions. 
y = zeros(n,1);

% Array of prediction error. 
e = zeros(n,1);

% Number of data points used for computing V and Aq. 
[nd,~] = size(V);

% Array for storing values of the covariance function. 
R = zeros(1,nd);
for i=1:n
    for j=1:nd
		% Compute covariance relative to data points.
        R(j) = exp((-(xnorm(i)-xdatanorm(j))^2)/(sigma^2));
    end
	% Perform Kriging interpolation.
    y(i) = R*V+QuadFeatures(xnorm(i))*Aq;
	% Obtain expected error for the prediction.
    e(i) = real(sqrt(1-R*SZ*R'));
end