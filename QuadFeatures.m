%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Octavio Narvaez-Aroche                                                  %
% ocnaar@berkeley.edu                                                     %
% Berkeley Center for Control and Identification                          %
% Fall 2016                                                               % 
%                                                                         %
% Function for building a matrix of features to perform a quadratic       %
% regression.                                                             %
%                                                                         %
% Input                                                                   %
% 	X: m by n array, with m>n.                                            %
%                                                                         %
% Output                                                                  %
% 	phi: m by n+1+sum(1:n) array.                                         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function phi = QuadFeatures(X)

% Number of samples, and the dimension of their space.
[ns,np] = size(X);

% Number of quadratic and cross terms.
nc = sum(1:np);

% Vector of quadratic and cross terms.
xcs = zeros(1,nc);

% Compute matrix of features. 
phi = zeros(ns,np+nc+1);
for i=1:ns
	% Obtain quadratic, and cross terms.
    k = 1;
    l = k;
    for j=1:nc
        xcs(j) = X(i,k)*X(i,l);
        l = l+1;
        if l>np
            k = k+1;
            l = k;
        end
    end
	% Vector of features for each sample in X.
    phi(i,:) = [1,X(i,:),xcs];
end