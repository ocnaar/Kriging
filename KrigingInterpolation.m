%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Octavio Narvaez-Aroche                                                  %
% ocnaar@berkeley.edu                                                     %
% Berkeley Center for Control and Identification                          %
% Fall 2016                                                               % 
%                                                                         %
% Script to exemplify the Kriging interpolation of a function f:R->R.     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set a clean workspace. 
clear
clc
close all

%% Sample n Data Points from a function
tic
% Define function to interpolate

% Product of linear and exponential functions. 
f = @(x)x.*exp(-x);	

% Sigmoid function.
% f = @(x)sigmoid(x);

% Lower and upper bounds for the independent variable. 
 
% For the product of functions. 
xmin = 0;
xmax = 8;

% For the sigmoid function.
% xmin = -6;
% xmax = 6;

% Sample points for computing the linear minimum-variance estimator.
n = 11;  % Number of sampled points. 
xdata = linspace(xmin,xmax,n)';
ydata = f(xdata);

% Sample a fine grid to plot target function. 
xf = linspace(xmin,xmax,20*n)';
yf = f(xf);

% Choose different values for the hyperparameter sigma.
sigma = 10.^(-2:1);

% Arrays for storing results for different sigma values.
x = linspace(xmin,xmax,20*n)';
yp = zeros(length(x),length(sigma));
ep = zeros(length(x),length(sigma));
yq = zeros(length(x),length(sigma));
cR = zeros(1,length(sigma));
Aq = zeros(3,length(sigma));
xquad = (x-xmin)./(xmax-xmin);

% Perform quadratic surface fit, and Kriging interpolation
for i=1:length(sigma)
    % Calculate invariant elements used in Kriging interpolation. 
    [SZ,V,Aq(:,i),cR(i)] = Kriging(xdata,xmin,xmax,ydata,sigma(i));
    % Quadratic surface fit.
    yq(:,i) = QuadFeatures(xquad)*Aq(:,i);
    % Perform Kriging interpolation for values in x. 
    [yp(:,i),~] = KrigingPredict(x,xmin,xmax,xdata,SZ,V,Aq(:,i),sigma(i));
end

% Plot Regressions
for i=1:length(sigma)
    figure(i)
    plot(xf,yf,'r-',xdata,ydata,'rx',x,yq(:,i),'g-',x,yp(:,i),'LineWidth',2)
    legend('f(x)','Sample Points','Quadratic Fit','Kriging Interpolation','Location','Best')
    grid
    xlabel('x')
    title({['Kriging Regression to interpolate function f(x) with \sigma=',num2str(sigma(i))],['cond(R)=',num2str(cR(i))]})
end

%% Choose bounds on sigma to perform cross-validation
ncv = 1000;       % Number of cross-validations.
sigmamin = 0.1;
sigmamax = 1;
sigmacv = linspace(sigmamin,sigmamax,ncv);

% Sample a Latin Hypercube to randomly select training and validation
% samples for performing cross-validation. 
rng(1);
xcv = (xmax-xmin)*lhsdesign(n+5,1)+xmin;
idx = randperm(length(xcv)); 
xcvtrain = sort(xcv(idx(1:n)));
xcvval = sort(xcv(idx(n+1:end)));

%% Array for storing cross-validation error
cverror = zeros(1,ncv);

% Perform Cross-Validation
for i = 1:ncv
    % Calculate invariant elements used in Kriging regression with training
    % samples. 
    [SZ,V,Aq,~] = Kriging(xcvtrain,xmin,xmax,f(xcvtrain),sigmacv(i));
    % Perform Kriging interpolation for values in vector x. 
    [ycv,~] = KrigingPredict(xcvval,xmin,xmax,xcvtrain,SZ,V,Aq,sigmacv(i));
    % Calculate values of the function at validation samples. 
    yreal = f(xcvval);
    cverror(i) = sum(abs(yreal-ycv));
end

% Plot cross-validation error
figure
plot(sigmacv,cverror,'LineWidth',2)
grid
xlabel('\sigma')
ylabel('Cross-validation error')

% Choose best value of sigma
[~,I] = min(cverror);
bestsigma = sigmacv(I);

%% Plot Interpolation with best value of sigma

% Compute invariant elements in Kriging interpolation.
[SZ,V,Aq,cR] = Kriging(xdata,xmin,xmax,ydata,bestsigma);
% Perform Kriging interpolation for values in vector x.
[yp,~] = KrigingPredict(x,xmin,xmax,xdata,SZ,V,Aq,bestsigma);

figure
plot(xf,yf,'r-',xdata,ydata,'rx',x,yp,'b-','LineWidth',2)
legend('f(x)','Sample Points','Kriging Interpolation','Location','Best')
grid
xlabel('x')
title({['Kriging Regression to interpolate function f(x) with \sigma=',num2str(bestsigma)],['cond(R)=',num2str(cR)]})


%% Plot error estimates

% Choose a value of sigma to plot confidence bounds.
esigma = 0.1;

% Calculate invariant elements in Kriging interpolation. 
[SZ,V,Aq,cR] = Kriging(xdata,xmin,xmax,ydata,esigma);
% Perform Kriging interpolation for x.
[yp,ep] = KrigingPredict(x,xmin,xmax,xdata,SZ,V,Aq,esigma);

figure
plot(xf,yf,'r-',xdata,ydata,'rx',x,yp,'b-',x,yp+ep,'b--',x,yp-ep,'b--','LineWidth',2)
legend('f(x)','Sample Points','Kriging Interpolation','Error upper bound','Error lower bound','Location','Best')
grid
xlabel('x')
title({['Kriging Regression to interpolate function f(x) with \sigma=',num2str(esigma)],['cond(R)=',num2str(cR)]})
toc