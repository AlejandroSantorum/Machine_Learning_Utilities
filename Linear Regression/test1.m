% Initialization
clear ; close all; clc

fprintf('Preparing data ...\n')
data = load('data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

fprintf('Program paused. Press enter to continue.\n');
pause;

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = costFunctionLinR(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescentLinR(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');

theta2 = normalEquationLinR(X,y);
theta2
