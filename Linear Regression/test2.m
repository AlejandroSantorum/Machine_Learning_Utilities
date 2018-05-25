% Initialization
clear ; close all; clc

fprintf('Preparing data test\n')
data = load('data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu, sigma] = featureNormalization(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 4000;

% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
[theta, J_history] = gradientDescentLinR(X, y, theta, alpha, num_iters);
theta
fprintf('Program paused. Press enter to continue.\n');
pause;

% Calculate the parameters from the normal equation
theta = normalEquationLinR(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
