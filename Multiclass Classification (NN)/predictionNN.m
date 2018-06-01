% =========================================== %
%
% Project:   Machine Learning utilities
% File:      predictionNN.m
% Version:   2.0
% Date:      May 27, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - theta1: first layer trained weights (matrix) of the neural network
%   - theta2: second layer trainer weights (matrix) of the neural network
%   - X: data matrix
%
% OUTPUT:
%   - P: prediction vector
%
% FUNCTIONALITY:
%   It returns a vector of the neural network's results got from
%   each row of the data matrix
%
% NOTE:
%   This function considers a neural network of one input layer,
%   one hidden layer and one output layer
function [P] = predictionNN(theta1, theta2, X)
    m = size(X,1);
    P = zeros(m,1);

    X = [ones(m,1) X];
    z1 = X * theta1';
    H1 = sigmoid(z1);

    H1 = [ones(m,1) H1];
    z2 = H1 * theta2';
    H2 = sigmoid(z2);

    [pVal, P] = max(H2, [], 2);
end
