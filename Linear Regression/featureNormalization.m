% =========================================== %
%
% Project:   Machine Learning utilities
% File:      featureNormalization.m
% Version:   2.0
% Date:      May 22, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: training set matrix (design matrix)
%
% OUTPUT:
%   - X_normalized: improvced learning parameters after gradient descent
%   - mu: vector of mean values of each feature (column of X)
%   - sigma: vector of standard deviation of each feature
%
% FUNCTIONALITY:
%   It returns a normalized version of X data, where the mean value of
%   each feature is 0 and the standard deviation is 1.
function [X_normalized, mu, sigma] = featureNormalization(X)
    n = size(X,2);
    
    mu = mean(X); % mean of each column of matrix X
    sigma = std(X); % standard deviation of each column of matrix X
    for i = 1:n
        X_normalized(:,i) = (X(:,i) - mu(i))./sigma(i); 
    end
end