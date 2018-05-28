% =========================================== %
%
% Project:   Machine Learning utilities
% File:      prediction.m
% Version:   2.0
% Date:      May 23, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - theta: fitting parameters vector for logistic regression
%   - X: design matrix (training examples) for logistic regression
%   - threshold: boundary that determines whenever a new input value
%                evaluated in the hypothesis is positive (y=1) or
%                negative (y=0)
%
% OUTPUT:
%   - P: prediction vector
%
% FUNCTIONALITY:
%   It returns a vector of 1's and 0's as a result of evaluate new
%   examples with the fitting parameters theta in the hypothesis.
function P = prediction(theta, X, threshold)
    m = size(X,1);
    P = zeros(m,1);
    H = sigmoid(hypothesis(theta, X));
    for k = 1:m
        if h(k) >= threshold
            P(k) = 1;
        else
            P(k) = 0;
        end
    end
end