% =========================================== %
%
% Project:   Machine Learning utilities
% File:      normalEquationLinR.m
% Version:   2.0
% Date:      May 22, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: training set matrix (design matrix)
%   - y: vector of expected results
%
% OUTPUT:
%   - theta: fitting parameters
%
% FUNCTIONALITY:
%   It returns the fitting parameters in the vector theta for
%   linear regression. This implementation is much faster than
%   gradient descent if the number of features is relatively
%   small (1-10000) approx.
function [theta] = normalEquationLinR(X,y)
    theta = zeros(size(X, 2), 1);
    theta = pinv(X'*X)*X'*y;
end