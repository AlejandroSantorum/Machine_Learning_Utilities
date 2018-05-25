% =========================================== %
%
% Project:   Machine Learning utilities
% File:      normalEquationLinR_Reg.m
% Version:   2.0
% Date:      May 23, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: training set matrix (design matrix)
%   - y: vector of expected results
%   - lambda: regularization parameter
%
% OUTPUT:
%   - theta: fitting parameters
%
% FUNCTIONALITY:
%   It returns the fitting parameters in the vector theta for
%   linear regression using regularization method to avoid
%   overfitting. This implementation is much faster than
%   gradient descent if the number of features is relatively
%   small (1-10000) approx.
function [theta] = normalEquationLinR_Reg(X, y, lambda)
    n = size(X,2);
    theta = zeros(n, 1);
    reg_matrix = lambda*eye(n);
    reg_matrix(1,1) = 0;
    theta = pinv(X'*X+reg_matrix)*X'*y;
end