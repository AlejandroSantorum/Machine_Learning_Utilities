% =========================================== %
%
% Project:   Machine Learning utilities
% File:      costFunctionLinR_Reg.m
% Version:   2.0
% Date:      May 23, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: training set matrix (design matrix)
%   - y: vector of correct results (expected results)
%   - theta: vector of predicted parameters
%   - lambda: regularization parameter
%
% OUTPUT:
%   - J: cost function result
%
% FUNCTIONALITY:
%   It computes the cost function (linear regression) given
%   the predicted parameters theta, the design matrix (training set)
%   and the expected results. It also uses regularization method in
%   order to avoid overfitting.
function J = costFunctionLinR_Reg(X, y, theta, lambda)
    if size(y,1) < size(y,2) % row vector
        y_aux = y';
    else
        y_aux = y;
    end

    if size(X,1) ~= size(y_aux,1)
        error('Num. of rows of X is different than the num. of rows of vector y');
    end
    
    m = length(y);
    H = hypothesis(X,theta);
    J = (1/(2*m))*(sum((H-y).^2)+(lambda*sum(theta.^2))); 
end