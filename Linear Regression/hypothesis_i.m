% =========================================== %
%
% Project:   Machine Learning utilities
% File:      hypothesis_i.m
% Version:   2.0
% Date:      May 22, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - Xvector: vector of features (i-th row of training examples)
%   - theta: vector of predicted parameters
%
% OUTPUT:
%   - H: hypothesis value for the i-th row
%
% FUNCTIONALITY:
%   It computes the learning hypothesis for the i-th row of
%   the training set based on the learning parameter vector
%   theta.
%   H = theta(0)*X(0) + theta(1)*X(1) + ... + theta(n)*X(n)
function H = hypothesis_i(Xvector, theta)
    if length(Xvector) ~= length(theta)
        error('Error: X vector size is different from the size of vector Theta');
        error('Please, check that X has the first element equals to 1');
    end

    H = dot(Xvector,theta);
end
