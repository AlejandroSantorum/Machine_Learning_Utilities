% =========================================== %
%
% Project:   Machine Learning utilities
% File:      hypothesis.m
% Version:   2.0
% Date:      May 22, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: matrix of training set
%   - theta: vector of predicted parameters
%
% OUTPUT:
%   - H: hypothesis vector
%
% FUNCTIONALITY:
%   It computes the learning hypothesis for every row
%   of the training set matrix. So the result is a
%   vector that each element is the hypothesis result
%   of each row of matrix X (design matrix).
%
%   [X(1,0)  X(1,1) ... X(1,n)]  [theta(0)]     [H(1)]
%   [X(2,0)  X(2,1) ... X(2,n)]  [theta(1)]     [H(2)]
%     .                    .         .       =    .
%     .                    .         .            .
%     .                    .         .            .
%   [X(m,0)  X(m,1) ... X(m,n)]  [theta(n)]     [H(m)]
function H = hypothesis(X, theta)
    if size(theta,1) < size(theta,2) % row vector
        theta_aux = theta';
    else
        theta_aux = theta;
    end

    if size(X,2) ~= size(theta_aux,1)
        error('Num. columns of X is different than length of Theta')
        error('Check X has the first column equals to ones');
    end

    H = X*theta_aux;
end
