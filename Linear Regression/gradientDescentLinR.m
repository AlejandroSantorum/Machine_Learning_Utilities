% =========================================== %
%
% Project:   Machine Learning utilities
% File:      gradientDescentLinR.m
% Version:   2.0
% Date:      May 22, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: training set matrix (design matrix)
%   - y: vector (column) of correct results (expected results)
%   - theta: initial vector (column) of predicted parameters
%   - alpha: learning rate
%   - nIter: number of iterations of gradient descent
%
% OUTPUT:
%   - theta: improvced learning parameters after gradient descent
%   - J_history: cost function result of each iteration
%
% FUNCTIONALITY:
%   It computes nIter iterations of gradient descent algorithm of
%   linear regression. The new theta vector and the J history
%   is given as result.
function [theta, J_history] = gradientDescentLinR(X, y, theta, alpha, nIter)
    m = length(y);
    n = length(theta);
    J_history = zeros(nIter, 1);

    % === Vectorized implementation: much faster === %
    for k = 1:nIter
        delta = (1/m)*(X'*X*theta-X'*y);
        theta=theta-alpha.*delta;
        J_history(k) = costFunctionLinR(X,y,theta);
    end

    % === Without vectorization: more intuitive, slower === %
%     for k = 1:nIter
%         actualize = ones(n,1);
%         for j = 1:n
%             sumat = 0;
%             for i = 1:m
%                 sumat = sumat + (hypothesis_i(X(i,:), theta) - y(i))*X(i,j);
%             end
%             sumat = alpha*(1/m)*sumat;
%             actualize(j) = theta(j) - sumat;
%         end
%         theta = actualize;
%         J_history(k) = costFunctionLinR(X,y,theta);
%     end
end
