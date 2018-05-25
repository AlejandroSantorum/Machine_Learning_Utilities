% =========================================== %
%
% Project:   Machine Learning utilities
% File:      gradientDescentLinR_Reg.m
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
%   - alpha: learning rate
%   - lambda: regularization parameter
%   - nIter: number of iterations of gradient descent
%
% OUTPUT:
%   - theta: improved learning parameters after gradient descent
%   - J_history: cost function result of each iteration
%
% FUNCTIONALITY:
%   It computes nIter iterations of gradient descent algorithm of
%   linear regression. The new theta vector and the J history are
%   given as result. It uses regularization method in order to avoid
%   overfitting.
function [theta, J_history] = gradientDescentLinR_Reg(X, y, theta, alpha, lambda, nIter)
    m = length(y);
    n = length(theta);
    J_history = zeros(nIter, 1);

    % === Vectorized implementation: much faster === %
    for k = 1:nIter
        delta = (1/m)*((X'*X*theta-X'*y)+sum(theta.^2));
        theta=theta-alpha.*delta;
        J_history(k) = costFunctionLinR_Reg(X,y,theta);
    end

    % === Without vectorization: more intuitive, slower === %
%     for k = 1:nIter
%         actualize = ones(n,1);
%         for j = 1:n
%             sumat = 0;
%             for i = 1:m
%                 sumat = sumat + (hypothesis_i(X(i,:), theta) - y(i))*X(i,j)+(lambda/m)*theta(j);
%             end
%             sumat = alpha*(1/m)*sumat;
%             actualize(j) = theta(j) - sumat;
%         end
%         theta = actualize;
%         J_history(k) = costFunctionLinR_Reg(X,y,theta);
%     end
end