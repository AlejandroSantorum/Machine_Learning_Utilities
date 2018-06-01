% =========================================== %
%
% Project:   Machine Learning utilities
% File:      costGradientLogR.m
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
%   - y: expected values
%
% OUTPUT:
%   - J: cost function for logistic regression
%   - gradient: partial derivates to compute the gradient descent
%
% FUNCTIONALITY:
%   It returns the code to compute the cost function and the partial
%   derivatives in order to get the gradient descent. This function
%   is designed to be used as a parameter of another optimazed algorithm,
%   like for example BFGS or L-BFGS. (In matlab --> fminunc(...))
function [J, gradient] = costGradientLogR(theta, X, y)
    m = length(y);
    n = length(theta);
    J = 0;
    gradient = zeros(size(theta));
    
    % === VECTORIZED IMPLEMENTATION === %
    h = hypothesis(X, theta);
    g = sigmoid(h);
    auxiliar = y'*log(g) + (1-y)'*log(1-g);
    J = (-1/m)*sum(auxiliar); % sum() works for columns
    
    gradient = (1/m)*((X'*g-X'*y)');
    % ================================= %
    
    % === UNVECTORIZED IMPLEMENTATION === %
%     sumat = 0;
%     for i = 1:m
%         sumat = sumat + (y(i)*log(sigmoid(hypothesis_i(X(i,:),theta)))+(1-y(i))*log(1-sigmoid(hypothesis_i(X(i,:),theta))));
%     end
%     J = (-1/m)*sumat;
% 
%     for j = 1:n
%         sumat = 0;
%         for i = 1:m
%             sumat = sumat + ((sigmoid(hypothesis_i(X(i,:),theta))-y(i))*X(i,j)); 
%         end
%         gradient(j) = (1/m)*sumat;
%     end
    % =================================== %
end