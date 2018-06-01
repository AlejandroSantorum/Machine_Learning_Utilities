% =========================================== %
%
% Project:   Machine Learning utilities
% File:      oneVsAll_ClassReg.m
% Version:   2.0
% Date:      May 27, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: training set matrix (design matrix)
%   - y: expected values
%   - num_labels: number of the different sets we want to classify
%   - initial_theta: initial fitting parameters for all of the
%   classifications using logisitic regression algorithm
%   - lambda: regularization parameter
%   - num_iter: number of iterations of gradient descent for each classfier
%
% OUTPUT:
%   - all_theta: matrix where the i-th row corresponds to the classifier
%   for label i. It means all_theta matrix is going to have num_labels
%   rows, one per classifier, and (n+1) columns where n is the number of
%   features
%
% FUNCTIONALITY:
%   It trains num_labels logistic regression regularized classifiers and
%   returns each of these classifiers results in the matrix all_theta,
%   where the i-th row corresponds to the classifier for label i.
%   It addiction, this routine is improved and optimed using fmincg(...)
%   function, very similar to fminunc, but more powerful when we work with
%   a huge number of inputs.
%
% IMPORTANT IMPLEMENTATION NOTE:
%   X, the design or data matrix is suppossed to be a n*m matrix, where n
%   is the number of features and m the number of training examples. This
%   function adds a new column of 1's, so the final dimension is (n+1)*m
function [all_theta] = oneVsAll_ClassReg(X, y, num_labels, initial_theta, lambda, num_iter)
    m = size(X, 1);
    n = size(X, 2);
    
    % Add ones to the X data matrix
    X = [ones(m, 1) X];
    all_theta = zeros(num_labels, n + 1);
    
    options = optimset('GradObj', 'on', 'MaxIter', num_iter);
    
    for k = 1:num_labels
        all_theta(k, :) = fmincg (@(t)(costGradientLogR_Reg(t, X, (y == k), lambda)), initial_theta, options);
    end
end