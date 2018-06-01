% =========================================== %
%
% Project:   Machine Learning utilities
% File:      predictionOneVsAll.m
% Version:   2.0
% Date:      May 27, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - X: data matrix. This is not the training set, this is the data
%        to be evaluated and get its classfier
%   - all_theta: fitting parameters for each classifier
%
% OUTPUT:
%   - P: prediction vector
%
% FUNCTIONALITY:
%   It returns a vector of indexes, where the i-th value is the 
%   (better) class for the data of the i-th column
function [P] = predictionOneVsAll(all_theta, X)
    m = size(X,1);
    P = zeros(m,1);
    X = [ones(m,1) X];
    H = sigmoid(X * all_theta');
    % pVal returns the highest value of each row, while
    % P returns the position of the highest value of each row 
    [pVal, P] = max(H, [], 2);
end