% =========================================== %
%
% Project:   Machine Learning utilities
% File:      sigmoid.m
% Version:   2.0
% Date:      May 23, 2018
%
% (C) Alejandro Santorum Varela
%     alejandro.santorum@gmail.com
%
% =========================================== %


% INPUT:
%   - z: number, vector or matrix which the function is going
%        to be applied to.
%
% OUTPUT:
%   - g: result of the sigmoid function
%
% FUNCTIONALITY:
%   It returns the result of evaluating z in the sigmoid function.
%   Remember --> g(z) = 1/(1+exp(-z)
function g = sigmoid(z)
    g = 1./(1+exp(-z));
end