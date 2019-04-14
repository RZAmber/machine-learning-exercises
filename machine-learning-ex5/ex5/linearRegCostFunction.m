function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute cost function J. used for part2 of ex5.m
% theta = [1 ; 1]; size 2*1; X size 12*2; y size 12*1
L = lambda / (2 * m) * [theta(2:end)]' * theta(2:end);
J = 1 / (2 * m) * (theta' * X' - y') * (X * theta - y) + L;

% Compute gradient. used for part3 of ex5.m
grad(1) = 1 / m * X(:,1)' * (X * theta - y);
grad(2:end) = 1 / m * (X(:,2:end))' * (X * theta - y) + lambda / m * theta(2:end);






% =========================================================================

grad = grad(:);

end
