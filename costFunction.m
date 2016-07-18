function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% get "theta transpose X" by just doing X * theta (hypothesis is a m x 1 matrix)
hypothesis = sigmoid(X*theta);
J = (-1)/m .* ((log(hypothesis)' * y) + (log(1-hypothesis)' * (1-y)));

% get the gradient (needs to be (n+1) x 1 matrix like theta)
% (hypothesis - y) is m x 1 and X' is (n+1) x m
grad = 1/m .* (X' * (hypothesis - y));

% =============================================================

end
