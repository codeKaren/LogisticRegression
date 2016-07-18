function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% get "theta transpose X" by just doing X * theta (hypothesis is a m x 1 matrix)
hypothesis = sigmoid(X*theta);
originalCost = (-1)/m .* ((log(hypothesis)' * y) + (log(1-hypothesis)' * (1-y)));

% need theta(2:end) because we don't want to regularize theta(1) AKA theta-0 (j = 0)
regularizationTerm = lambda./(2.*m) .* sum(theta(2:end).^2);
J = originalCost + regularizationTerm;


% gradient is original gradient for j = 0, but need additional regularization term for j > 1
% get the gradient without additional term first (needs to be (n+1) x 1 matrix like theta)
% (hypothesis - y) is m x 1 and X' is (n+1) x m
originalGrad = 1/m .* (X' * (hypothesis - y));

% add additional term by first creating a vector to add to grad
regularizationTerm = theta .* lambda./m;
% set first term to 0 (remember that matrices are 1-indexed!)
regularizationTerm(1) = 0;
grad = originalGrad + regularizationTerm;

% =============================================================

end
