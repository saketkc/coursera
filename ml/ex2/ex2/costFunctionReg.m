function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some 
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X*theta);
reg_theta = theta(2:end,:);
lambda_term = (lambda/(2*m))*(reg_theta'*reg_theta);
cost = y'*log(h_theta)+(1-y')*(log(1-h_theta));

J = (-1/m)*cost + lambda_term;
reg_lambda_term =  (lambda/m)*theta;
reg_lambda_term(1) =0;
normal_term = (1/m)*(X'*(h_theta-y));
grad = reg_lambda_term + normal_term;




% =============================================================

end
