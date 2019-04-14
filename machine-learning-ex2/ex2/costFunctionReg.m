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
J1=0;
J2=0;
[n,n1]=size(theta);
A=X*theta;
for i =1 :m,
    h=1/(1+exp(-A(i)));
    E=-y(i)*log(h)-(1-y(i))*log(1-h);
    J1=J1+E;
    grad(1)=grad(1)+(h-y(i))*X(i,1);
    for k=2:n,
        grad(k)=grad(k)+(h-y(i))*X(i,k);
    end
end
grad=grad/m;
for j=2:n,
    J2=J2+theta(j)*theta(j);
    grad(j)=grad(j)+lambda/m*theta(j);
    end 
J=J1/m+J2/m*lambda/2;


% =============================================================

end
