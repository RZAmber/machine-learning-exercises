function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%% part 1: calculate cost function. used for part3 of ex4.m
%Theta1 25*401
%Theta2 10*26

Y = zeros(m, num_labels);
for i=1:m
  Y(i, y(i)) = 1;
end

a1 = [ones(1,m); X'];
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1, size(a2,2)); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
a3 = a3';

for i=1:m
  for k=1:num_labels
    J = J - Y(i,k) * log(a3(i,k)) - (1 - Y(i,k)) * log(1 - a3(i,k));
   end
end

J = J / m;

%regularized cost function. used for part4 of ex4.m
L = 0;
[r1, c1] = size(Theta1);
for i=1:r1
  for j=2:c1
    L = L + Theta1(i,j)^2;
  end
end

[r2, c2] = size(Theta2);
for i=1:r2
  for j=2:c2
    L = L + Theta2(i,j)^2;
  end
end

J = J + lambda / (2*m) * L;


%%part 2: calculate theta using Bp. used for part7 of ex4.m

C_delta1 = zeros(size(Theta1));
C_delta2 = zeros(size(Theta2));

for t=1:m
  a1 = [1; X(t,:)']; % 401*1
  %Step1: calculate activation function using FP
  z2 = Theta1 * a1; %25*1
  a2 = [1; sigmoid(z2)]; % 26*1
  z3 = Theta2 * a2; %10*1
  a3 = sigmoid(z3); % 10*1
  
  %Step2: using y get delta L
  delta3 = a3 - Y(t,:)'; %10*1
  
  %Step3: calculate delta2
  z2 = [1; z2];
  delta2 = Theta2' * delta3 .* sigmoidGradient(z2);
  
  %Step4: get capital delta1&2
  C_delta2 = C_delta2 + delta3 * a2';
  delta2 = delta2(2:end);
  C_delta1 = C_delta1 + delta2 * a1';
 
end

% Step5: unregularized gradient for NN
Theta1_grad = C_delta1/m;
Theta2_grad = C_delta2/m;


%%part3: regularized NN. used for part8 of ex4.m

Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;
Theta1_grad(:,1) = Theta1_grad(:,1) - (lambda / m) * Theta1(:, 1);
Theta2_grad(:,1) = Theta2_grad(:,1) - (lambda / m) * Theta2(:, 1);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
