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
X = [ones(m, 1) X];
% Compute the hidden layer with size m * (l + 1), each row from a example data
a2 = sigmoid(Theta1 * X');
a2 = [ones(size(a2, 2), 1) a2'];
% Compute the output layer with size m * k, each row has k outputs
a3 = sigmoid(Theta2 * a2');
% Transform the true values of y into vectors then into a maxtrix
y_compare = zeros(num_labels, m);
for i = 1:m
  y_compare(y(i), i) = 1;
end

J = sum(sum((- y_compare) .* log(a3) - (1 - y_compare) .* log(1 - a3))) / m;
% Regularization
% Remove the terms that correspond to the bias
Theta1_v = Theta1(:, 2:end);
Theta2_v = Theta2(:, 2:end);
J = J + lambda / 2 / m * (sum(sum(Theta1_v .^ 2)) + sum(sum(Theta2_v .^ 2)));
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
%Backpropagation for each row of data, then accumulate
for t = 1:m
  % Forward propagation
  % Step 1
  a1 = X(t, :); % 1x401
  z2 = Theta1 * a1'; % 25x401 x 401x1 = 25x1
  a2 = [1; sigmoid(z2)]; % 26x1
  z3 = Theta2 * a2; % 10x26 x 26x1 = 10x1
  a3 = sigmoid(z3);
  % Backpropagation
  % Step 2
  delta3 = a3 - y_compare(:, t); % 10x1
  % Step 3
  z2_back = [1; z2]; % 26x1
  delta2 = Theta2' * delta3 .* sigmoidGradient(z2_back); % 26x1
  % Step 4
  Theta2_grad = Theta2_grad + delta3 * a2'; % 10x26
  Theta1_grad = Theta1_grad + delta2(2:end) * a1; % 25x401  
end

% Step5
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Regularization
Theta1_grad = Theta1_grad + [zeros(size(Theta1, 1), 1) (lambda / m * Theta1(:, 2:end))];
Theta2_grad = Theta2_grad + [zeros(size(Theta2, 1), 1) (lambda / m * Theta2(:, 2:end))];


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
