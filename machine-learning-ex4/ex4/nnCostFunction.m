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
%Theta2 = reshape(nn_params((num_labels * (hidden_layer_size + 1)):end), ...
%                 num_labels, (hidden_layer_size + 1));



% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
         
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

% Compute h_theta first
% X is matrix m x n (or m x input_layer_size) 
% theta1 is matrix size hidden_layer_size X input_layer_size

X = [ones(m, 1) X];

z_2 = X * Theta1';
%'
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];

% a_2 is dimension m x (hidden_layer_size + 1)
% Theta2 is dimension num_labels x (hidden_layer_size + 1)
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

%a_3 is dimension m x num_labels
h_theta = a_3;

%creating the y arrays
y_arrays = zeros(m, num_labels);
iterations = 1:m ;
for iter = iterations
  y_arrays(iter, y(iter)) = 1; 
end

%now we calculate the cost

one_minus_h = 1 .- h_theta;
negative_y = y_arrays .* -1;
one_minus_y = 1 .- y_arrays;
innerLoopTerm = (negative_y .* log(h_theta)) .- (one_minus_y .* log(one_minus_h));
% innerLoopTerm dimension is m x num_labels;
% we now sum across the num_labels
sumAcrossLabels = ones(num_labels, 1);
innerSummatory = innerLoopTerm * sumAcrossLabels;
%innerSummatory dimension is m * 1
sumAcrossTrainingExamples = ones(1, m);
outerSum = sumAcrossTrainingExamples * innerSummatory;
outerSum = (1 / m ) * outerSum;
J = outerSum;




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

%will need the theta2 minus its first column related to the bias parameter 
theta2_deleted_first_col = Theta2;
theta2_deleted_first_col(:,[1]) = [];


% d_3 dimension is 1 * num_labels
% d_2 dimension is 1 * hidden_layer_size
% a_1 dimension is 1 * (input_layer_size + 1)
% a_2 dimension is 1 * (hidden_layer_size + 1)
delta_l_1 = zeros(hidden_layer_size, input_layer_size + 1);
delta_l_2 = zeros(num_labels, hidden_layer_size + 1);


for t = 1:m

  %we do the forward propagation first for each training at a time:

  a_1 = X(t,:);
  %X already has the bias unit added, so we do not need to add again.
  z_2 = a_1 * Theta1';
  %'
  a_2 = sigmoid(z_2);
  a_2 = [1 a_2];
  z_3 = a_2 * Theta2';
  %'
  a_3 = sigmoid(z_3);
  y_array = zeros(1, num_labels);
  y_array(y(t)) = 1;
  d_3 = a_3 .- y_array;

  %print "size of z_2"
  %size(z_2)
  %print "size of (theta2_deleted_first_col T * d_3 T )T "
  %size((theta2_deleted_first_col' * d_3' )')

  d_2 = (theta2_deleted_first_col' * d_3' )' .* sigmoidGradient(z_2);
  % already removed that column in theta2_deleted_first_col, no need to remove the first d_2 
  % d_2 = d_2(2:end);

  %now fill the deltas 
  delta_l_1 = delta_l_1 .+ (d_2' * a_1);
  %'
  delta_l_2 = delta_l_2 .+ (d_3' * a_2);
  %'

end

delta_l_1 = (1/m) .* delta_l_1;
delta_l_2 = (1/m) .* delta_l_2;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% now we do regularization for cost function
% first modify the theta1 and theta2 so that its first columns (bias term) is zeros

theta1_minus_first_col = Theta1;
theta1_minus_first_col(:,1) = 0;
theta2_minus_first_col = Theta2;
theta2_minus_first_col(:,1) = 0;
theta1_sq = theta1_minus_first_col .* theta1_minus_first_col;
theta2_sq = theta2_minus_first_col .* theta2_minus_first_col;
sumThetas = sum(sum(theta1_sq)) + sum(sum(theta2_sq));
regu = (lambda / (2*m) ) * sumThetas;
J = J + regu;


% and we now do regularization for the gradients :D 
% we already have the thetas without the first column (that corresponds to the bias element)
% we can use those to calculate gradients

theta1_times_lambda_divided_by_m = theta1_minus_first_col .* (lambda / m);
theta2_times_lambda_divided_by_m = theta2_minus_first_col .* (lambda / m);
delta_l_1 = delta_l_1 .+ theta1_times_lambda_divided_by_m;
delta_l_2 = delta_l_2 .+ theta2_times_lambda_divided_by_m;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [delta_l_1(:); delta_l_2(:)];


end
