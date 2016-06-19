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

h_theta = X * theta;
%h_theta
h_theta = sigmoid(h_theta);
one_minus_h_theta = 1 .- h_theta;
lg_h = log(h_theta);
lg_one_minus_h = log(one_minus_h_theta);
positive_side = (-1 .* y) .* lg_h;
negative_side = (1 .- y) .* lg_one_minus_h;
compressed_form = positive_side - negative_side;
J = (ones(1, m) * compressed_form) / m;

% Now calculates the gradients 

sigma_h_theta_minus_y_times_xj =  X .* (h_theta - y)  ;
compress_into_1_row_n_columns_matrix = ones(1,m) * sigma_h_theta_minus_y_times_xj;
grad = compress_into_1_row_n_columns_matrix' ./ m;









% =============================================================

end
