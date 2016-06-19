function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
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
theta_minus_first = theta(2:length(theta));
theta_sq = theta_minus_first' * theta_minus_first;
%'
regu = (lambda * theta_sq ) / (2 * m);
J = J + regu;


% Now calculates the gradients 

%Simplified version
grad =   ( X' * (h_theta - y) ) ./m ;

%'
%slightly more complex version:
%sigma_h_theta_minus_y_times_xj =  X .* (h_theta - y)  ;
%compress_into_1_row_n_columns_matrix = ones(1,m) * sigma_h_theta_minus_y_times_xj;
%grad = compress_into_1_row_n_columns_matrix' ./ m;

%' Now add regularization component
lambda_div_by_m = zeros(size(theta)) .+ (lambda / m);
% first element does not get regularization.
lambda_div_by_m(1) = 0;
lambda_div_by_m_times_theta = lambda_div_by_m .* theta;
grad = grad .+ lambda_div_by_m_times_theta;



% =============================================================


end
