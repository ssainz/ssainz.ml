function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];


%X is m x n , Theta1 is k x n. K is the number of nodes in the next layer (layer 2)
a_2 = X * Theta1' ;
%
a_2 = sigmoid(a_2);
% now the results are in a_2 ( m x k)
% a_2 has dimensions m x k , we should add the first bias index, as below:
a_2 = [ones(m, 1) a_2];

% multiply the results from the layer 2 a_2 (m x K) by the values of the Theta2 (K2 x K) to calculate layer 3 results. 
% K2 is the number of nodes in the third layer. K is the number of nodes in the second layer
% Theta2 is a k2 x k dimension (k2 = 10, k = 25)
a_3 = a_2 * Theta2';
% We do the sigmoid function. a_3 is (m x K2)
a_3 = sigmoid(a_3);

% we must return the results as p (m x 1) with the index 
[max_value, max_index] = max(a_3, [], 2);

p = max_index;











% =========================================================================


end
