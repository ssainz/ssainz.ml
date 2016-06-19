function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01 0.03 0.1 0.3 1 3 10];
Sigma_vec = [0.01 0.03 0.1 0.3 1 3 10];
prediction_error = 100;
firstIteration = 1;

% ssainz: we need to find the best C and Sigma. Therefore we just go through all combinations... BRUTE FORCE STYLE!
for i = 1:length(C_vec)
  for j = 1:length(Sigma_vec)
    c_temp = C_vec(i);
    sigma_temp = Sigma_vec(j);
    %ssainz: we first train the model using the training set (X, y) and the designation of sigma and C 
    model = svmTrain(X, y, c_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
    %ssainz: we then use the model to predict on the cross validation set (Xval)
    predictions = svmPredict(model, Xval);
    %ssainz: we check how well the model did based on the amount of error 
    prediction_error_temp = mean(double(predictions ~= yval));
    if prediction_error_temp < prediction_error || firstIteration == 1
      firstIteration = 0;
      C = c_temp;
      sigma = sigma_temp;
      prediction_error = prediction_error_temp;
    end  
end





% =========================================================================

end
