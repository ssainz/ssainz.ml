function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
n = size(x1,1);


% Scaling
%minim = min(x1);
%avg = mean(x1);
%maxim = max(x1);
%x1 = (x1 .- avg) ./ (maxim - minim);

%minim = min(x2);
%avg = mean(x2);
%maxim = max(x2);
%x2 = (x2 .- avg) ./ (maxim - minim);

% Normalization
%avg = mean(x1);
%stdDev = std(x1);
%x1 = x1 .- avg;
%x1 = x1 ./ stdDev;

%avg = mean(x2);
%stdDev = std(x2);
%x2 = x2 .- avg;
%x2 = x2 ./ stdDev;
%x1 = x1(2:n);
%x2 = x2(2:n);

%sigma = sigma * 4;

x1;
x2;
sim = (x1 .- x2);
sim = power(sim,2);
sim = sum(sim);
denominator = (2 * power(sigma,2)) ;
sim = -1 * sim / denominator;
sim = exp(sim);




% =============================================================
    
end
