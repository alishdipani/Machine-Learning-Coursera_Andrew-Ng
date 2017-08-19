function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = (X) * (theta);
%disp(h);
k = (h) - (y);
%disp(k);
k1 = (k) .* (k);
%disp(k1);
ksum = sum(k1(:));
%disp(ksum);
j1 = (1/(2*m));
J = j1 * ksum;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
