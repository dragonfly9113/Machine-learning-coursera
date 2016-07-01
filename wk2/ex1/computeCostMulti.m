function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% predictions is m * 1 vector which is h_theta(x): predicted value using current theta parameters
% for multi variables, X is m * (n+1) and theta is (n+1) * 1
predictions = X * theta;	

% use the predictions to compute J(theta)
% instead of using .^ 2 like in computeCost(), we use a vectorized form like below:
J = (1 / (2*m)) * (predictions - y)' * (predictions - y);


% =========================================================================

end
