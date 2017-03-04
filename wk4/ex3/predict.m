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
% s2 is the number of nodes at the hidden layer
s2 = 25;

% s3 is the number of nodes at the output layer
s3 = num_labels;

% res is a matrix to hold predictions of each class for each input
res = zeros(m, num_labels);

% Add bias units to inputs X
X = [ones(m, 1) X];

% Use for loop to compute predictions for each input
for i = 1:m
    % take one input: 1 * (n+1)
    a1 = X(i, :);   
    % compute hidden layer features a2: s2 * 1
    a2 = sigmoid(Theta1 * a1');
    % add ones to a2: (s2 + 1) * 1
    a2 = [1 ; a2];
    % compute output layer a3: s3 * 1
    a3 = sigmoid(Theta2 * a2);
    % store results in matrix res
    res(i, :) = a3';
end

% p can be computed from res, p: m * 1
[Y p] = max(res, [], 2);

% =========================================================================


end
