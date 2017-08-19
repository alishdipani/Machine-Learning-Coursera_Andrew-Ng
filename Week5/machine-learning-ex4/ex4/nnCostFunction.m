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

yi = zeros(size(y,1),num_labels);
for i=1:m,
	yi(i,y(i,1))=1;
end

%disp("size of X,y,yi is ")
%size(X)
%size(y)
%size(yi)

%disp("size of theta1 is")
%size(Theta1)
%disp("size of theta2 is")
%size(Theta2)

%for i = 1:num_labels,
a1 = [ones(size(X,1),1) X];
x2 = sigmoid(a1 * Theta1');
a2 = [ones(size(X,1),1) x2];
h = sigmoid(a2 * Theta2');
	%maxv = zeros(size(y));
	%p = zeros(size(y));
	%[maxv, p] == max(h,[],2);
	%J = J +  (  (-1/m) * ( sum((yi .* log(h'))(:)) + sum(( (1-yi).* log(1-h') )(:)) )  );
	%pii = (p==i);
J = J +  (  (-1/m) * ( sum((yi .* log(h))(:)) + sum(( (1-yi).* log(1-h) )(:)) )  );	

temp1t = Theta1 .* Theta1;
temp1 = temp1t(:,2:(input_layer_size+1));

temp2t = Theta2 .* Theta2; 
temp2 = temp2t(:,2:(hidden_layer_size + 1));
J = J + (lambda/(2*m))*(sum(temp1(:)) + sum(temp2(:)));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

D2 = zeros(size(Theta1));
D3 = zeros(size(Theta2));

for i = 1:m,
	a1i = X(i,:)' ;
	yi = zeros(num_labels,1);
	yi(y(i,1),1)=1;
	a2i = sigmoid(Theta1*[1; a1i]);
	hi = sigmoid(Theta2*[1; a2i]);	
	d3i = hi - yi;
	d2i = (Theta2' * d3i) .* ([1; a2i] .* (1 - [1; a2i]));
	d2i = d2i(2:end,1);
	D3 = D3 + (d3i * [1; a2i]'); 
	D2 = D2 + (d2i * [1; a1i]');
end;

t1 = Theta1;
t1(:,1) = zeros(size(Theta1,1),1);
t2 = Theta2;
t2(:,1) = zeros(size(Theta2,1),1);

Theta1_grad = (1/m)*(D2) + ((lambda/m)*t1);
%size(Theta1_grad)
Theta2_grad = (1/m)*(D3) + ((lambda/m)*t2);
%size(Theta2_grad)
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%size(grad)

%end
%disp("size of a1 is")
%size(a1)
%disp("size of a2 is")
%size(a2)
%disp("size of h is")
%size(h)

