function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%             iterations to execute (useful when debugging in case your
%             code is not converging correctly!)
%             (if unspecified can be set to 1000)
%
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"



if ~exist('epsilon','var')
    % epsilon parameter does not exist, so default it to something
    epsilon = 1e-5;
end

if ~exist('maxiter','var')
    % maxiter parameter does not exist, so default it to something
    maxiter = 1000;
end

n = size(data, 2);
weights = zeros(n, 1);

for i=1:maxiter
    
    % Newton-Raphson method described in the PRML book
    y = sigmf(data * weights, [1,0]);
    
    % Adding 0.01 to prevent singular matrix
    R = diag(y .* (1 - y) + 0.01);
        
    % Update the weight values as specified in (4.99) and (4.100)
    z = (data * weights) - (R^(-1) * (y - labels));
    weights = (data' * R * data)^(-1) * data' * R * z;
    
    y_new = sigmf(data * weights, [1,0]);
    error = mean(abs(y_new - y));
    if error < epsilon
        break;
    end
end



