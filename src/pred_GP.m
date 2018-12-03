function f = pred_GP(X_pred, y, K, X, sig, ells, beta)
%This function returns the predictive mean for a test point, given the
%covariance function, data, and MAP estimate of the utilities
%
%Inputs
%X_pred: t x d matrix where t is the number of test points and d is the
%number of dimensions
%y: n x 1 vector where n is the number of sample points
%X: n x d data matrix of distinct points
%sig: sigma hyperparameter in squared exponential kernel
%ells: d x 1 or 1 x d vector of length scales in squared exponential
%kernel
%beta: d x 1 vector of coefficients in linear prior
%
%Outputs
%f: t x 1 vector of predictions

k_pred = gram_matrix(X, X_pred, sig, ells);

%Calculate prediction (by solving a linear system rather than finding
%inv(K) directly)
L = chol(K, 'lower');
f = X_pred*beta + k_pred'*(L'\(L\(y - X*beta)));

function K = gram_matrix(X1, X2, sig, ells)
%efficiently computes the gram matrix using vectorisation
%an order of magnitude faster than using nested loops

m = size(X1, 1);
n = size(X2, 1);

Lambda = diag(ells.^2);
K = sig^2*exp(-0.5*(repmat(diag(X1*Lambda^-1*X1'), 1, n) - 2*X1*Lambda^-1*X2' + repmat(diag(X2*Lambda^-1*X2')', m, 1)));

end

end