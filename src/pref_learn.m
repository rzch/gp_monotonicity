function [f_MLE, y, f_MAP, alpha, K, sig, ells, beta_star_normalised] = pref_learn(X, xA, xB, indA, indB)
%This function returns the variables required to conduct predictions using
%Gaussian processes
%
%Inputs
%X: n x d data matrix of distinct points
%xA: d x M data of comparisons with not preferred items
%xB: d x M data of comparisons with preferred items
%indA: 1 x M vector of indices in {1, ..., n} corresponding to each item in
%xA
%indB: 1 x M vector of indices in {1, ..., n} corresponding to each item in
%xB
%
%Outputs
%f_MLE: n x 1 latent utility vector from MLE
%y: n x 1 latent utility vector weighted between MLE and MAP
%f_MAP: n x 1 latent utility vector from MAP with linear prior
%alpha: weighting factor
%K: n x n Gram matrix from pairs in X
%sig: sigma hyperparameter used in squared exponential kernel
%ells: 1 x d length scale hyperparameters used in squared exponential
%kernel
%beta_star_normalised: slope coefficients in linear prior

gridsize = 5;
options = optimoptions(@fmincon, 'MaxFunctionEvaluations', 20000, 'Display', 'off');

%initial value for MLE
beta0 = [0.5; 0.5];
sig_noise = 1/sqrt(2);
%anonymous function to call negative log likelihood
cost = @(beta) negLogL(beta, xA, xB);

%optimise parameters using maximum likelihood
beta_star = fmincon(cost, beta0, -eye(2), [eps; eps], [], [], [], [], [], options);
%normalised parameters
beta_star_normalised = beta_star/sum(beta_star);
sig_noise = sig_noise/sum(beta_star);

%data matrix made from all pairs
n = size(X, 1);
d = size(xA, 1);

%compute and plot MLE surface
f_MLE = X*beta_star_normalised;

%GP hyperparameters
sig = sig_noise;
ells = ones(1, d)./gridsize;

%construct covariance matrix
K = gram_matrix(X, X, sig, ells);

%anonymous function to call negative log posterior
cost = @(f) negLogPosterior(f, K, sig_noise, X, beta_star_normalised, indA, indB);

%optimise utility values using MAP
f_MAP = fmincon(cost, zeros(n, 1), [], [], [], [], [], [], [], options); %for affine prior

%% Enforce monotonicity

%grid to check monotonicity over
[X1, X2] = meshgrid(linspace(0, 1), linspace(0, 1));

%find infimum gradient over grid
ncheck = numel(X1);
xcheck = [reshape(X1, ncheck, 1), reshape(X2, ncheck, 1)];

gradcheck = gradPosteriorMean(xcheck, f_MAP, K, X, sig, ells, beta_star_normalised);
[infimum1, index1] = min(gradcheck(:, 1));
[infimum2, index2] = min(gradcheck(:, 2));

minimiser1 = xcheck(index1, :)';
minimiser2 = xcheck(index2, :)';

grad1 = @(x) [1 0]*gradPosteriorMean(x', f_MAP, K, X, sig, ells, beta_star_normalised)';
grad2 = @(x) [0 1]*gradPosteriorMean(x', f_MAP, K, X, sig, ells, beta_star_normalised)';

%Inequality constraints
Aineq = [eye(2); -eye(2)];
Bineq = [1; 1; 0; 0];
%Local search to refine the infimums
[minimiser1, infimum1] = fmincon(grad1, minimiser1, Aineq, Bineq, [], [], [], [], [], options);
[minimiser2, infimum2] = fmincon(grad2, minimiser2, Aineq, Bineq, [], [], [], [], [], options);


alpha_star = max([beta_star_normalised(1)/(-beta_star_normalised(1) + min(0, infimum1));
    beta_star_normalised(2)/(-beta_star_normalised(2) + min(0, infimum2))]) + 1;

epsilon = 0.01;
alpha = min(alpha_star + epsilon, 1);
y = alpha*f_MLE + (1 - alpha)*f_MAP;

%%

%negative log likelihood function for affine
function J = negLogL(beta, xA, xB)

%sum the log-likelihoods
J = -sum(log(normcdf(beta'*xB - beta'*xA)));

end

%negative log likelihood function
function J = negLogLikelihood(f, sig_noise, indA, indB)

%sum the log-likelihoods
J = -sum(log(normcdf((f(indB) - f(indA))/(sqrt(2)*sig_noise))));

end

%negative log posterior function
function J = negLogPosterior(f, K, sig_noise, X, beta, indA, indB)

%sum the log-likelihoods and add the log-prior
L = chol(K, 'lower');
J = negLogLikelihood(f, sig_noise, indA, indB) + 0.5*(f - X*beta)'*(L'\(L\(f - X*beta)));

end

%gradient of the posterior mean
function g = gradPosteriorMean(X_pred, y, K, X, sig, ells, beta)

t = size(X_pred, 1);
n = size(X, 1);
d = size(X, 2);

Lambda = diag(ells.^2);
invLambda = Lambda^-1;

k_vec = gram_matrix(X, X_pred, sig, ells);

L = chol(K, 'lower');
invKz = L'\(L\(y - X*beta));

g = zeros(t, d);
for tt = 1:t
   g(tt, :) = (beta - invLambda*(((repmat(X_pred(tt, :)', 1, n) - X').*k_vec(:, tt)')*invKz))'; 
end

end

end

function K = gram_matrix(X1, X2, sig, ells)
%efficiently computes the gram matrix using vectorisation
%an order of magnitude faster than using nested loops

m = size(X1, 1);
n = size(X2, 1);

Lambda = diag(ells.^2);
K = sig^2*exp(-0.5*(repmat(diag(X1*Lambda^-1*X1'), 1, n) - 2*X1*Lambda^-1*X2' + repmat(diag(X2*Lambda^-1*X2')', m, 1)));

end