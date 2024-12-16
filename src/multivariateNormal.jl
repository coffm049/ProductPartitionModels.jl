# using LinearAlgebra
# using Random
# using Distributions
# 
# # Simulated data
# n = 100  # Number of samples
# p = 3    # Number of predictors
# q = 2    # Number of responses
# 
# X = randn(n, p)  # Design matrix
# true_beta = [2.0 1.5; -1.0 0.5; 1.0 -1.5]  # True coefficients (p x q)
# true_sigma = [1.0 0.3; 0.3 1.0]  # True covariance matrix (q x q)
# Y = X * true_beta + rand(MvNormal(zeros(q), true_sigma), n)'

function sampleMultiNorm(X, Y, nu)
  (n, p) = size(X)
  (n, q) = size(Y)

  # Conjugate priors
  M = zeros(p, q)  # Mean matrix for the prior on beta
  V = 10 * I(p)    # Column covariance matrix for beta
  nu = q + 2       # Degrees of freedom for Inverse Wishart
  S = I(q)         # Scale matrix for Inverse Wishart
  
  V_post = inv(inv(V) + X'X)  # Updated column covariance for beta
  M_post = V_post * (inv(V) * M + X'Y)  # Updated mean for beta
  nu_post = nu + n  # Updated degrees of freedom
  S_post = S + Y' * Y + M' * inv(V) * M - M_post' * inv(V_post) * M_post  # Updated scale
  
  # Sample from posterior
  sigma = rand(InverseWishart(nu_post, S_post))
  beta = rand(MatrixNormal(M_post, V_post, sigma))
  return beta, sigma
end

test2 = sampleMultiNorm(X, Y, 3)
