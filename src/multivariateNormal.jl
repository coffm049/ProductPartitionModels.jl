using LinearAlgebra
using Random
using Distributions
using ProductPartitionModels

# Simulated data
n = 100  # Number of samples
p = 3    # Number of predictors
q = 2    # Number of responses

X = randn(n, p)  # Design matrix
true_beta = [2.0 1.5; -1.0 0.5; 1.0 -1.5]  # True coefficients (p x q)
true_sigma = [1.0 0.3; 0.3 1.0]  # True covariance matrix (q x q)
Y = X * true_beta + rand(MvNormal(zeros(q), true_sigma), n)'

function sampleMultiNorm(X, Y)
  (n, p) = size(X)
  (n, q) = size(Y)

  # Conjugate priors
  M = zeros(p, q)  # Mean matrix for the prior on beta
  V = Hermitian(I(p))    # Column covariance matrix for beta
  nu = q       # Degrees of freedom for Inverse Wishart
  S = Hermitian(I(q))         # Scale matrix for Inverse Wishart
  
  V_post = Hermitian(inv(inv(V) + X'X))  # Updated column covariance for beta
  M_post = V_post * (inv(V) * M + X'Y)  # Updated mean for beta
  nu_post = nu + n  # Updated degrees of freedom
  S_post = Hermitian(S + Y' * Y + M' * inv(V) * M - M_post' * inv(V_post) * M_post)  # Updated scale
  
  # Sample from posterior
  sigma = rand(InverseWishart(nu_post, Matrix(S_post)))
  beta = rand(MatrixNormal(M_post, Matrix(V_post), Matrix(sigma)))
  return beta, sigma
end

test2 = sampleMultiNorm(X, Y)
model = Model_PPMx(Y, X, 1, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
sim = mcmc!(model, 1000)
