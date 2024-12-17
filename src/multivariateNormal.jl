# using LinearAlgebra
# using Random
# using Distributions
# using ProductPartitionModels

export sampleMultiNorm;

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
