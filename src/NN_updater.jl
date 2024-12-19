using Distributions
using LinearAlgebra


# sample the posterior mean for a normal normal model
# sample center for 
function baseCenterSampler(X)
  μ = mean(X,  dims = 2)
  
  # assume
  # Uniform prior for μ
  # Uniform prior for log(σ^2)
  (p, n) = size(X)
  
  # posterior
  σ2 = [rand(InverseGamma(n/2, sum((X[:,i] .- μ[i]).^2)/2)) for i in 1:p]
  μ = rand(MvNormal(μ[:], σ2 .* I(p) ./n))
  return μ, σ2 .* I(p)
end

X = rand(MatrixNormal(3, 10))

test, test2 = baseCenterSampler(X)
