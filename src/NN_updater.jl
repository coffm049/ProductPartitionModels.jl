# using Distributions
# using LinearAlgebra


# sample the posterior mean for a normal normal model
# sample center for 
function baseCenterSampler(X)
  μ = mean(X,  dims = 2)
  
  # assume
  # Uniform prior for μ
  # Uniform prior for log(σ^2)
  (p, n) = size(X)
  
  # posterior
  σ2 = [rand(InverseGamma(n/2, abs(sum((X[i,:] .- μ[i]).^2)/2) / 100)) for i in 1:p]
  μ = rand(MvNormal(μ[:], σ2 .* I(p) ./n))
  return μ, σ2 .* I(p)
end

# # concatenate all of the betas across lik_params[j] into a matrix
# Betas = [model.state.lik_params[j].beta for j in 1:2]
# # convert the vector of vectos to a matrix (p x K)
# Betas = hcat(Betas...)
# 
# test, test2 = baseCenterSampler(Betas)
