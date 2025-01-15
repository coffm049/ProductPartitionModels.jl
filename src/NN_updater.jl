# using Distributions
# using LinearAlgebra
# using Random
# using Plots


# sample the posterior mean for a normal normal model
# sample center for 
function baseCenterSampler(X)
  μ = mean(X,  dims = 2)
  
  # assume
  # Uniform prior for μ
  # Uniform prior for log(σ^2)
  (p, n) = size(X)
  
  # posterior
  σ2 = [rand(InverseGamma(n/2, abs(sum((X[i,:] .- μ[i]).^2)/2) / 5)) for i in 1:p]
  μ = rand(MvNormal(μ[:], σ2 .* I(p) ./n))
  return μ, σ2 .* I(p)
end

# # concatenate all of the betas across lik_params[j] into a matrix
# Betas = [model.state.lik_params[j].beta for j in 1:2]
# # convert the vector of vectos to a matrix (p x K)
# Betas = hcat(Betas...)
# 
# test, test2 = baseCenterSampler(Betas)

function n_niw_sampler(X, mu0, kappa0, nu0, S0, nsamples)
    n, p = size(X)  # Number of observations (n) and dimensions (p)
    X_bar = mean(X, dims=1)'  # Sample mean (column vector)
    T_n = (X .- X_bar')' * (X .- X_bar')  # Sample scatter matrix

    # Posterior hyperparameters
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * X_bar) / kappa_n
    nu_n = nu0 + n
    S_n = S0 + T_n + (kappa0 * n / kappa_n) * (X_bar - mu0) * (X_bar - mu0)'

    # Storage for samples
    mu_samples = Vector{Vector{Float64}}(undef, nsamples)
    sigma_samples = Vector{Matrix{Float64}}(undef, nsamples)

    for i in 1:nsamples
        # Sample Σ from Inverse-Wishart
        sigma_samples[i] = rand(InverseWishart(nu_n, S_n))

        # Sample μ from Multivariate Normal
        mu_samples[i] = rand(MvNormal(mu_n[:, 1], Σ / kappa_n))
    end

    return mu_samples, sigma_samples
end


# Simulated data
Random.seed!(123)
n, p = 50, 3
true_mu = [1.0, 2.0, 3.0]
true_sigma = [1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0]
X = rand(MvNormal(true_mu, true_sigma), n)'

# Prior hyperparameters
mu0 = [0.0, 0.0, 0.0]
kappa0 = 1.0
nu0 = p + 2  # Must be > p - 1
S0 = 0.1 * I(p)

# Number of posterior samples
nsamples = 1000

# Run the sampler
mu_samples, sigma_samples = n_niw_sampler(X, mu0, kappa0, nu0, S0, nsamples)

# Inspect results
println("First sampled μ: ", mu_samples[:, 1])
println("First sampled Σ: ", sigma_samples[1])

plot([it[1] for it in mu_samples], label = "mu1")
plot!(true_mu[1] * ones(nsamples), label = nothing, color = "black")
plot!([it[2] for it in mu_samples], label = "mu2")
plot!(true_mu[2] * ones(nsamples), label = nothing, color = "black")
plot!([it[3] for it in mu_samples], label = "mu3")
plot!(true_mu[3] * ones(nsamples), label = nothing, color = "black")

