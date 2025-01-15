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
        mu_samples[i] = rand(MvNormal(mu_n[:, 1], sigma_samples[i] / kappa_n))
    end

    return mu_samples, sigma_samples
end


# Simulated data
# Random.seed!(123)
# n, p = 50, 3
# true_mu = [1.0, 2.0, 3.0]
# true_sigma = [1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0]
# X = rand(MvNormal(true_mu, true_sigma), n)'
# 
# # Prior hyperparameters
# mu0 = [0.0, 0.0, 0.0]
# kappa0 = 1.0
# nu0 = p + 2  # Must be > p - 1
# S0 = 0.1 * I(p)
# 
# # Number of posterior samples
# nsamples = 1000
# 
# # Run the sampler
# mu_samples, sigma_samples = n_niw_sampler(X, mu0, kappa0, nu0, S0, nsamples)
# 
# # Inspect results
# println("First sampled μ: ", mu_samples[:, 1])
# println("First sampled Σ: ", sigma_samples[1])
# 
# plot([it[1] for it in mu_samples], label = "mu1")
# plot!(true_mu[1] * ones(nsamples), label = nothing, color = "black")
# plot!([it[2] for it in mu_samples], label = "mu2")
# plot!(true_mu[2] * ones(nsamples), label = nothing, color = "black")
# plot!([it[3] for it in mu_samples], label = "mu3")
# plot!(true_mu[3] * ones(nsamples), label = nothing, color = "black")

function independent_sampler(X, mu0, kappa0, alpha0, beta0, nsamples)
    n, p = size(X)  # Number of observations and dimensions
    X_bar = mean(X, dims=1)  # Sample means for each dimension

    # Storage for samples
    mu_samples = Matrix{Float64}(undef, p, nsamples)
    sigma2_samples = Matrix{Float64}(undef, p, nsamples)

    for j in 1:p
        # Data for the j-th dimension
        xj = X[:, j]

        # Posterior hyperparameters for the j-th dimension
        kappa_n = kappa0[j] + n
        mu_n = (kappa0[j] * mu0[j] + n * X_bar[j]) / kappa_n
        alpha_n = alpha0[j] + n / 2
        beta_n = beta0[j] + 0.5 * sum((xj .- X_bar[j]).^2) +
                 (kappa0[j] * n * (X_bar[j] - mu0[j])^2) / (2 * kappa_n)

        for i in 1:nsamples
            # Sample σ_j^2 from Inverse-Gamma
            sigma2 = rand(InverseGamma(alpha_n, beta_n))
            sigma2_samples[j, i] = sigma2

            # Sample μ_j from Normal
            mu = rand(Normal(mu_n, sqrt(sigma2 / kappa_n)))
            mu_samples[j, i] = mu
        end
    end

    return mu_samples, sigma2_samples
end


# Simulated data
Random.seed!(123)
n, p = 50, 3
true_mu = [1.0, 2.0, 3.0]
true_sigma2 = [1.0, 0.5, 0.2]
X = hcat([rand(Normal(true_mu[j], sqrt(true_sigma2[j])), n) for j in 1:p]...)

# Prior hyperparameters
mu0 = [0.0, 0.0, 0.0]  # Prior mean
kappa0 = [1.0, 1.0, 1.0]  # Prior precision
alpha0 = [2.0, 2.0, 2.0]  # Prior shape for σ^2
beta0 = [1.0, 1.0, 1.0]  # Prior scale for σ^2

# Number of posterior samples
nsamples = 10

# Run the sampler
mu_samples, sigma2_samples = independent_sampler(X, mu0, kappa0, alpha0, beta0, nsamples)

# Inspect results
println("First sampled μ: ", mu_samples[:, 1])
println("First sampled σ^2: ", sigma2_samples[:, 1])



# 
