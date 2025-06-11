# using Distributions
# using LinearAlgebra
# using Random
# using Plots


# sample the posterior mean for a normal normal model
# sample center for 
function baseCenterSampler(X)
    μ = mean(X, dims=2)

    # assume
    # Uniform prior for μ
    # Uniform prior for log(σ^2)
    (p, n) = size(X)

    # posterior
    σ2 = [rand(InverseGamma(n / 2, abs(sum((X[i, :] .- μ[i]) .^ 2) / 2) / 5)) for i in 1:p]
    μ = rand(MvNormal(μ[:], σ2 .* I(p) ./ n))
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

function independent_sampler(X, mu0, kappa0, alpha0, beta0, nsamps)
    n, p = size(X)  # Number of observations and dimensions
    X_bar = mean(X, dims=1)  # Sample means for each dimension

    # Storage for samples
    mu_samples = Matrix{Float64}(undef, p, nsamps)
    sigma2_samples = Matrix{Float64}(undef, p, nsamps)

    kappa_n = kappa0 .+ n

    for j in 1:p
        # Posterior hyperparameters for the j-th dimension
        mu_n = (kappa0[j] * mu0[j] + n * X_bar[j]) / kappa_n[j]
        alpha_n = alpha0[j] + n / 2
        beta_n = beta0[j] + 0.5 * sum((X[:, j] .- X_bar[j]) .^ 2) +
                 (kappa0[j] * n * (X_bar[j] - mu0[j])^2) / (2 * kappa_n[j])

        for i in 1:nsamps
            # Sample σ_j^2 from Inverse-Gamma
            sigma2 = rand(InverseGamma(alpha_n, beta_n))
            sigma2_samples[j, i] = sigma2

            # Sample μ_j from Normal
            mu = rand(Normal(mu_n, sqrt(sigma2 / kappa_n[j])))
            mu_samples[j, i] = mu
        end
    end

    return mu_samples, sigma2_samples
end

function NN_shrinkage(X, mu_global, tau2, kappa0, alpha0, beta0, nsamps)
    n, p = size(X)  # Number of observations and dimensions
    X_bar = mean(X, dims=1)  # Sample mean for each variable

    # Storage for samples
    mu_samples = Matrix{Float64}(undef, p, nsamps)
    sigma2_samples = Matrix{Float64}(undef, p, nsamps)

    for j in 1:p
        # Extract data for the j-th variable
        x̄ = mean(X[:, j])

        # Shrinkage: New prior mean and precision for μ_j
        # Prior: μ_j ~ Normal(mu_global, τ²), with prior precision = 1/τ²
        # Posterior updates combine this with data likelihood

        kappa_n = kappa0[j] + n
        mu_n = (mu_global / tau2 + (x̄ * n / kappa0[j])) / (n / kappa0[j] + 1 / tau2)

        alpha_n = alpha0[j] + n / 2
        beta_n = beta0[j] + 0.5 * sum((X[:, j] .- x̄) .^ 2) +
                 (kappa0[j] * n * (x̄ - mu_global)^2) / (2 * kappa_n)

        for i in 1:nsamps
            # Sample σ² from Inverse Gamma
            sigma2 = rand(InverseGamma(alpha_n, beta_n))
            sigma2_samples[j, i] = sigma2

            # Sample μ from Normal
            mu = rand(Normal(mu_n, sqrt(sigma2 / kappa_n)))
            mu_samples[j, i] = mu
        end
    end

    return mu_samples, sigma2_samples
end



# Simulated data
# Random.seed!(123)
# N, p = 7, 3
# true_mu = [5.0, 5.0, 5.0]
# true_sigma2 = [0.1, 0.1, 0.1]
# # X = hcat([rand(Normal(true_mu[j], sqrt(true_sigma2[j])), N) for j in 1:p]...)
# X = repeat([quantile(Normal(5.0, 0.1), i / (N + 1)) for i in 1:N], inner=(1, 3))
# # 
# # # Prior hyperparameters
# mu0 = [0.0, 0.0, 0.0]  # Prior mean
# kappa0 = [0.01, 0.01, 0.01]  # Prior precision
# alpha0 = [0.1, 1.0, 10.0]  # Prior shape for σ^2
# beta0 = [0.1, 1.0, 10.0]  # Prior scale for σ^2
# # 
# # # Number of posterior samples
# nsamples = 100
# # 
# # # Run the sampler
# mu_samples, sigma2_samples = independent_sampler(X, mu0, kappa0, alpha0, beta0, nsamples)
# mu_samples, sigma2_samples = NN_shrinkage(X, 0.0, 1, kappa0, alpha0, beta0, 100)
# mu_samples, sigma2_samples = NN_shrinkage(Betas, 0.0, 1e-8, kappa0, alpha0, beta0, 100)
# mean(mu_samples, dims=2)
# mean((mu_samples .- true_mu) .^2, dims = 2)
# 
# # Inspect results
# p1 = histogram(mu_samples[1, :], label="sample", title="Norm-IG sampler")
# vline!([true_mu[1]], label = "true")
# p2 = histogram(mu_samples[2,:], label = nothing)
# vline!([true_mu[2]], label = nothing)
# p3 = histogram(mu_samples[3,:], label = nothing)
# vline!([true_mu[3]], label = nothing)
# p = plot(p1, p2, p3, layout=(3,1))
# # save plot
# savefig(p, "NormIGverification.png")
# 
