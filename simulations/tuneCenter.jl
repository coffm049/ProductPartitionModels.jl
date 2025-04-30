using Distributions
using LinearAlgebra
using Random
using Plots



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
N, p = 7, 3
true_mu = repeat([0.25], p)
true_sigma2 = repeat([0.1], p)
# X = hcat([rand(Normal(true_mu[j], sqrt(true_sigma2[j])), N) for j in 1:p]...)
X = [quantile(Normal(true_mu[p], true_sigma2[p]), i/(N + 1)) for i in 1:N, j in 1:p]

# Prior hyperparameters
mu0 = [0.0, 0.0, 0.0]  # Prior mean
kappa0 = [1, 1e-1, 1e-2]  # Prior precision
alpha0 = [30.0, 30.0, 30.0]  # Prior shape for σ^2
beta0 = [1e-1, 1e-1, 1e-1]  # Prior scale for σ^2

# Number of posterior samples
nsamples = 100

# Run the sampler
mu_samples, sigma2_samples = independent_sampler(X, mu0, kappa0, alpha0, beta0, nsamples)
store = zeros(length(kappa0), length(alpha0), length(beta0))
# loop over all combinations ofr priors in for loop 
# use the independent sampler given those priors then add 
# MSE to store vector
results = []

for kap, alph, bet in collect(Base.Iterators.product(kappa0, alpha0, beta0))
    mu_samples, sigma2_samples = independent_sampler(X, mu0[p], kap, alph, bet, nsamples)
    mse = mean((mu_samples .- true_mu) .^2, dims = 2)
    push!(results, mse)
end

for kap, alph, bet in collect(Base.Iterators.product(kappa0, alpha0, beta0))
  mu_samples, sigma2_samples = independent_sampler(X, mu0[p], kap, alph, bet, nsamples)
  #mean((mu_samples .- true_mu) .^2, dims = 2)
  mse = mean((mu_samples .- true_mu) .^2, dims = 2)
  push!(results, mse)
end


# Inspect results
p1 = histogram(mu_samples[1,:], label = "sample", title = "Norm-IG sampler")
vline!([true_mu[1]], label = "true")
p2 = histogram(mu_samples[2,:], label = nothing)
vline!([true_mu[2]], label = nothing)
p3 = histogram(mu_samples[3,:], label = nothing)
vline!([true_mu[3]], label = nothing)
p = plot(p1, p2, p3, layout=(3,1))
# save plot
savefig(p, "NormIGverification.png")


