using Distributions
using LinearAlgebra
using DataFrames
using Random
#using UnicodePlots
using Tables
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

function simMSE(X, mu0, kap, alph, bet, nsamples)
    dim = size(X)[2]
    mu_samples, sigma2_samples = independent_sampler(
    X, repeat([mu0], dim), repeat([kap], dim), repeat([alph], dim), repeat([bet], dim), nsamples)
    mse = mean((mu_samples .- true_mu) .^2, dims = 2)[1]
    mse2 = mean((sigma2_samples .- true_sigma2) .^2, dims = 2)[1]
  return mse, mse2
end

# Simulated data
Random.seed!(123)
N, p = 7, 3
true_mu = repeat([1.0], p)
true_sigma2 = repeat([0.5], p)
# X = hcat([rand(Normal(true_mu[j], sqrt(true_sigma2[j])), N) for j in 1:p]...)
X = [quantile(Normal(true_mu[p], true_sigma2[p]), i/(N + 1)) for i in 1:N, j in 1:p]

# Prior hyperparameters
mu0 = [0.0]  # Prior mean
kappa0 = [0.1]  # Prior precision variance is the 1/0.1 = 10
alpha0 = [0.1, 1.0]  # Prior shape for σ^2
beta0 = [0.01, 0.1, 0.5, 1.0, 2.0]  # Prior scale for σ^2 for alph 0.1


# Number of posterior samples
nsamples = 100

# loop over all combinations ofr priors in for loop 
# use the independent sampler given those priors then add 
# MSE to store vector
experimentDF = DataFrame(collect(Base.Iterators.product(mu0, kappa0, alpha0, beta0)), [:priorμ, :priorκ, :priorα, :priorβ])
test = Tables.matrix([simMSE(X, row.priorμ, row.priorκ, row.priorα, row.priorβ, nsamples) for row in eachrow(experimentDF)])
experimentDF[!, :mse] = test[:,1]
experimentDF[!, :mse2] = test[:,2]


# scatterplot(experimentDF.priorκ, experimentDF.mse) # 1.0
# scatterplot(experimentDF.priorα, experimentDF.mse) # 5.0
# scatterplot(experimentDF.priorβ, experimentDF.mse) # 1.0
# scatterplot(experimentDF.priorκ, experimentDF.mse2) # 1.0
# scatterplot(experimentDF.priorα, experimentDF.mse2) # 5.0
# scatterplot(experimentDF.priorβ, experimentDF.mse2) # 1.0
scatter(experimentDF.priorκ, experimentDF.mse) # 1.0
scatter(experimentDF.priorα, experimentDF.mse) # 3.0
scatter(experimentDF.priorβ, experimentDF.mse) # 1.0
scatter(experimentDF.priorκ, experimentDF.mse2) # 1.0
scatter(experimentDF.priorα, experimentDF.mse2) # 3.0
scatter(experimentDF.priorβ, experimentDF.mse2) # 1.0

# resulting prior distribution
mud = Normal(0, 1.0)
sigd = InverseGamma(1, 1.0)
x = -4:0.01:4
plot(x, pdf.(mud, x), title = "center prior")
plot(abs.(x), pdf.(sigd, abs.(x)), title = "spread prior")

# Example of the fit
mu_samples, sigma2_samples = independent_sampler(
X, repeat([0], 3), repeat([1.0], 3), repeat([1.0], 3), repeat([1.0], 3), nsamples)
# Inspect results
p1 = histogram(mu_samples[1,:], label = "sample", title = "Center")
vline!([true_mu[1], mean(mu_samples[1,:])], label = ["true" "mean"], color = ["black" "red"])
p2 = histogram(sigma2_samples[1,:], label = "sample", title = "Variance")
vline!([true_sigma2[1], mean(sigma2_samples[1,:])], label = ["true" "mean"], color = [:black :red])
plot(p1, p2)
# p2 = histogram(mu_samples[2,:], label = nothing)
# vline!([true_mu[2]], label = nothing)
# vline!([true_mu[3]], label = nothing)
# p = plot(p1, p2, p3, layout=(3,1))
# save plot
savefig(p, "NormIGverification.png")


