using Turing, Random, Distributions, LinearAlgebra
using Clustering

# 1. Simulate data
Random.seed!(1234)
n1, n2 = 100, 100
x1 = rand(Normal(0, 1), n1)
x2 = rand(Normal(3, 1), n2)
y1 = 2 .+ 1.5 .* x1 .+ rand(Normal(0, 0.1), n1)
y2 = -1 .- 2.0 .* x2 .+ rand(Normal(0, 0.11), n2)


scatter(x1, y1)
scatter!(x2, y2)

x = vcat(x1, x2)
y = vcat(y1, y2)
z_true = vcat(fill(1, n1), fill(2, n2))  # for Rand Index

X = hcat(ones(length(x)), x)  # Design matrix with intercept
N, D = size(X)
K = 6  # Truncation level

# 2. Define the model
@model function dp_mix_regression_conjugate(X, y, K::Int)
    N, D = size(X)

    # Hyperparameters
    μ₀ = zeros(D)
    Λ₀ = Matrix{Float64}(I, D, D)
    Λ₀_inv = inv(Λ₀)
    a₀ = 2.0
    b₀ = 1.0

    # Shared variance
    σ² ~ InverseGamma(a₀, b₀)

    # Stick-breaking prior
    α ~ Gamma(1.0, 1.0)
    v ~ filldist(Beta(1.0, α), K)
    w = Vector{Real}(undef, K)
    prod_term = 1.0
    for k in 1:K
        if k == 1
            w[k] = v[1]
        else
            prod_term *= (1 - v[k - 1])
            w[k] = v[k] * prod_term
        end
    end
    w = w / sum(w)

    # Cluster-specific regression coefficients
    βs = Vector{Vector{Real}}(undef, K)
    for k in 1:K
        βs[k] ~ MvNormal(μ₀, σ² * Λ₀_inv)
    end

    # Latent cluster indicators
    z ~ filldist(Categorical(w), N)

    # Likelihood
    for i in 1:N
        k = Int(z[i])  # Ensure index is integer (avoid Dual type)
        μ = dot(X[i, :], βs[k])
        y[i] ~ Normal(μ, sqrt(σ²))
    end
end
model = dp_mix_regression_conjugate(X, y, K)
sampMethod = Gibbs(
    :z => PG(50),                               # PG for the latent discrete z
    (:σ², :βs, :v, :α) => NUTS(0.65)            # NUTS for continuous variables
)

# Run the sampler
chain = sample(model, sampMethod, 50)
chain2 = sample(model, Gibbs(), 50)



rand_indices = [
  maximum(convert.(Int, [chain[s][Symbol("z[$i]")][1,1] for i in 1:N]))
  for s in 1:length(chain)
]


# Rand Index and Adjusted Rand Index in list comprehension
rand_indices = [
  randindex(
    convert.(Int, [chain[s][Symbol("z[$i]")][1,1] for i in 1:N]),
    z_true
  )[2] for s in 1:n_samples
]
