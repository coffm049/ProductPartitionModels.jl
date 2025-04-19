using Turing, Random, Distributions, LinearAlgebra
using Clustering

# 1. Simulate data
Random.seed!(1234)
n1, n2 = 100, 100
x1 = rand(Normal(0, 1), n1)
x2 = rand(Normal(3, 1), n2)
y1 = 0 .+ 1 .* x1 .+ rand(Normal(0, 1), n1)
y2 = 0 .- 2.0 .* x2 .+ rand(Normal(0, 1), n2)

scatter(x1, y1)
scatter!(x2, y2)

# Original features (without intercept)
x = vcat(x1, x2)
X_raw = x  # assuming x is a vector from simulation

# Standardize x
x_mean = mean(X_raw)
x_std  = std(X_raw)
x_stdzd = (X_raw .- x_mean) ./ x_std

y = vcat(y1, y2)
Y_raw = y  # assuming x is a vector from simulation

# Standardize x
y_mean = mean(Y_raw)
y_std  = std(Y_raw)
y_stdzd = (Y_raw .- y_mean) ./ y_std

# Now construct design matrix with intercept
X = hcat(ones(length(x_stdzd)), x_stdzd)



z_true = vcat(fill(1, n1), fill(2, n2))  # for Rand Index

N, D = size(X)
K = 100  # Truncation level

@model function dp_mix_regression_separate_variances(X, y, K::Int, c1, c2)
    N, D = size(X)

    # --- Hyperparameters ---
    μ₀ = zeros(D)
    Λ₀ = 5.0 * I

    # Priors
    σ² ~ InverseGamma(3.0, 2.0)        # observation noise variance
    τ² ~ InverseGamma(2.0, 1.0)        # prior variance for β_k

    # --- Concentration parameter α ---
    α ~ Gamma(c1, c2)

    # --- Stick-breaking prior ---
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

    # --- Cluster-specific regression coefficients β_k ---
    βs = Vector{Vector{Real}}(undef, K)
    for k in 1:K
        βs[k] ~ MvNormal(μ₀, τ² * inv(Λ₀))
    end

    # --- Cluster assignments ---
    z ~ filldist(Categorical(w), N)

    # --- Likelihood ---
    for i in 1:N
        k = Int(z[i])
        μ = dot(X[i, :], βs[k])
        y[i] ~ Normal(μ, sqrt(σ²))
    end
end

model = dp_mix_regression_separate_variances(X, y_stdzd, K, 1.5, 1.5)

sampler = Gibbs(
    :z => MH(),
    (:σ², :τ², :βs, :v, :α) => NUTS(0.9)
)

chain = sample(model, MH(), 1000)[500:5:end]

ncs = [
  maximum(convert.(Int, [chain[s][Symbol("z[$i]")][1] for i in 1:N]))
  for s in 1:length(chain)
]
maximum([chain[i][Symbol("z[1]")][1] for i in 1:length(chain)])
rand_indices = [
  randindex(
    convert.(Int, [chain[s][Symbol("z[$i]")][1] for i in 1:N]),
    z_true
)[2]
  for s in 1:length(chain)
]


