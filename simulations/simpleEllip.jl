n, p = size(Betas)
prior_var_beta = repeat([2.0], 3)
Σ = prior_var_beta .* I(p)

d = Betas .- prior_mean_beta'

d * inv(Σ) *d

-(n*p/2) * log(2π) - n/2 * det(Σ) - 1/2


function loglik_diag_mvn(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector)
    n, p = size(X)
    @assert length(μ) == p "μ must have length p."
    @assert length(σ) == p "σ must have length p."
    @assert all(σ .> 0)    "σ must contain positive values."

    invσ²   = 1.0 ./ (σ .^ 2)            # element-wise 1/σ²
    logdet  = sum(log.(σ .^ 2))          # log|Σ| for diagonal Σ
    con  = -0.5 * n * p * log(2π) - 0.5 * n * logdet

    diff    = X .- μ'                    # broadcast μ as a row
    quad = sum(diff * ( invσ² .* I(p)) * diff', dims = 2)
    #quad    = sum(diff.^2 .* invσ²', dims=2)   # per-row Mahalanobis terms
    ll      = con - 0.5 * sum(quad)    # sum across all rows

    return ll
end

