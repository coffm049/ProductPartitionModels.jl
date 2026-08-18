# salsoUtils.jl
# Decision-theoretic point estimates of the posterior partition via the R
# `salso` package (Dahl, Johnson, & Müller, 2022; https://CRAN.R-project.org/package=salso).
# Supports Binder loss and variation of information (VI), which reviewers requested.
#
# Relies on RCall + the R `salso` package. If either is unavailable the helpers
# return `nothing` and callers should treat results as missing.

function salso_available()
    try
        @eval using RCall
        RCall.reval("suppressPackageStartupMessages(library(salso))")
        return true
    catch
        @warn "RCall or the R 'salso' package is unavailable; SALSO metrics will be missing."
        return false
    end
end

"""
    salso_partition(C_mat, loss::Symbol=:VI; nRuns=16)

Given an `S × N` matrix of posterior cluster allocations (`S` MCMC draws, `N`
subjects), find the point-estimate partition minimizing posterior expected loss
under Binder loss (`loss=:binder`) or variation of information (`loss=:VI`).
Returns an `N`-vector of cluster labels, or `nothing` if R/salso is unavailable.
"""
function     salso_partition(C_mat::AbstractMatrix{<:Integer}; loss::Symbol=:VI, nRuns::Int=1000)
    salso_available() || return nothing

    # RCall expects a 0-based? No - integer matrix; salso treats equal labels as same cluster.
    CmatR = Int.(C_mat)

    if loss == :binder
        RCall.@rput CmatR
        RCall.@rput nRuns
        RCall.reval("part <- salso(CmatR, loss=binder(), nRuns=nRuns)")
    elseif loss == :VI
        RCall.@rput CmatR
        RCall.@rput nRuns
        RCall.reval("part <- salso(CmatR, loss=VI(), nRuns=nRuns)")
    else
        error("loss must be :binder or :VI")
    end

    part = RCall.rcopy(RCall.reval("part"))
    return Vector{Int}(vec(part))
end

"""
    salso_ari(C_mat, truth; loss=:VI, nRuns=16)

Partition point estimate (Binder or VI loss) followed by the adjusted Rand
index against `truth`. Returns a NamedTuple `(ari, nclusters)` or
`(ari=missing, nclusters=missing)` if R/salso unavailable.
"""
function salso_ari(C_mat::AbstractMatrix{<:Integer}, truth::AbstractVector{<:Integer};
                   loss::Symbol=:VI, nRuns::Int=16)
    part = salso_partition(C_mat; loss=loss, nRuns=nRuns)
    if part === nothing
        return (ari=missing, nclusters=missing)
    end
    ari = Clustering.randindex(part, collect(Int, truth))[1]
    return (ari=ari, nclusters=length(unique(part)))
end