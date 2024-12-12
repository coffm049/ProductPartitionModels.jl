# update_baseline.jl

# export ;

function update_mu0!(base::Union{Baseline_NormDLUnif, Baseline_NormUnif}, 
    mu_vec::Vector{T}, 
    prior::Union{Prior_baseline_NormDLUnif, Prior_baseline_NormUnif}) where T <: Real

    n = length(mu_vec)
    sum_mu = sum(mu_vec)

    prior_mu0_sig2 = prior.mu0_sd^2
    sig02_now = base.sig0^2

    v1 = 1.0 / (1.0 / prior_mu0_sig2 + n / sig02_now)
    m1 = v1 * (prior.mu0_mean / prior_mu0_sig2 + sum_mu / sig02_now )

    #base.mu0 = randn()*sqrt(v1) + m1
    base.mu0 = 0.0 
    return nothing
end

mutable struct TargetArgs_NormSigUnif{T <: Real} <: TargetArgs
    y::Vector{T}
    mu::T
end

function logtarget(sig::Real, args::TargetArgs_NormSigUnif)
    ee = args.y .- args.mu
    ss = sum(ee.^2)
    out = -0.5 * ss / sig^2 - length(args.y)*log(sig)
    return Dict(:llik => out, :ss => ss)
end

function update_sig0!(base::Union{Baseline_NormDLUnif, Baseline_NormUnif}, 
    mu_vec::Vector{T}, 
    prior::Union{Prior_baseline_NormDLUnif, Prior_baseline_NormUnif}, 
    sliceiter::Int) where T <: Real

    sigout, lt, it = shrinkSlice(base.sig0, 0.0, prior.sig0_upper,
                    logtarget, TargetArgs_NormSigUnif(mu_vec, base.mu0),
                    sliceiter)
    #base.sig0 = sigout
    base.sig0 = 0.0
    return nothing
end

function update_baseline!(model::Model_PPMx, update_params::Vector{Symbol}, sliceiter::Int)

    K = length(model.state.lik_params)
    mu_vec = [ model.state.lik_params[k].mu for k in 1:K ]
    #mu_vec = [ 0.0 for k in 1:K ]

    if :mu0 in update_params
        update_mu0!(model.state.baseline, mu_vec, model.prior.baseline)
    end

    if :sig0 in update_params
        update_sig0!(model.state.baseline, mu_vec, model.prior.baseline, sliceiter)
    end

    return nothing
end
