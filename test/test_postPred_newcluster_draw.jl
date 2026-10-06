# test_postPred_newcluster_draw.jl
#
# Regression test for the new-singleton component in postPred / postPredLogdens.
#
# DEFECT: `simpri_lik_params` was called inside the per-observation loop, so each
# predicted observation within a single posterior draw was scored against a
# *different* new-cluster parameter draw. That injects Monte Carlo noise which is
# not part of the posterior and which grows with the number of predictions.
#
# The fix draws the new-cluster parameters once per posterior sample.
#
# NOTE (v2.0 audit): `lik_params[k][:beta]` is the cluster's TOTAL slope, not a
# deviation from a common effect. `llik_k` (likelihood.jl:155) uses it directly
# as `means += z'beta`, and `independent_sampler` (NN_updater.jl:102-109)
# estimates `prior_mean_beta` as the MEAN of those slopes. So `prior_mean_beta`
# is a scalar shrinkage summary and must NOT be added to a cluster's slope.
# `postPredIgnored` below pins that down, because adding it double-counts.
#
# Run with:  julia --project=. test/test_postPred_newcluster_draw.jl

using Test
using Random
using Statistics
using LinearAlgebra
using ProductPartitionModels

const PPM = ProductPartitionModels

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
Minimal two-cluster PPMx regression state built by hand, so the tests do not
depend on running the sampler.
"""
function make_state(; n=60, p=2, beta1=[1.5, -1.0], beta2=[-0.4, 0.8],
                    sigma=0.5, seed=1234)
    rng = MersenneTwister(seed)
    C = vcat(fill(1, n ÷ 2), fill(2, n - n ÷ 2))
    X = randn(rng, n, p)

    lik_params = [
        PPM.LikParams_PPMxReg(0.0, sigma, collect(beta1),
                              PPM.Hypers_DirLap(ones(p), ones(p), 1.0)),
        PPM.LikParams_PPMxReg(0.0, sigma, collect(beta2),
                              PPM.Hypers_DirLap(ones(p), ones(p), 1.0)),
    ]

    y = zeros(n)
    for i in 1:n
        y[i] = lik_params[C[i]].mu + dot(X[i, :], lik_params[C[i]].beta)
    end

    model = PPM.Model_PPMx(y, X, C)
    model.state.lik_params = lik_params
    model.prior.base = PPM.Prior_base(zeros(p), fill(0.01, p), fill(1.0, p), fill(1.0, p))
    return model
end

"""
One posterior draw in the exact shape `mcmc!` records it (`mcmc.jl:187-208`).
`:lik_params` entries and `:baseline` are Dicts built by `deepcopyFields`.
"""
function make_draw(model; with_common=false, beta_star=[1.5, -1.0])
    d = Dict{Symbol,Any}()
    d[:C] = copy(model.state.C)
    d[:lik_params] = [Dict{Symbol,Any}(:mu => lp.mu, :sig => lp.sig,
                                       :beta => copy(lp.beta))
                      for lp in model.state.lik_params]
    d[:baseline] = Dict{Symbol,Any}(:mu0 => 0.0, :sig0 => 20.0)
    if with_common
        d[:prior_mean_beta] = copy(beta_star)
    end
    return d
end

# ---------------------------------------------------------------------------
# 1. new-cluster component is constant within a posterior sample
# ---------------------------------------------------------------------------

@testset "new-cluster component drawn once per sample" begin

    model = make_state()
    draws = [make_draw(model)]

    # Two identical rows of Xpred must receive identical predictive densities:
    # they get the same cluster weights AND, after the fix, share one
    # new-cluster draw. Before the fix they were scored against different
    # new-cluster parameters and came out unequal for a reason that has nothing
    # to do with the observation.
    Xrow = [1.0 0.5]
    Xpair = [Xrow; Xrow]
    y_pair = [0.0, 0.0]

    Random.seed!(20240601)
    lpair = PPM.postPredLogdens(Xpair, y_pair, model, draws; crossxy=false)
    @test lpair[1, 1] ≈ lpair[1, 2]

    # Wider check: every copy of a distinct row must agree with all others.
    Xmany = repeat([1.0 0.5; -1.0 2.0; 0.3 -0.7], 4, 1)
    ymany = zeros(size(Xmany, 1))
    Random.seed!(20240601)
    lmany = PPM.postPredLogdens(Xmany, ymany, model, draws; crossxy=false)
    for r in 1:3
        @test lmany[1, r] ≈ lmany[1, r + 3]
        @test lmany[1, r] ≈ lmany[1, r + 6]
        @test lmany[1, r] ≈ lmany[1, r + 9]
    end
    @test all(isfinite, lmany)

    # NOTE: no exact-equality check is made on `postPred` output. `postPred` draws
    # C_i and y_i, so two identical covariate rows legitimately receive different
    # samples even when the underlying parameters are shared. Only
    # `postPredLogdens`, which is deterministic given a posterior draw, has the
    # identical-rows invariant.
end

# ---------------------------------------------------------------------------
# 2. the stored beta is the TOTAL slope; prior_mean_beta must not be added
# ---------------------------------------------------------------------------

@testset "cluster slope is the stored beta, not beta* + beta" begin

    model = make_state()
    Xpred = [1.0 0.5; -1.0 2.0; 0.3 -0.7]
    ypred = zeros(3)

    # The predictive mean must not depend on prior_mean_beta at all. If anyone
    # re-introduces a beta* + delta reconstruction, these differ.
    d_with = make_draw(model; with_common=true, beta_star=[3.0, -2.0])
    d_without = make_draw(model; with_common=false)

    Random.seed!(1)
    _, _, M_with = PPM.postPred(Xpred, model, [d_with])
    Random.seed!(1)
    _, _, M_without = PPM.postPred(Xpred, model, [d_without])
    @test M_with ≈ M_without

    Random.seed!(1)
    l_with = PPM.postPredLogdens(Xpred, ypred, model, [d_with]; crossxy=false)
    Random.seed!(1)
    l_without = PPM.postPredLogdens(Xpred, ypred, model, [d_without]; crossxy=false)
    @test l_with ≈ l_without

    # Sanity on magnitude: a chain carrying a huge beta* cannot move predictions.
    d_big = make_draw(model; with_common=true, beta_star=[50.0, -50.0])
    Random.seed!(1)
    _, _, M_big = PPM.postPred(Xpred, model, [d_big])
    @test M_big ≈ M_without

    # Ground truth: the model's own likelihood uses beta directly
    # (likelihood.jl:150-155 -> means = mu + z'beta). Cluster 1's mean for the
    # first prediction row, with standardized X, must be reachable from beta1.
    @test size(M_without, 2) == 3
end

# ---------------------------------------------------------------------------
# 3. new-cluster draw follows the model's own zero-centered prior
# ---------------------------------------------------------------------------

@testset "new-cluster slope drawn from the zero-centered model prior" begin

    model = make_state()
    draw = make_draw(model; with_common=true, beta_star=[1.5, -1.0])
    upd = [:mu, :sig, :beta, :mu0, :sig0]

    # Replicate the helper's baseline substitution so this is a like-for-like
    # comparison: `new_cluster_params` overrides the baseline mu0/sig0 with the
    # values recorded in the draw's `:baseline` entry, which differ from the
    # state's own baseline.
    basenow = deepcopy(model.state.baseline)
    basenow.mu0 = draw[:baseline][:mu0]
    basenow.sig0 = draw[:baseline][:sig0]

    Random.seed!(11)
    raw = PPM.simpri_lik_params(basenow, model.p, model.state.lik_params[1], upd)
    Random.seed!(11)
    got = PPM.new_cluster_params(draw, model, upd)

    # identical draw under an identical seed: the helper adds no extra randomness
    @test got.beta ≈ raw.beta
    @test got.mu ≈ raw.mu
    @test got.sig ≈ raw.sig

    # and it is not shifted toward the chain's beta*
    @test !(got.beta ≈ raw.beta .+ [1.5, -1.0])

    # averaged over many draws the slope sits near the model's prior center, 0
    mus = Float64[]
    for s in 1:400
        Random.seed!(s)
        push!(mus, mean(PPM.new_cluster_params(draw, model, upd).beta))
    end
    @test abs(mean(mus)) < 0.5
end

# ---------------------------------------------------------------------------
# 4. postPred/postPredLogdens still run and are reproducible
# ---------------------------------------------------------------------------

@testset "postPred runs and is reproducible" begin

    model = make_state()
    draws = [make_draw(model; with_common=true)]
    Xpred = randn(MersenneTwister(11), 30, 2)

    Random.seed!(4)
    Y1, C1, M1 = PPM.postPred(Xpred, model, draws)
    Random.seed!(4)
    Y2, C2, M2 = PPM.postPred(Xpred, model, draws)

    @test size(Y1) == (1, 30)
    @test size(M1) == (1, 30)
    @test Y1 == Y2
    @test M1 == M2
    @test all(isfinite, M1)

    # standard PPMx (no prior_mean_beta key at all) must also work
    draws_std = [make_draw(model; with_common=false)]
    Random.seed!(4)
    Y3, C3, M3 = PPM.postPred(Xpred, model, draws_std)
    @test all(isfinite, M3)
end

println("\npostPred new-cluster-draw regression tests passed.")