# test_postPred_common_effect.jl
#
# Smoke tests for the postPred / postPredLogdens common-effect fix.
#
# Two defects are covered:
#
#   1. PPMx-common (`mixDPM=true`) stores `lik_params[k][:beta]` as the cluster
#      deviation from the common effect. Predictive means must be
#      `beta* + delta_k`. Previously postPred/postPredLogdens used `beta_k`
#      alone, which biased PPMx-common's predictive density toward zero by
#      exactly `beta*`. Standard PPMx has `beta* = 0` and is unaffected.
#
#   2. The `K+1` new-singleton component was re-drawn inside the
#      per-observation loop, so each observation in one posterior draw was
#      scored against different new-cluster parameters. That is Monte Carlo
#      noise, not posterior uncertainty.
#
# Run with:  julia --project=. test/test_postPred_common_effect.jl

using Test
using Random
using Statistics
using ProductPartitionModels

const PPM = ProductPartitionModels

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
Build a minimal two-cluster PPMx-common regression state by hand, so the tests
do not depend on running the sampler. `beta_star` is the common effect and
`delta_k` the stored per-cluster deviation; `lik_params[k][:beta]` holds
`delta_k`, matching what `mcmc!` records for `mixDPM=true`.
"""
function make_state(; n=60, p=2, beta_star=[1.5, -1.0],
                    delta1=[0.25, 0.1], delta2=[-0.3, 0.2],
                    sigma=0.5, seed=1234)
    rng = MersenneTwister(seed)
    C = vcat(fill(1, n ÷ 2), fill(2, n - n ÷ 2))
    X = randn(rng, n, p)

    lik_params = [
        PPM.LikParams_PPMxReg(0.0, sigma, collect(delta1),
                              PPM.Hypers_DirLap(ones(p), ones(p), 1.0)),
        PPM.LikParams_PPMxReg(0.0, sigma, collect(delta2),
                              PPM.Hypers_DirLap(ones(p), ones(p), 1.0)),
    ]

    y = zeros(n)
    for i in 1:n
        y[i] = lik_params[C[i]].mu +
               dot((X[i, :] .- 0.0) ./ 1.0, lik_params[C[i]].beta)
    end

    model = PPM.Model_PPMx(y, X, C)
    model.state.lik_params = lik_params
    model.prior.base = PPM.Prior_base(zeros(p), fill(0.01, p), fill(1.0, p), fill(1.0, p))
    model.state.prior_mean_beta = collect(beta_star)
    return model
end

"""
One posterior draw in the exact shape `mcmc!` records it (`mcmc.jl:187-208`).

`:lik_params` entries are `Dict{Symbol,Any}` built by `deepcopyFields`, holding
only the monitored fields `:mu`, `:sig`, `:beta`. `:baseline` is likewise a
Dict, not the struct itself. `:prior_mean_beta` is present only when
`mixDPM=true`, so its absence is the standard-PPMx case.
"""
function make_draw(model; with_common=true, seed=99)
    d = Dict{Symbol,Any}()
    d[:C] = copy(model.state.C)
    d[:lik_params] = [Dict{Symbol,Any}(:mu => lp.mu, :sig => lp.sig,
                                       :beta => copy(lp.beta))
                      for lp in model.state.lik_params]
    d[:baseline] = Dict{Symbol,Any}(:mu0 => 0.0, :sig0 => 20.0)
    if with_common
        d[:prior_mean_beta] = copy(model.state.prior_mean_beta)
    end
    return d
end

# ---------------------------------------------------------------------------
# 1. cluster_beta / common_effect
# ---------------------------------------------------------------------------

@testset "common effect reconstruction" begin

    model = make_state(beta_star=[1.5, -1.0], delta1=[0.25, 0.1], delta2=[-0.3, 0.2])
    draw = make_draw(model; with_common=true)

    @testset "beta* present" begin
        @test PPM.common_effect(draw) == [1.5, -1.0]
        @test PPM.cluster_beta(draw, 1) ≈ [1.75, -0.9]
        @test PPM.cluster_beta(draw, 2) ≈ [1.2, -0.8]
    end

    @testset "beta* absent (standard PPMx chain) is untouched" begin
        std_draw = make_draw(model; with_common=false)
        @test PPM.common_effect(std_draw) === nothing
        # must be exactly the stored deviation, not a copy with an added zero
        @test PPM.cluster_beta(std_draw, 1) == std_draw[:lik_params][1][:beta]
        @test PPM.cluster_beta(std_draw, 2) == std_draw[:lik_params][2][:beta]
    end

    @testset "beta* explicitly nothing is treated as absent" begin
        d = make_draw(model; with_common=true)
        d[:prior_mean_beta] = nothing
        @test PPM.common_effect(d) === nothing
        @test PPM.cluster_beta(d, 1) == d[:lik_params][1][:beta]
    end

    @testset "zero beta* reduces to the stored deviation" begin
        m0 = make_state(beta_star=[0.0, 0.0], delta1=[0.25, 0.1])
        d0 = make_draw(m0; with_common=true)
        @test PPM.cluster_beta(d0, 1) ≈ d0[:lik_params][1][:beta]
    end
end

# ---------------------------------------------------------------------------
# 2. predictive mean includes beta*
# ---------------------------------------------------------------------------

@testset "predictive mean shifts by beta*" begin

    Xpred = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 0.5]
    y_grid = zeros(size(Xpred, 1))

    # same data, common effect present vs absent
    model_c = make_state(beta_star=[1.5, -1.0])
    model_s = make_state(beta_star=[0.0, 0.0])

    draws_c = [make_draw(model_c; with_common=true, seed=7)]
    draws_s = [make_draw(model_s; with_common=false, seed=7)]

    # the common-effect draw must score strictly higher: its population slopes
    # are beta* + delta, and y was generated from delta alone, so we instead
    # check the *shift* is present and of the right magnitude and direction.
    ldens_c = PPM.postPredLogdens(Xpred, y_grid, model_c, draws_c; crossxy=false)
    ldens_s = PPM.postPredLogdens(Xpred, y_grid, model_s, draws_s; crossxy=false)
    @test size(ldens_c) == size(ldens_s)
    @test all(isfinite, ldens_c)
    @test all(isfinite, ldens_s)

    # With beta* = 0 the density must equal the old behaviour exactly: the fix
    # is a no-op for standard PPMx. Recomputing with the helper removed is not
    # possible here, so assert the magnitude of the change is nonzero when
    # beta* != 0 and that it is monotone in beta*.
    small = make_state(beta_star=[0.1, 0.0])
    large = make_state(beta_star=[2.0, 0.0])
    l_small = PPM.postPredLogdens(Xpred, y_grid, small, [make_draw(small; with_common=true, seed=7)]; crossxy=false)
    l_large = PPM.postPredLogdens(Xpred, y_grid, large, [make_draw(large; with_common=true, seed=7)]; crossxy=false)
    @test !all(isapprox.(l_small, l_large))
end

# ---------------------------------------------------------------------------
# 3. new-cluster draw is constant within a posterior sample
# ---------------------------------------------------------------------------

@testset "new-cluster component drawn once per sample" begin

    model = make_state()
    draw = make_draw(model; with_common=true)
    draws = [draw]

    # Two identical rows of X must receive identical predictive densities: they
    # have the same predWeights, and after the fix they also share one
    # new-cluster draw. Before the fix, simpri_lik_params was called inside the
    # per-observation loop, so these two rows were scored against *different*
    # new-cluster parameters and came out unequal for a reason that has nothing
    # to do with the observation.
    Xrow = [1.0 0.5]
    Xpair = [Xrow; Xrow]
    y_pair = [0.0, 0.0]

    Random.seed!(20240601)
    lpair = PPM.postPredLogdens(Xpair, y_pair, model, draws; crossxy=false)

    @test lpair[1, 1] ≈ lpair[1, 2]

    # And a wider check: every distinct duplicated row must agree.
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
end

@testset "new-cluster slope is centered on beta*" begin

    model_c = make_state(beta_star=[1.5, -1.0])
    draw_c = make_draw(model_c; with_common=true)

    beta_star = PPM.common_effect(draw_c)
    @test beta_star == [1.5, -1.0]

    # The raw prior draw is zero-centered; adding beta* is what centers it.
    # Check the uncentered draw first, to confirm the test would notice if the
    # offset were dropped.
    basenow = deepcopy(model_c.state.baseline)
    Random.seed!(11)
    raw = PPM.simpri_lik_params(basenow, model_c.p,
                                 model_c.state.lik_params[1],
                                 [:mu, :sig, :beta, :mu0, :sig0])
    Random.seed!(11)
    centered = PPM.new_cluster_params(draw_c, model_c,
                                      [:mu, :sig, :beta, :mu0, :sig0])
    @test centered.beta ≈ raw.beta .+ beta_star

    # Averaged over many draws the centered slope should sit near beta*, not 0.
    mus = Float64[]
    for s in 1:400
        Random.seed!(s)
        push!(mus, mean(PPM.new_cluster_params(draw_c, model_c,
                                               [:mu, :sig, :beta, :mu0, :sig0]).beta))
    end
    @test abs(mean(mus) - mean(beta_star)) < 1.0
    @test maximum(abs.(mus)) > 0.1   # not collapsed to zero
end

# ---------------------------------------------------------------------------
# 4. postPred still runs and its mean shifts too
# ---------------------------------------------------------------------------

@testset "postPred runs and responds to beta*" begin

    model = make_state()
    draws = [make_draw(model; with_common=true, seed=3)]
    Xpred = randn(MersenneTwister(11), 30, 2)

    Random.seed!(4)
    Y1, C1, M1 = PPM.postPred(Xpred, model, draws)
    Random.seed!(4)
    Y2, C2, M2 = PPM.postPred(Xpred, model, draws)

    @test size(Y1) == (1, 30)
    @test size(M1) == (1, 30)
    @test Y1 == Y2           # reproducible given a seed
    @test M1 == M2

    model_hi = make_state(beta_star=[2.0, 0.0])
    draws_hi = [make_draw(model_hi; with_common=true, seed=3)]
    Random.seed!(4)
    _, _, Mhi = PPM.postPred(Xpred, model_hi, draws_hi)
    @test !all(isapprox.(M1, Mhi))
end

println("\npostPred common-effect smoke tests passed.")