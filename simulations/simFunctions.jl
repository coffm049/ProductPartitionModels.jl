# using StatsBase
# using Statistics
# using StatsPlots
# using Distributions
# using Random
# using Plots
# using DataFrames
# using LinearAlgebra
# using ProductPartitionModels
# using Clustering
# using JLD2
# using CSV
# using GLM
# using HypothesisTests
# using LaTeXStrings
# using StatsBase

# n=200
# fractions = repeat([0.25], 4)
# variance = 1.0
# interEffect = 3.0
# common = 2.0
# plotSim=false
# using Random
# rng = MersenneTwister(1)
# niters= 500
# xdiff=1.0
#
include("gridSim.jl")

function assign_clusters(X, centroids)
    return [argmin([norm(X[:, i] - centroids[:, j]) for j in 1:size(centroids, 2)]) for i in 1:size(X, 2)]
end

# note rmse will hvae to be inverted
function findEquilibrated(; rmse::Vector{Float64}, rind::Vector{Float64}, tol::Float64=0.01)
    rmseEq = abs.(rmse .- minimum(rmse)) .<= (tol .* minimum(rmse))
    rindEq = abs.(rind .- maximum(rind)) .<= (tol .* maximum(rind))

    # 'and' v1 and v2 bit vectors together elementwise
    eq = rmseEq .& rindEq
    eq = findfirst(x -> x == 1, eq)
    if isnothing(eq)
        eq = 1
    end
    return eq
end


function simData(
    rng=Random.default_rng();
    N=300,
    dims=2,
    nclusts=3,
    fractions=repeat([0.2], 5),
    xdiff=0.5,
    interEffect=1.0,
    common=0.0,
    variance=1.0,
    plotSim=false
)
    nclusts = length(fractions)
    # Step 1: Create DataFrame with X1, X2, ..., Xdims
    df = DataFrame()
    for d in 1:dims
        df[!, "X$d"] = rand(rng, Normal(0, 1), N)
    end

    # Step 2: Assign groups
    df.group = zeros(Int, N)
    lastsub = 0
    for g in 1:nclusts
        gsize = floor(Int, N * fractions[g])
        start = lastsub + 1
        stop = g == nclusts ? N : lastsub + gsize
        df[start:stop, :group] .= g
        lastsub = stop
    end
    slopes = collect(range(common - (nclusts * interEffect), common + (nclusts * interEffect), length=nclusts))
    slopes = hcat(slopes, -slopes)
    # Step 4: Group-specific predictor shifts
    # xdiffs = (gri .- mean(gri, dims=1)) .* xdiff
    xdiffs = collect(range(-xdiff * nclusts, xdiff * nclusts, length=nclusts))
    xdiffs = hcat(xdiffs, -xdiffs)

    for d in 1:dims
        df[!, "X$d"] .+= xdiffs[df.group]
    end

    # Step 5: Center and scale X columns
    # for d in 1:dims
    #     xname = "X$d"
    #     x = df[!, xname]
    #     df[!, xname] = (x .- mean(x)) ./ std(x)
    # end

    # Step 6: Compute linear predictors
    X = Matrix(df[:, ["X$d" for d in 1:dims]])
    df.mean = [dot(X[i, :], slopes[df.group[i], :]) for i in 1:N]

    # THIS WORKS
    # df.groups = string.(df.group)
    # contrasts = Dict(:groups => EffectsCoding())  # deviation from mean
    # lm(@formula(mean ~ (X1 + X2) * groups), df; contrasts= contrasts)

    # THIS WORKS
    df.Y = df.mean .+ rand(rng, Normal(0, variance), N)
    # lm(@formula(Y ~ (X1 + X2) * groups), df; contrasts= contrasts)
    df.inter .= 1

    if plotSim & (dims == 2)
        # X marginal
        px1 = @df df density(:X1, group=:group, fillopacity=1 / 2, title="X1 marginal")
        px2 = @df df density(:X2, group=:group, fillopacity=1 / 2, title="X1 marginal")
        py = @df df density(:Y, group=:group, fillopacity=1 / 2, title="Y marginal"; permute=(:x, :y))
        xflip!(py)
        pyx1 = @df df scatter(:X1, :Y, group=:group, opacity=1 / 2, title="Y|X1")
        pyx2 = @df df scatter(:X2, :Y, group=:group, opacity=1 / 2, title="Y|X2")
        p0 = plot(legend=false, grid=false, foreground_color_subplot=:white)  #create subplot [2,2] to be blank
        l = @layout [
            ymarg{0.3w} scat1 scat2
            empt xmarg{0.3h} x2marg
        ]
        display(plot(py, pyx1, pyx2, p0, px1, px2, layout=l))
    end
    return df
end

# common = interEffect : promising
# common < interEffect : 
# common > interEffect : somewhat promising

function simExperiment(rng::AbstractRNG; N::Int=100, fractions::Vector{Float64}=[0.25, 0.25, 0.25, 0.25], variance::Real=1.0, interEffect::Float64=1.0, common::Float64=1.0, plotFit::Bool=false, niters::Int=1000, prec::Real=10.0, alph::Real=10.0, bet::Real=20.0, plotSim::Bool=false, xdiff::Real=2.0, dims::Int=2, massParams::Vector{Float64}=[1.0, 1.0])

    # Simulate data
    df = simData(rng; N=N, fractions=fractions, variance=variance, interEffect=interEffect, common=common, plotSim=plotSim, xdiff=xdiff, dims=dims)
    # THIS WORKS
    # contrasts = Dict(:groups => EffectsCoding())  # deviation from mean
    # df.groups = string.(df.group)
    # lm(@formula(Y ~ (X1 + X2) * groups), df; contrasts = contrasts)
    # ymean = mean(df.Y)
    # ystd = std(df.Y)
    #df.Y = (df.Y .- ymean) ./ ystd
    dfoos = simData(rng; N=N, fractions=fractions, variance=variance, interEffect=interEffect, common=common, plotSim=false, xdiff=xdiff, dims=dims)
    #dfoos.Y = (dfoos.Y .- ymean) ./ ystd

    # Fit 
    X = Matrix(df[:, Cols("inter", r"X")][:, Not(r"eff")])
    Xoos = Matrix(dfoos[:, Cols("inter", r"X")][:, Not(r"eff")])
    model = Model_PPMx(df.Y, X, df.group, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
    # set priors for base measure sampling
    model.prior.base = Prior_base(
        repeat([0.0], dims + 1),
        repeat([prec], dims + 1), #1.0
        repeat([alph], dims + 1), # 1.0
        repeat([bet], dims + 1) # 1.0
    )
    model.prior.massParams = massParams # 1e-3 for  common 10, inter 5 
    #model.prior.massParams = [3.0, 3.0]

    trimid = Int(niters * 3 / 5)
    simid = Int(niters * 2 / 5)
    model.state.baseline.tau0 = 1e5
    mcmc!(model, trimid; mixDPM=true)
    sim = mcmc!(model, simid; mixDPM=true)

    model2 = Model_PPMx(df.Y, X, df.group, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=false)
    mcmc!(model2, trimid; mixDPM=false)
    sim2 = mcmc!(model2, simid; mixDPM=false)

    #trimid = Int(niters/2)
    thin = 1
    sim = sim[1:thin:end]
    sim2 = sim2[1:thin:end]
    Ypred1, Cpred1 = postPred(X, model, sim)
    Ypred2, Cpred2 = postPred(X, model2, sim2)
    Ypred1oos, Cpred1oos = postPred(Xoos, model, sim)
    Ypred2oos, Cpred2oos = postPred(Xoos, model2, sim2)

    # clustering
    # clustCounts = countmap(model.state.C)
    # Clustering.randindex(model.state.C, df.group)
    # adjrindMixvec = [Clustering.randindex(s, df.group)[1] for s in eachrow(Cpred1)]
    adjrindMixvec = [Clustering.randindex(s[:C], df.group)[1] for s in sim]
    adjrindDPMvec = [Clustering.randindex(s[:C], df.group)[1] for s in sim2]
    rindMixvec = [Clustering.randindex(s[:C], df.group)[2] for s in sim]
    rindDPMvec = [Clustering.randindex(s[:C], df.group)[2] for s in sim2]
    rindMixvecoos = map(x -> Clustering.randindex(dfoos.group, x)[2], eachrow(Cpred1oos))
    rindDPMvecoos = map(x -> Clustering.randindex(dfoos.group, x)[2], eachrow(Cpred2oos))
    adjrindMixvecoos = map(x -> Clustering.randindex(dfoos.group, x)[1], eachrow(Cpred1oos))
    adjrindDPMvecoos = map(x -> Clustering.randindex(dfoos.group, x)[1], eachrow(Cpred2oos))

    # predictive
    resid1 = df.Y .- Ypred1'
    resid2 = df.Y .- Ypred2'
    resid1oos = dfoos.Y .- Ypred1oos'
    resid2oos = dfoos.Y .- Ypred2oos'

    # simulated variance
    rmseMix = sqrt.(mean(resid1 .^ 2, dims=1))[1, :]
    rmseDPM = sqrt.(mean(resid2 .^ 2, dims=1))[1, :]

    # detect when equilibrated
    # mixEq = findEquilibrated(;rmse = rmseMix, rind = rindMixvec)
    # dpmEq = findEquilibrated(;rmse = rmseDPM, rind = rindDPMvec)
    mixEq = 1
    dpmEq = 1

    # subset chains accordingly
    rmseMix = rmseMix[mixEq:end]
    rindMixvec = rindMixvec[mixEq:end]
    rmseDPM = rmseDPM[dpmEq:end]
    rindDPMvec = rindDPMvec[dpmEq:end]
    rindMixvecoos = rindMixvecoos[mixEq:end]
    rindDPMvecoos = rindDPMvecoos[dpmEq:end]

    rindMix = median(rindMixvec)
    rindDPM = median(rindDPMvec)
    rindMixoos = median(rindMixvecoos)
    rindDPMoos = median(rindDPMvecoos)
    adjrindMix = median(adjrindMixvec)
    adjrindDPM = median(adjrindDPMvec)
    adjrindMixoos = median(adjrindMixvecoos)
    adjrindDPMoos = median(adjrindDPMvecoos)

    rmseMixoos = sqrt.(mean(resid1oos[mixEq:end] .^ 2, dims=1))[1, :]
    rmseDPMoos = sqrt.(mean(resid2oos[dpmEq:end] .^ 2, dims=1))[1, :]
    midMix = mean(rmseMix)
    midDPM = mean(rmseDPM)
    midMixoos = mean(rmseMixoos)
    midDPMoos = mean(rmseDPMoos)


    # Kolmogorov-Smirnov pvalue for marginal distribution of Y
    # ks= pvalue(ApproximateTwoSampleKSTest(mean(Ypred1, dims= 1)[1,:], mean(Ypred2, dims= 1)[1,:]))
    sim = sim[mixEq:thin:end]
    sim2 = sim2[dpmEq:thin:end]

    # empirical mean for comparison
    # output_list = map(step -> median([c[:beta][2] for c in step[:lik_params]]), sim)
    # output_list = map(step -> median([c[:beta][2] for c in step[:lik_params]]), sim)
    # output_list = mean(map(step -> median([c[:beta][2] for c in step[:lik_params]]), sim2))
    # lineplot(output_list)
    # plot(output_list)

    # output_list = map(step -> [c[:beta] for c in step[:lik_params]], sim)
    # clusterEff = reduce(hcat, reduce(vcat, output_list))[2:3, :]'
    # clusterEff = clusterEff[all(abs.(clusterEff) .< 20.0, dims=2)[:, 1], :]
    # scatterplot(clusterEff[:, 1], clusterEff[:, 2]; markersize=0.5)
    # KDE on an automatically chosen grid (fast FFT method)
    # kde2d = kde((clusterEff[:, 1], clusterEff[:, 2]); npoints=(20, 20))
    # p = contourf(xg, yg, dens_mat';
    #     fill=true,        # colour fill
    #     levels=20,          # # of contour bands
    #     c=terrain,
    #     linewidth=0,         # no border on filled bands
    #     clims=(0, maximum(dens_mat)),  # full dynamic range
    #     xlabel="x", ylabel="y",
    #     aspect_ratio=:equal,
    #     title="2-D KDE topography")
    # # Plot – transpose so orientation matches xg/yg
    # contourf(kde2d.x, kde2d.y, kde2d.density' .+ 1e-6;            # note '
    #     xlabel="x", ylabel="y",
    #     title="2-D kernel-density estimate",
    #     colorbar_title="density")#, colorbar_scale=:log10)
    # xlims!(-1, 2.5)
    # ylims!(-2.5, 1.1)



    commonBeta0 = [s[:prior_mean_beta][1] for s in sim]
    meanBeta0 = mean(commonBeta0)
    commonBeta1 = [s[:prior_mean_beta][2] for s in sim]

    # lineplot(commonBeta1)
    # dpmCI = quantile(commonBeta1[abs.(commonBeta1).<10], [0.05, 0.95])
    dpmCI = quantile(commonBeta1, [0.05, 0.95])
    # plot(commonBeta1)
    # histogram(commonBeta1)
    # vline!(dpmCI)
    # meanBeta1 = mean(commonBeta1[abs.(commonBeta1) .< 10])
    meanBeta1 = median(commonBeta1)
    commonBeta2 = [s[:prior_mean_beta][3] for s in sim]
    dpmCI2 = quantile(commonBeta2, [0.05, 0.95])
    # dpmCI2 = quantile(commonBeta2[abs.(commonBeta2).<10], [0.05, 0.95])
    # lineplot(commonBeta2)
    # plot(commonBeta2)
    # histogram(commonBeta2)
    # vline!(dpmCI2)
    # meanBeta2 = mean(commonBeta2[abs.(commonBeta2) .< 10])
    meanBeta2 = median(commonBeta2)

    # find central 90% credible interval
    # using KernelDensity
    # cb1dens = kde(commonBeta1)
    # dx = cb1dens.x[2] - cb1dens.x[1]
    # idx = sortperm(cb1dens.density, rev=true)
    # ecdf = cumsum(cb1dens.density[idx]) * dx
    # minSet = cb1dens.x[idx[1:findfirst(ecdf .> 0.9)]]
    # minx = minimum(minSet)
    # maxx = maximum(minSet)
    # histogram(commonBeta1)
    # vline!([minx, maxx], label = "HPD")
    # vline!(quantile(commonBeta1), [0.05, 0.95], label = "CI")
    # xlims!(-0.2, 1.4)
    # find central 90% credible interval

    # check if 3.0 is in dpmCI
    zeroInDPM = dpmCI[1] < 0.0 < dpmCI[2]
    commonInDPM = dpmCI[1] < common < dpmCI[2]
    zeroInDPM2 = dpmCI2[1] < 0.0 < dpmCI2[2]
    commonInDPM2 = dpmCI2[1] < common < dpmCI2[2]


    # is each obs in the 0.05 0.95 quantiles of the posterior predictive?
    bayesPmix = median(mean(df.Y .<= Ypred1', dims=2))
    bayesPDPM = median(mean(df.Y .<= Ypred2', dims=2))
    bayesPmixoos = median(mean(df.Y .<= Ypred1oos', dims=2))
    bayesPDPMoos = median(mean(df.Y .<= Ypred2oos', dims=2))

    # Simple linear regresion
    slr = lm(@formula(Y ~ X1 + X2), df)
    #slr = lm(@formula(Y ~ X1 + X2), df)
    #df.groups .= string.(df.group)
    # dfoos.group .= string.(dfoos.group)
    # contrasts = Dict(:groups => EffectsCoding())  # deviation from mean
    #slr = lm(@formula(Y ~ (X1 + X2) * groups), df; contrasts=contrasts)
    betaCI = confint(slr)[2, :]
    betaCI2 = confint(slr)[3, :]
    # test if 0 is between the two values in betaCI
    zeroInSLR = (0.0 .>= betaCI[1]) & (0.0 <= betaCI[2])
    commonInSLR = (common .>= betaCI[1]) & (common <= betaCI[2])
    slrRMSE = sqrt(mean(residuals(slr) .^ 2))
    zeroInSLR2 = (0.0 .>= betaCI2[1]) & (0.0 <= betaCI2[2])
    commonInSLR2 = (common .>= betaCI2[1]) & (common <= betaCI2[2])


    # Add in Kmeans with full interaction effect model
    # - select best clustering based off X
    # - interaction model
    # kclust = argmin([kmeans(X', i).totalcost for i in 1:10])
    #kmodel = kmeans(X', kclust)
    nclusts = length(fractions)
    kmodel = kmeans(X', nclusts)
    df.kclust = string.(kmodel.assignments)
    dfoos.kclust = string.(assign_clusters(Xoos', kmodel.centers))

    rindKclust = Clustering.randindex(parse.(Int, df.kclust), df.group)
    rindKclustoos = Clustering.randindex(parse.(Int, dfoos.kclust), dfoos.group)
    # linear model with Y vs X1, X2
    contrasts = Dict(:kclust => EffectsCoding())  # deviation from mean
    clustlm = lm(@formula(Y ~ (X1 + X2) * kclust), df; contrasts=contrasts)
    # df.group = string.(df.group)
    # clustlm = lm(@formula(Y ~ (X1 + X2) * group), df)
    kmeanMSE = sqrt(mean(residuals(clustlm) .^ 2))
    kmeanMSEoos = sqrt(mean(((predict(clustlm, dfoos)) - dfoos.Y) .^ 2))
    meanBetakclust1 = coef(clustlm)[2]
    meanBetakclust2 = coef(clustlm)[3]

    # seef if 95% CI for X1 coefficient in clustlm includes 0
    kclustCI = confint(clustlm)[2, :]
    zeroInk = (0.0 .>= kclustCI[1]) & (0.0 <= kclustCI[2])
    commonInk = (common .>= kclustCI[1]) & (common <= kclustCI[2])
    kclustCI2 = confint(clustlm)[3, :]
    zeroInk2 = (0.0 .>= kclustCI2[1]) & (0.0 <= kclustCI2[2])
    commonInk2 = (common .>= kclustCI2[1]) & (common <= kclustCI2[2])

    ncMix = mode([maximum(s[:C]) for s in sim])
    ncDPM = mode([maximum(s[:C]) for s in sim2])
    #ncK = kclust
    ncK = nclusts

    Mix_beta1_c1 = [s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == ncMix]
    dpm_beta1_c1 = [s[:lik_params][1][:beta][1] for s in sim2 if maximum(s[:C]) == ncDPM]


    result = DataFrame(
        # mixture model
        rind_Mix=rindMix,
        rindMixoos=rindMixoos,
        adjrind_Mix=adjrindMix,
        adjrindMixoos=adjrindMixoos,
        midMix=midMix,
        midMixoos=midMixoos,
        bayesPmix=bayesPmix,
        bayesPmixoos=bayesPmixoos,
        meanBeta1=meanBeta1,
        meanBeta2=meanBeta2,
        Mix_beta1_c1=median(Mix_beta1_c1),
        mixEq=mixEq,
        zeroInDPM=zeroInDPM,
        commonInDPM=commonInDPM,
        zeroInDPM2=zeroInDPM2,
        commonInDPM2=commonInDPM2,

        # PPMx model
        rind_DPM=rindDPM,
        rind_DPMoos=rindDPMoos,
        adjrind_DPM=adjrindDPM,
        adjrind_DPMoos=adjrindDPMoos,
        midDPM=midDPM,
        midDPMoos=midDPMoos,
        bayesPDPM=bayesPDPM,
        bayesPDPMoos=bayesPDPMoos,
        dpm_beta1_c1=median(dpm_beta1_c1),
        dpmEq=dpmEq,

        # rind_K
        adjrind_K=rindKclust[1],
        adjrind_Koos=rindKclustoos[1],
        rind_K=rindKclust[2],
        rind_Koos=rindKclustoos[2],
        kmean_MSE=kmeanMSE,
        kmean_MSEoos=kmeanMSEoos,
        meanBetakclust1=meanBetakclust1,
        zeroInk=zeroInk,
        commonInk=commonInk,
        meanBetakclust2=meanBetakclust2,
        zeroInk2=zeroInk2,
        commonInk2=commonInk2,

        # SLR
        meanBetaSLR=coef(slr)[2],
        meanBetaSLR2=coef(slr)[3],
        zeroInSLR=zeroInSLR,
        commonInSLR=commonInSLR,
        zeroInSLR2=zeroInSLR2,
        commonInSLR2=commonInSLR2,
        slrRMSE=slrRMSE,

        # number of clusts
        ncMix=ncMix,
        ncDPM=ncDPM,
        ncK=ncK,

        # setup
        N=N, fractions=string(fractions), variance=variance,
        interEffect=interEffect, common=common, prec=prec, alph=alph, bet=bet, xdiff=xdiff
    )

    if plotFit
        # 4 
        # clustering 
        histogram([rmseMix, rmseDPM], label=["PPMx-shared" "PPMx"], fillalpha=0.33, title="RMSE")
        vline!([kmeanMSE], label="K-means")
        display(current())

        # prediction
        histogram([rindMixvec, rindDPMvec], label=["PPMx-shared" "PPMx"], fillalpha=0.33, title="Rand Index")
        vline!([rindKclust[2]], label="K-means")
        display(current())

        # inference
        # Common
        histogram(commonBeta1, label=L"\beta_1", title="Common Effects estimates", fillalpha=0.33)
        vline!([common], label="True", color="black", linewidth=3)
        vline!([0], label="Zero", color="black", linewidth=3, linestyle=:dot)
        display(current())
    end

    return result
end
