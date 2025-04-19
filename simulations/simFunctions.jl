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


function assign_clusters(X, centroids)
    return [argmin([norm(X[:, i] - centroids[:, j]) for j in 1:size(centroids, 2)]) for i in 1:size(X, 2)]
end

# note rmse will hvae to be inverted
function findEquilibrated(;rmse::Vector{Float64}, rind::Vector{Float64}, tol::Float64=0.01)
    rmseEq = abs.(rmse .- minimum(rmse)) .<= (tol.* minimum(rmse))
    rindEq = abs.(rind .- maximum(rind)) .<= (tol.* maximum(rind))
    
    # 'and' v1 and v2 bit vectors together elementwise
    eq = rmseEq .& rindEq
    eq = findfirst(x -> x == 1, eq)
    if isnothing(eq)
        eq = 1
    end
    return eq
end


function simData(rng::AbstractRNG; N::Int=100, fractions::Vector{Float64}=[0.25, 0.25, 0.25, 0.25], variance::Real=1.0, interEffect::Real=1.0, common::Float64=1.0, plotSim::Bool=false, xdiff::Real=1.0, dims::Int=2)
    
    nclusts = length(fractions) # Create a DataFrame with two normally distributed columns, X1 and X2
    
    df = DataFrame(Dict( "X" * string(i) => rand(rng, Normal(0,1), N) for i in 1:dims ))

    # assign groups
    df.group .= 0
    
    lastsub =  0
    for g in 1:nclusts
        g1 = floor(Int, N * fractions[g])
        firstsub = lastsub+1
        lastsub = lastsub + g1
        if g != nclusts
            df[firstsub:lastsub, :group] .= g
        else 
            df[firstsub:end, :group] .= g
        end
    end
    
    # add xdiff * group to all columns starting with "X"
    df[:, r"^X"] .= df[:, r"^X"] .+ (df.group .* xdiff)
    
    # group effect
    groupEffect = [quantile(Normal(common, interEffect), i/(nclusts + 1)) for i in 1:nclusts]
    df.groupEffect = map(x -> groupEffect[x], df.group)
    
    df.mean .= (dims .* df.groupEffect) .+ sum(Matrix(df.groupEffect .* df[:,r"^X"]); dims = 2)
    #df.mean = df.groupEffect .+ df.X1 .* df.X2
    
    # Generate the Y column as the sum of globalMean, groupDeviations, and noise
    df.Y = df.mean .+ rand(rng, Normal(0, variance), N)
    df.inter .= 1
    
    if plotSim & dims == 2
        # X marginal
        px1 = @df df density(:X1, group = :group, fillopacity = 1/2, title = "X1 marginal")
        px2 = @df df density(:X2, group = :group, fillopacity = 1/2, title = "X1 marginal")
        py = @df df density(:Y, group = :group, fillopacity = 1/2, title = "Y marginal"; permute = (:x, :y))
        xflip!(py)
        pyx1 = @df df scatter(:X1, :Y, group = :group, opacity = 1/2, title = "Y|X1")
        pyx2 = @df df scatter(:X2, :Y, group = :group, opacity = 1/2, title = "Y|X2")
        p0 = plot(legend=false,grid=false,foreground_color_subplot=:white)  #create subplot [2,2] to be blank
        l = @layout [
        ymarg{0.3w} scat1 scat2
        empt xmarg{0.3h} x2marg
        ]
        display(plot(py, pyx1, pyx2, p0, px1, px2, layout = l))
    end
    return df
end

#df = simData(rng; n=n, fractions=fractions, variance=variance, interEffect=interEffect, common=common, plotSim = true, xdiff=xdiff)


function simExperiment(rng::AbstractRNG, n::Int=100, fractions::Vector{Float64}=[0.25,0.25,0.25,0.25], variance::Real=1.0, interEffect::Float64=1.0, common::Float64=1.0, plotFit::Bool=false, niters::Int=1000; plotSim::Bool = false, xdiff::Real= 1.0, dims::Int=2)

    # common = interEffect : promising
    # common < interEffect : 
    # common > interEffect : somewhat promising

    local result = 0
    # Simulate data
    df = simData(rng; N, fractions, variance, interEffect, common, plotSim=plotSim, xdiff = xdiff, dims = dims)
    dfoos = simData(rng; N, fractions, variance, interEffect, common, plotSim=false, xdiff = xdiff, dims = dims)
    # standardize
    Ymean = mean(df.Y)
    Ystd  = std(df.Y)
    Ystand = (df.Y .- Ymean) ./ Ystd
    Ystandoos = (dfoos.Y .- Ymean) ./ Ystd
    
    # Fit 
    X = Matrix(df[:, Cols("inter", r"X")][:, Not(r"eff")])
    Xoos = Matrix(dfoos[:, Cols("inter", r"X")][:, Not(r"eff")])
    model = Model_PPMx(Ystand, X, df.group, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
    # closeGroup = df.group
    # closeGroup[1:10] .= 3
    #model = Model_PPMx(Ystand, X, closeGroup, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=false)
    # groupEffect = [quantile(Normal(common, interEffect), i/(length(fractions) + 1)) for i in 1:length(fractions)]
    # for c in 1:length(unique(model.state.C))
    #     model.state.lik_params[c].beta .= groupEffect[c] # ./Ystd
    # end
    # 
    # if length(fractions) == 2
    #     lcoh = repeat([148.5], length(fractions))
    #     #lcoh = repeat([152], length(fractions))
    # elseif length(fractions) == 4
    #     lcoh = repeat([63], length(fractions))
    # elseif length(fractions) == 7
    #     # for 100 sample
    #     # lcoh = repeat([200], length(fractions))
    #     # for 500 sampel
    #     lcoh = repeat([1000], length(fractions))
    # else 
    #     lcoh = model.state.lcohesions
    # end

    #model.state.lcohesions .= lcoh

    ################################################
    ## For two clusters Common = inter = xdiff = 1.0
    # model.state.Xstats = [
    #     [Similarity_NN_stats(108, 108.0, 108.0)
    #       Similarity_NN_stats(108, 35, 117)
    #       Similarity_NN_stats(108, 37, 118)],
    #     [Similarity_NN_stats(92, 92.0, 92.0)
    #       Similarity_NN_stats(92, -35, 81)
    #       Similarity_NN_stats(92, -37, 80)
    # ]] # super help
    # model.state.lcohesions = [396, 322] # super help
    #model.state.lcohesions = [148.5, 148.5] # trial for 2
    # model.state.lcohesions = [63, 63, 63, 63] # trial for 4
    # model.state.prior_mean_beta = [10, 0.3, -5.5] # didn't help.
    # model.state.lsimilarities = [[-102, -154, -154],
    #[-87, -120, -119]] # didn't help
    ################################################
    ################################################
    sim = mcmc!(model, niters; mixDPM=true)
    rindMixvec = [Clustering.randindex(s[:C], df.group)[2] for s in sim]
    X2 = Matrix(df[:, Cols("inter", r"X")][:, Not(r"eff")])
    X2oos = Matrix(dfoos[:, Cols("inter", r"X")][:, Not(r"eff")])
    model2 = Model_PPMx(Ystand, X2, df.group, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=false)
    sim2 = mcmc!(model2, niters; mixDPM=false)
    trimid = Int(niters/2)
    thin = 1
    sim = sim[trimid:thin:end]
    sim2 = sim2[trimid:thin:end]
    Ypred1, Cpred1 = postPred(X, model, sim)
    Ypred2, Cpred2 = postPred(X2, model2, sim2)
    Ypred1oos, Cpred1oos = postPred(Xoos, model, sim)
    Ypred2oos, Cpred2oos = postPred(X2oos, model2, sim2)
    
    # clustering
    rindMixvec = [Clustering.randindex(s[:C], df.group)[2] for s in sim]
    rindDPMvec = [Clustering.randindex(s[:C], df.group)[2] for s in sim2]
    rindMixvecoos = map(x -> Clustering.randindex(dfoos.group, x)[2], eachrow(Cpred1oos))
    rindDPMvecoos = map(x -> Clustering.randindex(dfoos.group, x)[2], eachrow(Cpred2oos))
    
    
    # transform back
    Ypred1 = Ypred1 .* Ystd .+ Ymean
    Ypred2 = Ypred2 .* Ystd .+ Ymean
    Ypred1oos = Ypred1oos .* Ystd .+ Ymean
    Ypred2oos = Ypred2oos .* Ystd .+ Ymean
    
    
    # predictive
    resid1 = df.Y .- Ypred1'
    resid2 = df.Y .- Ypred2'
    resid1oos = dfoos.Y .- Ypred1oos'
    resid2oos = dfoos.Y .- Ypred2oos'
    
    # simulated variance
    rmseMix = sqrt.(mean(resid1.^2, dims = 1))[1,:]
    rmseDPM = sqrt.(mean(resid2.^2, dims = 1))[1,:]
    
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
    
    rmseMixoos = sqrt.(mean(resid1oos[mixEq:end].^2, dims = 1))[1,:]
    rmseDPMoos = sqrt.(mean(resid2oos[dpmEq:end].^2, dims = 1))[1,:]
    midMix = mean(rmseMix)
    midDPM = mean(rmseDPM)
    midMixoos = mean(rmseMixoos)
    midDPMoos = mean(rmseDPMoos)
    
    
    # Kolmogorov-Smirnov pvalue for marginal distribution of Y
    # ks= pvalue(ApproximateTwoSampleKSTest(mean(Ypred1, dims= 1)[1,:], mean(Ypred2, dims= 1)[1,:]))
    sim = sim[mixEq:5:end]
    sim2 = sim2[dpmEq:5:end]

    commonBeta0 = [s[:prior_mean_beta][1] for s in sim] .* Ystd
    meanBeta0 = mean(commonBeta0)
    commonBeta1 = [s[:prior_mean_beta][2] for s in sim] .* Ystd
    meanBeta1 = mean(commonBeta1)
    
    # is each obs in the 0.05 0.95 quantiles of the posterior predictive?
    bayesPmix = median(mean(df.Y .<= Ypred1', dims = 2))
    bayesPDPM = median(mean(Ystand .<= Ypred2', dims = 2))
    bayesPmixoos = median(mean(df.Y .<= Ypred1oos', dims = 2))
    bayesPDPMoos = median(mean(Ystandoos .<= Ypred2oos', dims = 2))
    
    # Add in Kmeans with full interaction effect model
    # - select best clustering based off X
    # - interaction model
    kclust = argmin([kmeans(X', i).totalcost for i in 1:6])
    kmodel = kmeans(X', kclust)
    df.kclust = kmodel.assignments
    dfoos.kclust = assign_clusters(Xoos', kmodel.centers)
    
    rindKclust = Clustering.randindex(df.kclust, df.group)[2]
    rindKclustoos = Clustering.randindex(dfoos.kclust, dfoos.group)[2]
    # linear model with Y vs X1, X2
    clustlm = lm(@formula(Y ~ (X1) * kclust), df)
    kmeanMSE = sqrt(mean(residuals(clustlm).^2))
    kmeanMSEoos= sqrt(mean((predict(clustlm, dfoos) - dfoos.Y).^2))
    meanBetakclust1 = coef(clustlm)[2]
    
    ncMix = mode([maximum(s[:C]) for s in sim ])
    ncDPM = mode([maximum(s[:C]) for s in sim ])
    ncK = mode([maximum(s[:C]) for s in sim ])
    
    nclusts = length(fractions)
    Mix_beta1_c1 = [s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == ncMix] .* Ystd
    Mix_beta1_c2 = [s[:lik_params][2][:beta][2] for s in sim if maximum(s[:C]) == ncMix] .* Ystd
    dpm_beta1_c1 = [s[:lik_params][1][:beta][1] for s in sim if maximum(s[:C]) == ncDPM] .* Ystd
    dpm_beta1_c2 = [s[:lik_params][2][:beta][1] for s in sim if maximum(s[:C]) == ncDPM] .* Ystd
    
    
    result = DataFrame(
      # mixture model
      rind_Mix = rindMix,
      rindMixoos = rindMixoos,
      midMix = midMix,
      midMixoos = midMixoos,
      bayesPmix = bayesPmix,
      bayesPmixoos = bayesPmixoos,
      meanBeta1 = meanBeta1,
      Mix_beta1_c1 = median(Mix_beta1_c1),
      Mix_beta1_c2 = median(Mix_beta1_c2),
      mixEq = mixEq,
      
      # PPMx model
      rind_DPM = rindDPM,
      rind_DPMoos = rindDPMoos,
      midDPM = midDPM,
      midDPMoos = midDPMoos,
      bayesPDPM = bayesPDPM,
      bayesPDPMoos = bayesPDPMoos,
      dpm_beta1_c1 = median(dpm_beta1_c1),
      dpm_beta1_c2 = median(dpm_beta1_c2),
      dpmEq = dpmEq,
      
      # rind_K
      rind_K = rindKclust,
      rind_Koos = rindKclustoos,
      kmean_MSE = kmeanMSE, 
      kmean_MSEoos = kmeanMSEoos, 
      meanBetakclust1 = meanBetakclust1,
      
      # number of clusts
      ncMix = ncMix,
      ncDPM = ncDPM,
      ncK = ncK,
      
      # setup
      n=n, fractions = string(fractions), variance = variance, 
      interEffect = interEffect, common = common
    )
    
    if plotFit
      # 4 
      # clustering 
      histogram([rmseMix, rmseDPM], label = ["PPMx-shared" "PPMx"], fillalpha = 0.33, title = "RMSE (4 clusters)")
      vline!([kmeanMSE], label = "K-means")
      display(current())
      
      # prediction
      histogram([rindMixvec, rindDPMvec], label = ["PPMx-shared" "PPMx"], fillalpha = 0.33, title = "Adjusted Rand Index (4 clusters)")
      vline!([rindKclust], label = "K-means")
      display(current())
      
      # inference
      # Common
      histogram(commonBeta1, label = L"\beta_1", title = "Common Effects estimates", fillalpha = 0.33)
      vline!([common, -common], label = "True", color = "black")
      xlims!(-3 * common, 3*common)
      display(current())
      
      # specific
      plot(
        histogram(Mix_beta1_c1, label = "Cluster 1, " * L"\beta_1"),
        histogram(Mix_beta1_c2, label = "Cluster 2, " * L"\beta_1"),
    #    histogram(Mix_beta2_c1, label = "Cluster 1, " * L"\beta_2"),
    #    histogram(Mix_beta2_c2, label = "Cluster 2, " * L"\beta_2"),
        plot_title = "Cluster specific estimates \n Mixture"
        )
      display(current())
      plot(
        histogram(dpm_beta1_c1, label = "Cluster 1, " * L"\beta_1"),
        histogram(dpm_beta1_c2, label = "Cluster 2, " * L"\beta_1"),
        plot_title = "Cluster specific estimates \n PPMx"
        )
      display(current())
    end
    
    return result
end
#results = simExperiment(rng, n, fractions, variance, interEffect, common, true, niters, xdiff = xdiff)
