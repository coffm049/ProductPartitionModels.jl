using StatsBase
using Statistics
using StatsPlots
using Distributions
using Random
using Plots
using DataFrames
using LinearAlgebra
using ProductPartitionModels
using Clustering
using JLD2
using CSV
using GLM
using HypothesisTests
using LaTeXStrings
import ProductPartitionModels.postPred

function assign_clusters(X, centroids)
    return [argmin([norm(X[:, i] - centroids[:, j]) for j in 1:size(centroids, 2)]) for i in 1:size(X, 2)]
end

function postPred(Xpred::Union{Matrix{T},Matrix{Union{T,Missing}}},
    model::Model_PPMx,
    sims::Vector{Dict{Symbol,Any}},
    update_params::Vector{Symbol}=[:mu, :sig, :beta, :mu0, :sig0]) where {T<:Real}

    ## currently assumes cohesion and similarity parameters are fixed
    ## treats each input as the n+1th observation with no consideration of them clustering together
    ## does not update the likelihood centering with the prediction obs, stats

    n_pred, p_pred = size(Xpred)
    p_pred == model.p || throw("Xpred and original X have different numbers of predictors.")

    obsXIndx_pred = [ObsXIndx(Xpred[i, :]) for i in 1:n_pred]

    n_sim = length(sims)

    Cpred = Matrix{Int}(undef, n_sim, n_pred)
    Ypred = Matrix{typeof(model.y[1])}(undef, n_sim, n_pred)
    Mean_pred = Matrix{typeof(model.y[1])}(undef, n_sim, n_pred)

    lcohes1 = log_cohesion(Cohesion_CRP(model.state.cohesion.logα, 1, true))
    x_mean_empty, x_sd_empty = aux_moments_empty(model.state.similarity)

    for ii in 1:n_sim

        lcohesions, Xstats, lsimilarities = get_lcohlsim(sims[ii][:C], model.X, model.state.cohesion, model.state.similarity)
        K = length(lcohesions)
        S = StatsBase.counts(sims[ii][:C], K)

        if typeof(model.state.lik_params[1]) <: LikParams_PPMxReg
            Xbars = Matrix{typeof(model.y[1])}(undef, K, model.p)
            Sds = Matrix{typeof(model.y[1])}(undef, K, model.p)

            for k in 1:K
                Xbars[k, :], Sds[k, :] = aux_moments_k(Xstats[k], model.state.similarity)
            end
        end

        for i in 1:n_pred

            lw = predWeights(i, Xpred, lcohesions, Xstats, lsimilarities, K, S, model, lcohes1)

            # sample membership for obs i
            C_i = StatsBase.sample(StatsBase.Weights(exp.(lw)))
            if C_i > K
                C_i = 0
            end
            Cpred[ii, i] = C_i

            # draw y value
            if C_i > 0
                mean_now = deepcopy(sims[ii][:lik_params][C_i][:mu])
                sig2_now = sims[ii][:lik_params][C_i][:sig]^2

                if typeof(model.state.lik_params[1]) <: LikParams_PPMxReg
                    if obsXIndx_pred[i].n_mis > 0
                        sig2_now += sum(sims[ii][:lik_params][C_i][:beta][obsXIndx_pred[i].indx_mis] .^ 2)
                    end

                    if obsXIndx_pred[i].n_obs > 0
                        z = (Xpred[i, obsXIndx_pred[i].indx_obs] - Xbars[C_i, obsXIndx_pred[i].indx_obs]) ./ Sds[C_i, obsXIndx_pred[i].indx_obs]
                        mean_now += z' * sims[ii][:lik_params][C_i][:beta][obsXIndx_pred[i].indx_obs]
                    end
                end

            else

                basenow = deepcopy(model.state.baseline)

                if (:mu0 in update_params)
                    basenow.mu0 = deepcopy(sims[ii][:baseline][:mu0])
                end

                if (:sig0 in update_params)
                    basenow.sig0 = deepcopy(sims[ii][:baseline][:sig0])
                end

                lik_params_new = simpri_lik_params(basenow,
                        model.p, model.state.lik_params[1], update_params
                )

                mean_now = deepcopy(lik_params_new.mu)
                sig2_now = lik_params_new.sig^2

                if typeof(model.state.lik_params[1]) <: LikParams_PPMxReg
                    if obsXIndx_pred[i].n_mis > 0
                        sig2_now += sum(lik_params_new.beta[obsXIndx_pred[i].indx_mis] .^ 2)
                    end

                    if obsXIndx_pred[i].n_obs > 0
                        z = (Xpred[i, obsXIndx_pred[i].indx_obs] .- x_mean_empty) ./ x_sd_empty
                        mean_now += z' * lik_params_new.beta[obsXIndx_pred[i].indx_obs]
                    end
                end
            end

            Mean_pred[ii, i] = deepcopy(mean_now)
            Ypred[ii, i] = randn() .* sqrt(sig2_now) + mean_now

        end
    end

    return Ypred, Cpred, Mean_pred
end


function postPred(model::Model_PPMx,
    sims::Vector{Dict{Symbol,Any}})

    ## currently assumes cohesion and similarity parameters are fixed
    ## treats each input as the n+1th observation with no consideration of them clustering together
    ## does not update the likelihood centering with the prediction obs, stats

    n_sim = length(sims)

    Ypred = Matrix{typeof(model.y[1])}(undef, n_sim, model.n)
    Mean_pred = Matrix{typeof(model.y[1])}(undef, n_sim, model.n)

    for ii in 1:n_sim

        lcohesions, Xstats, lsimilarities = get_lcohlsim(sims[ii][:C], model.X, model.state.cohesion, model.state.similarity)
        K = length(lcohesions)
        # S = StatsBase.counts(sims[ii][:C], K)

        if typeof(model.state.lik_params[1]) <: LikParams_PPMxReg
            Xbars = Matrix{typeof(model.y[1])}(undef, K, model.p)
            Sds = Matrix{typeof(model.y[1])}(undef, K, model.p)

            for k in 1:K
                Xbars[k, :], Sds[k, :] = aux_moments_k(Xstats[k], model.state.similarity)
            end
        end

        for i in 1:model.n
            # draw y value
            C_i = deepcopy(sims[ii][:C][i])
            mean_now = deepcopy(sims[ii][:lik_params][C_i][:mu])
            sig2_now = sims[ii][:lik_params][C_i][:sig]^2

            if typeof(model.state.lik_params[1]) <: LikParams_PPMxReg
                if model.obsXIndx[i].n_mis > 0
                    sig2_now += sum(sims[ii][:lik_params][C_i][:beta][model.obsXIndx[i].indx_mis] .^ 2)
                end
            
                if model.obsXIndx[i].n_obs > 0
                    z = (model.X[i, model.obsXIndx[i].indx_obs] - Xbars[C_i, model.obsXIndx[i].indx_obs]) ./ Sds[C_i, model.obsXIndx[i].indx_obs]
                    mean_now += z' * sims[ii][:lik_params][C_i][:beta][model.obsXIndx[i].indx_obs]
                end
            end

            Mean_pred[ii, i] = deepcopy(mean_now)
            Ypred[ii, i] = randn() .* sqrt(sig2_now) + mean_now

        end
    end

    return Ypred, Mean_pred
end



function simData(rng::AbstractRNG, n::Int=100, fractions::Vector{Float64}=[0.25, 0.25, 0.25, 0.25], variance::Real=1.0, interEffect::Real=1.0, common::Float64=1.0; plotSim::Bool=false)
    
    nclusts = length(fractions)
    # Create a DataFrame with two normally distributed columns, X1 and X2
    df = DataFrame(X1=randn(rng, n), X2=randn(rng, n))
    
    # assign groups
    df.group .= 0
    
    lastsub =  0
    for g in 1:nclusts
        g1 = floor(Int, n * fractions[g])
        firstsub = lastsub+1
        lastsub = lastsub + g1
        if g != nclusts
            df[firstsub:lastsub, :group] .= g
        else 
            df[firstsub:end, :group] .= g
        end
    end
    
    df.X1 .= df.X1 .+ (df.group ./ 1)
    df.X2 .= df.X2 .+ (df.group ./ 1)
    
    # common effect 
    df.commonMean = common .+ df.X1 .* -common .+ df.X2 .*-common
    
    df.groupEffect = ((df.group .- mean(df.group))  ./ 4 ) .* interEffect
    
    df.groupDeviations = interEffect .* df.groupEffect  .+ (df.X1 .* df.groupEffect) .- (df.X2 .* df.groupEffect)
    
    # Generate the Y column as the sum of globalMean, groupDeviations, and noise
    df.Y = df.commonMean .+ df.groupDeviations .+ variance .* randn(rng, n)
    df.Y = (df.Y .- mean(df.Y)) ./ std(df.Y)
    df.inter .= 1

    df.X1eff .= common
    df.X2eff .= -common

    if plotSim
        # X marginal
        px1 = @df df density(:X1, group = :group, fillopacity = 1/2, title = "X1 marginal")
        px2 = @df df density(:X2, group = :group, fillopacity = 1/2, title = "X2 marginal")
        py = @df df density(:Y, group = :group, fillopacity = 1/2, title = "Y marginal")
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


function simExperiment(rng::AbstractRNG, n::Int=100, fractions::Vector{Float64}=[0.25,0.25,0.25,0.25], variance::Real=1.0, interEffect::Float64=1.0, common::Float64=1.0, plotFit::Bool=false, niters::Int=1000; plotSim::Bool = false)
    
    local result = 0
    try
      # Simulate data
      df = simData(rng, n, fractions, variance, interEffect, common, plotSim=plotSim)
      dfoos = simData(rng, n, fractions, variance, interEffect, common)

      # Fit 
      X = Matrix(df[:, ["inter", "X1", "X2"]])
      Xoos = Matrix(dfoos[:, ["inter", "X1", "X2"]])
      model = Model_PPMx(df.Y, X, 1, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
      sim = mcmc!(model, niters, MixDPM=true)
      X2 = Matrix(df[:, ["X1", "X2"]])
      X2oos = Matrix(dfoos[:, ["X1", "X2"]])
      ## Standardize
      Ymean = mean(df.Y)
      Ystd  = std(df.Y)
      Ystand = (df.Y .- Ymean) ./ Ystd
      Ystandoos = (dfoos.Y .- Ymean) ./ Ystd
      model2 = Model_PPMx(Ystand, X2, 1, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
      sim2 = mcmc!(model2, niters, MixDPM=false)
      
      trimid=Int(niters/4)
      thin=5
      
      sim = sim[trimid:thin:end]
      sim2 = sim2[trimid:thin:end]
      Ypred1, Cpred1 = postPred(X, model, sim)
      Ypred2, Cpred2 = postPred(X2, model2, sim2)
      Ypred1oos, Cpred1oos = postPred(Xoos, model, sim)
      Ypred2oos, Cpred2oos = postPred(X2oos, model2, sim2)
      
      # clustering
      rindMixvec = [Clustering.randindex(s[:C], df.group)[1] for s in sim]
      rindMix = median(rindMixvec)
      rindDPMvec = [Clustering.randindex(s[:C], df.group)[1] for s in sim2]
      rindDPM = median(rindDPMvec)
      rindMixvecoos = map(x -> Clustering.randindex(dfoos.group, x)[1], eachrow(Cpred1oos))
      rindMixoos = median(rindMixvecoos)
      rindDPMvecoos = map(x -> Clustering.randindex(dfoos.group, x)[1], eachrow(Cpred2oos))
      rindDPMoos = median(rindDPMvecoos)

      # predictive
      resid1 = df.Y .- Ypred1'
      resid2 = df.Y .- ((Ypred2' * Ystd) .+ Ymean)
      resid1oos = dfoos.Y .- Ypred1oos'
      resid2oos = dfoos.Y .- ((Ypred2oos' * Ystd) .+ Ymean)
      # simulated variance
      simresid = df.Y .- (df.commonMean .+ df.groupDeviations)
      rmseMix = sqrt.(mean(resid1.^2, dims = 1))[1,:]
      rmseDPM = sqrt.(mean(resid2.^2, dims = 1))[1,:]
      rmseMixoos = sqrt.(mean(resid1oos.^2, dims = 1))[1,:]
      rmseDPMoos = sqrt.(mean(resid2oos.^2, dims = 1))[1,:]
      midMix = median(rmseMix)
      midDPM = median(rmseDPM)
      midMixoos = median(rmseMixoos)
      midDPMoos = median(rmseDPMoos)
      
      # Kolmogorov-Smirnov pvalue for marginal distribution of Y
      # ks= pvalue(ApproximateTwoSampleKSTest(mean(Ypred1, dims= 1)[1,:], mean(Ypred2, dims= 1)[1,:]))
      
      
      commonBeta1 = [s[:prior_mean_beta][2] for s in sim]
      meanBeta1 = mean(commonBeta1)
      commonBeta2 = [s[:prior_mean_beta][3] for s in sim]
      meanBeta2 = mean(commonBeta2)
      
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
      
      rindKclust = Clustering.randindex(df.kclust, df.group)[1]
      rindKclustoos = Clustering.randindex(dfoos.kclust, dfoos.group)[1]
      # linear model with Y vs X1, X2
      clustlm = lm(@formula(Y ~ (X1 + X2) * kclust), df)
      kmeanMSE = sqrt(mean(residuals(clustlm).^2))
      kmeanMSEoos= sqrt(mean((predict(clustlm, dfoos) - dfoos.Y).^2))
      meanBetakclust14 = coef(clustlm)[2]
      meanBetakclust24 = coef(clustlm)[3]
      
      
      Mix_beta1_c1_4 = [s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == 2]
      Mix_beta2_c1_4 = [s[:lik_params][1][:beta][3] for s in sim if maximum(s[:C]) == 2]
      Mix_beta1_c2_4 = [s[:lik_params][2][:beta][2] for s in sim if maximum(s[:C]) == 2]
      Mix_beta2_c2_4 = [s[:lik_params][2][:beta][3] for s in sim if maximum(s[:C]) == 2]
      dpm_beta1_c1_4 = [s[:lik_params][1][:beta][1] for s in sim if maximum(s[:C]) == 2]
      dpm_beta2_c1_4 = [s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == 2]
      dpm_beta1_c2_4 = [s[:lik_params][2][:beta][1] for s in sim if maximum(s[:C]) == 2]
      dpm_beta2_c2_4 = [s[:lik_params][2][:beta][2] for s in sim if maximum(s[:C]) == 2]
      
      
      # 2 clusters
      df = df[(df.group .== 1) .| (df.group .== 4), :]
      dfoos = dfoos[(dfoos.group .== 1) .| (dfoos.group .== 4), :]
      # n, minorFrac, variance, interEffect, common)
      X = Matrix(df[:, ["inter", "X1", "X2"]])
      Xoos = Matrix(dfoos[:, ["inter", "X1", "X2"]])
      model = Model_PPMx(df.Y, X, 1, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
      sim = mcmc!(model, niters, MixDPM=true)
      X2 = Matrix(df[:, ["X1", "X2"]])
      X2oos = Matrix(dfoos[:, ["X1", "X2"]])
      Ymean = mean(df.Y)
      Ystd  = std(df.Y)
      Ystand = (df.Y .- Ymean) ./ Ystd
      Ystandoos = (dfoos.Y .- Ymean) ./ Ystd
      model2 = Model_PPMx(Ystand, X2, 1, similarity_type=:NN, sampling_model=:Reg, init_lik_rand=true)
      sim2 = mcmc!(model2, niters, MixDPM=false)
      
      sim = sim[trimid:thin:end]
      sim2 = sim2[trimid:thin:end]
      Ypred1, Cpred1 = postPred(model, sim)
      Ypred2, Cpred2 = postPred(model2, sim2)
      Ypred1oos, Cpred1oos = postPred(Xoos, model, sim)
      Ypred2oos, Cpred2oos = postPred(X2oos, model2, sim2)
      
      # clustering
      rindMix2vec = [Clustering.randindex(s[:C], df.group)[1] for s in sim]
      rindMix2 = median(rindMix2vec)
      rindDpm2vec = [Clustering.randindex(s[:C], df.group)[1] for s in sim2]
      rindDPM2 = median(rindDpm2vec)
      rindMix2vecoos = map(x -> Clustering.randindex(dfoos.group, x)[1], eachrow(Cpred1oos))
      rindMix2oos = median(rindMix2vecoos)
      rindDPM2vecoos = map(x -> Clustering.randindex(dfoos.group, x)[1], eachrow(Cpred2oos))
      rindDPM2oos = median(rindDPM2vecoos)
   

      # predictive
      resid1 = df.Y .- Ypred1'
      resid2 = df.Y .- ((Ypred2' * Ystd) .+ Ymean)
      resid1oos = dfoos.Y .- Ypred1oos'
      resid2oos = dfoos.Y .- ((Ypred2oos' * Ystd) .+ Ymean)
      rmseMix2 = sqrt.(mean(resid1.^2, dims = 1))[1,:]
      rmseDPM2 = sqrt.(mean(resid2.^2, dims = 1))[1,:]
      rmseMixoos = sqrt.(mean(resid1oos.^2, dims = 1))[1,:]
      rmseDPMoos = sqrt.(mean(resid2oos.^2, dims = 1))[1,:]
      midMix2 = median(rmseMix)
      midDPM2 = median(rmseDPM)
      midMix2oos = median(rmseMixoos)
      midDPM2oos = median(rmseDPMoos)
      
      
      commonBeta12 = [s[:prior_mean_beta][2] for s in sim]
      meanBeta12 = mean(commonBeta12)
      commonBeta22 = [s[:prior_mean_beta][3] for s in sim]
      meanBeta22 = mean(commonBeta12)
      meanBetakclust12 = coef(clustlm)[2]
      meanBetakclust22 = coef(clustlm)[3]
      
      # kmean
      kclust = argmin([kmeans(X', i).totalcost for i in 1:6])
      kmodel = kmeans(X', kclust)
      df.kclust = kmodel.assignments
      dfoos.kclust = assign_clusters(Xoos', kmodel.centers)
      
      rindKclust_2 = Clustering.randindex(df.kclust, df.group)[1]
      rindKclust_2oos = Clustering.randindex(dfoos.kclust, dfoos.group)[1]
      # linear model with Y vs X1, X2
      clustlm = lm(@formula(Y ~ (X1 + X2) * kclust), df)
      kmeanMSE_2 = sqrt(mean(residuals(clustlm).^2))
      kmeanMSEoos_2 = sqrt(mean((predict(clustlm, dfoos) - dfoos.Y).^2))
      meanBetakclust14 = coef(clustlm)[2]
      meanBetakclust24 = coef(clustlm)[3]
      
      Mix_beta1_c1_2 = [s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == 2]
      Mix_beta2_c1_2 = [s[:lik_params][1][:beta][3] for s in sim if maximum(s[:C]) == 2]
      Mix_beta1_c2_2 = [s[:lik_params][2][:beta][2] for s in sim if maximum(s[:C]) == 2]
      Mix_beta2_c2_2 = [s[:lik_params][2][:beta][3] for s in sim if maximum(s[:C]) == 2]
      dpm_beta1_c1_2 = [s[:lik_params][1][:beta][1] for s in sim if maximum(s[:C]) == 2]
      dpm_beta2_c1_2 = [s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == 2]
      dpm_beta1_c2_2 = [s[:lik_params][2][:beta][1] for s in sim if maximum(s[:C]) == 2]
      dpm_beta2_c2_2 = [s[:lik_params][2][:beta][2] for s in sim if maximum(s[:C]) == 2]
      
      result = DataFrame(
        # mixture model (4 clust)
        rind_Mix_4 = rindMix,
        rindMixoos_4 = rindMixoos,
        midMix = midMix,
        midMixoos = midMixoos,
        bayesPmix = bayesPmix,
        bayesPmixoos = bayesPmixoos,
        meanBeta1 = meanBeta1, meanBeta2 = meanBeta2, 
        Mix_beta1_c1_4 = mean(Mix_beta1_c1_4),
        Mix_beta2_c1_4 = mean(Mix_beta2_c1_4),
        Mix_beta1_c2_4 = mean(Mix_beta1_c2_4),
        Mix_beta2_c2_4 = mean(Mix_beta2_c2_4),
        
        # PPMx model (4 clust)
        rind_DPM_4 = rindDPM,
        rind_DPMoos_4 = rindDPMoos,
        midDPM = midDPM,
        midDPMoos = midDPMoos,
        bayesPDPM = bayesPDPM,
        bayesPDPMoos = bayesPDPMoos,
        dpm_beta1_c1_4 = mean(dpm_beta1_c1_4),
        dpm_beta2_c1_4 = mean(dpm_beta2_c1_4),
        dpm_beta1_c2_4 = mean(dpm_beta1_c2_4),
        dpm_beta2_c2_4 = mean(dpm_beta2_c2_4),
        
        # rind_K_4
        rind_K_4 = rindKclust,
        rind_K_4oos = rindKclustoos,
        kmean_MSE_4 = kmeanMSE, 
        kmean_MSE_4oos = kmeanMSEoos, 
        meanBetakclust14 = meanBetakclust14,
        meanBetakclust2 = meanBetakclust24,
        
        # 4 clust compare
        #ks = ks,
        
        # mixture model (2 clust)
        rind_Mix_2 = rindMix2,
        rind_Mix_2oos = rindMix2oos,
        midMix2 = midMix2,
        midMix2oos = midMix2oos,
        Mix_beta1_c1_2 = mean([s[:lik_params][1][:beta][2] for s in sim if maximum(s[:C]) == 2]),
        Mix_beta2_c1_2 = mean([s[:lik_params][1][:beta][3] for s in sim if maximum(s[:C]) == 2]),
        Mix_beta1_c2_2 = mean([s[:lik_params][2][:beta][2] for s in sim if maximum(s[:C]) == 2]),
        Mix_beta2_c2_2 = mean([s[:lik_params][2][:beta][3] for s in sim if maximum(s[:C]) == 2]),
        
        # PPMx model (2 clust)
        rind_DPM_2 = rindDPM2,
        rind_DPM_2oos = rindDPM2oos,
        midDPM2 = midDPM2,
        midDPM2oos = midDPM2oos,
        meanBeta12 = meanBeta12, meanBeta22 = meanBeta22,
        dpm_beta1_c1_2 = mean([s[:lik_params][1][:beta][1] for s in sim2 if maximum(s[:C]) == 2]),
        dpm_beta2_c1_2 = mean([s[:lik_params][1][:beta][2] for s in sim2 if maximum(s[:C]) == 2]),
        dpm_beta1_c2_2 = mean([s[:lik_params][2][:beta][1] for s in sim2 if maximum(s[:C]) == 2]),
        dpm_beta2_c2_2 = mean([s[:lik_params][2][:beta][2] for s in sim2 if maximum(s[:C]) == 2]),
        
        # 2 clust compare
        rind_K_2 = rindKclust_2,
        rind_K_2oos = rindKclust_2oos,
        kmean_MSE_2 = kmeanMSE_2, 
        kmean_MSE_2oos = kmeanMSEoos_2, 
        
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
        histogram(commonBeta1, label = L"\beta_1", title = "Common Effects estimates")
        histogram!(commonBeta2, label = L"\beta_2")
        vline!([common, -common], label = "True", color = "black")
        vline!([meanBetakclust14, meanBetakclust24], label = "kmeans")
        xlims!(-3 * common, 3*common)
        display(current())
        
        # specific
        plot(
          histogram(Mix_beta1_c1_4, label = "Cluster 1, " * L"\beta_1"),
          histogram(Mix_beta1_c2_4, label = "Cluster 2, " * L"\beta_1"),
          histogram(Mix_beta2_c1_4, label = "Cluster 1, " * L"\beta_2"),
          histogram(Mix_beta2_c2_4, label = "Cluster 2, " * L"\beta_2"),
          plot_title = "Cluster specific estimates \n (4 clusters only showing first 2) \n Mixture"
          )
        display(current())
        plot(
          histogram(dpm_beta1_c1_4, label = "Cluster 1, " * L"\beta_1"),
          histogram(dpm_beta1_c2_4, label = "Cluster 2, " * L"\beta_1"),
          histogram(dpm_beta2_c1_4, label = "Cluster 1, " * L"\beta_2"),
          histogram(dpm_beta2_c2_4, label = "Cluster 2, " * L"\beta_2"),
          plot_title = "Cluster specific estimates \n (4 clusters only showing first 2) \n PPMx"
          )
        display(current())
        
        # 2
        # clustering
        histogram([rindMix2vec, rindDpm2vec], label = ["PPMx-shared" "PPMx"], fillalpha = 0.33, title = "Adjusted Rand Index (2 clusters)")
        vline!([rindKclust_2], label = "K-means")
        display(current())
        
        # prediction
        histogram([rmseMix2, rmseDPM2], label = ["PPMx-shared" "PPMx"], fillalpha = 0.33, title = "RMSE (2 clusters)")
        vline!([mean(rmseMix2), mean(rmseDPM2), kmeanMSE_2], label = ["PPMx-shared", "PPMx", "K-means"])
        display(current())
        
        # inference
        # Common
        histogram(commonBeta12, label = L"\beta_1", title = "Common Effects estimates")
        histogram!(commonBeta22, label = L"\beta_2")
        vline!([common, -common], label = "True", color = "black")
        vline!([mean(commonBeta12), mean(commonBeta22)], label = ["PPMx-shared", "PPMx"])
        vline!([meanBetakclust12, meanBetakclust22], label = "kmeans")
        xlims!(-3 * common, 3*common)
        display(current())
        
        # specific
        plot(
          histogram(Mix_beta1_c1_2, label = "Cluster 1, " * L"\beta_1"),
          histogram(Mix_beta1_c2_2, label = "Cluster 2, " * L"\beta_1"),
          histogram(Mix_beta2_c1_2, label = "Cluster 1, " * L"\beta_2"),
          histogram(Mix_beta2_c2_2, label = "Cluster 2, " * L"\beta_2"),
          plot_title = "Cluster specific estimates \n (2 clusters) \n Mixture"
          )
        display(current())
        plot(
          histogram(dpm_beta1_c1_2, label = "Cluster 1, " * L"\beta_1"),
          histogram(dpm_beta1_c2_2, label = "Cluster 2, " * L"\beta_1"),
          histogram(dpm_beta2_c1_2, label = "Cluster 1, " * L"\beta_2"),
          histogram(dpm_beta2_c2_2, label = "Cluster 2, " * L"\beta_2"),
          plot_title = "Cluster specific estimates \n (2 clusters) \n PPMx"
          )
        display(current())
      end
    catch err
      print(err)
      result = DataFrame(
        # mixture model (4 clust)
        rind_Mix_4 = missing,
        rindMixoos_4 = missing,
        midMix = missing,
        midMixoos = missing,
        bayesPmix = missing,
        bayesPmixoos = missing,
        meanBeta1 = missing, meanBeta2 = missing, 
        Mix_beta1_c1_4 = missing,
        Mix_beta2_c1_4 = missing,
        Mix_beta1_c2_4 = missing,
        Mix_beta2_c2_4 = missing,
        
        # PPMx model (4 clust)
        rind_DPM_4 = missing,
        rind_DPMoos_4 = missing,
        midDPM = missing,
        midDPMoos = missing,
        bayesPDPM = missing,
        bayesPDPMoos = missing,
        dpm_beta1_c1_4 = missing,
        dpm_beta2_c1_4 = missing,
        dpm_beta1_c2_4 = missing,
        dpm_beta2_c2_4 = missing,
        
        # rind_K_4
        rind_K_4 = missing,
        rind_K_4oos = missing,
        kmean_MSE_4 = missing, 
        kmean_MSE_4oos = missing, 
        meanBetakclust14 = missing,
        meanBetakclust2 = missing,
        
        # 4 clust compare
        #ks = ks,
        
        # mixture model (2 clust)
        rind_Mix_2 = missing,
        rind_Mix_2oos = missing,
        midMix2 = missing,
        midMix2oos = missing,
        Mix_beta1_c1_2 = missing,
        Mix_beta2_c1_2 = missing,
        Mix_beta1_c2_2 = missing,
        Mix_beta2_c2_2 = missing,
        
        # PPMx model (2 clust)
        rind_DPM_2 = missing,
        rind_DPM_2oos = missing,
        midDPM2 = missing,
        midDPM2oos = missing,
        meanBeta12 = missing, meanBeta22 = missing,
        dpm_beta1_c1_2 = missing,
        dpm_beta2_c1_2 = missing,
        dpm_beta1_c2_2 = missing,
        dpm_beta2_c2_2 = missing,
        
        # 2 clust compare
        rind_K_2 = missing,
        rind_K_2oos = missing,
        kmean_MSE_2 = missing, 
        kmean_MSE_2oos = missing, 
        
        # setup
        n=missing, fractions = missing, variance = missing, 
        interEffect = missing, common = missing
      )
    end
    
    return result
end
