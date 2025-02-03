# update_lik_params.jl

struct TargetArgs_EsliceBetas{T <: Real, TT <: LikParams_PPMxReg, TTT <: Similarity_PPMxStats, TTTT <: Similarity_PPMx} <: TargetArgs
    y_k::Vector{T}
    X_k::Union{Matrix{T}, Matrix{Union{T, Missing}}, Matrix{Missing}}
    ObsXIndx_k::Vector{ObsXIndx}
    lik_params_k::TT
    Xstats_k::Vector{TTT}
    similarity::TTTT
end

struct TargetArgs_sliceSig_Reg{T <: Real} <: TargetArgs
    y_k::Vector{T}
    means::Vector{T}
    vars::Vector{T}
    sig_old::T

    beta::Vector{T}
    tau0::T
    tau::T
    phi::Vector{T}
    psi::Vector{T}
end

struct TargetArgs_sliceSig_noReg{T <: Real} <: TargetArgs
    y_k::Vector{T}
    lik_params_k::LikParams_PPMxMean
end

function llik_k_forEsliceBeta(beta_cand::Vector{T}, args::TargetArgs_EsliceBetas) where T <: Real

    lik_params_cand = deepcopy(args.lik_params_k)
    lik_params_cand.beta = beta_cand

    return llik_k(args.y_k, args.X_k, args.ObsXIndx_k, lik_params_cand,
                  args.Xstats_k, args.similarity)
end

function llik_k_forSliceSig_Reg(sig_cand::Real, args::TargetArgs_sliceSig_Reg)
    
    llik_k_tmp = llik_k(args.y_k, args.means, args.vars, args.sig_old, sig_cand)
    llik_kk = llik_k_tmp[:llik]

    prior_var_beta = args.tau^2 .* sig_cand^2 .*
        args.tau0^2 .* args.phi.^2 .* args.psi

    lpri_beta = 0.0
    for ell in 1:length(args.beta)
        vv = prior_var_beta[ell]
        lpri_beta += -0.5*log(2π) - 0.5*log(vv) - 0.5*(args.beta[ell])^2/vv # prior mean is 0
    end

    llik_kk += lpri_beta

    return Dict(:llik => llik_kk, :means => llik_k_tmp[:means], :vars => llik_k_tmp[:vars])
end

function llik_k_forSliceSig_noReg(sig_cand::Real, args::TargetArgs_sliceSig_noReg)

    lik_params_k_cand = deepcopy(args.lik_params_k)
    lik_params_k_cand.sig = sig_cand
    
    return llik_k(args.y_k, lik_params_k_cand)
end

function update_lik_params!(model::Model_PPMx,
    update::Vector{Symbol}=[:mu, :sig, :beta],
    sliceiter::Int=5000)

    K = maximum(model.state.C)


    # [x] Update this as the average of betas (N-Jeffries)
    # concatenate all of the betas across lik_params[j] into a matrix
    Betas = [model.state.lik_params[k].beta for k in 1:K]
    # convert the vector of vectos to a matrix (p x K)
    Betas = hcat(Betas...)'
    n, p = size(Betas)

    mu0 = repeat([0.0], p)  # Prior mean
    kappa0 = repeat([0.01], p)  # Prior precision
    alpha0 = repeat([0.01], p)  # Prior shape for σ^2
    beta0 = repeat([15.0], p)  # Prior scale for σ^2
    
    # Run the sampler
    mu_sample, sigma2_sample = independent_sampler(Betas, mu0, kappa0, alpha0, beta0,1)
    model.state.prior_mean_beta = mu_sample[:,1]
    prior_mean_beta = model.state.prior_mean_beta
    prior_var_beta = sigma2_sample[:,1]
    #print(model.state.prior_mean_beta)
    #prior_mean_beta = zeros(model.p)

    for k in 1:K ## can parallelize; would need to pass rng through updates (including slice functions and hyper updates)

        indx_k = findall(model.state.C.==k)

        if typeof(model.state.lik_params[k]) <: LikParams_PPMxReg
            if ( :beta in update ) & (typeof(model.y) <: Vector)
                ## update betas, produces vectors of obs-specific means and variances
                prior_var_beta = model.state.lik_params[k].beta_hypers.tau^2 .* model.state.lik_params[k].sig^2 .*
                    model.state.baseline.tau0^2 .*
                    model.state.lik_params[k].beta_hypers.phi.^2 .*
                    model.state.lik_params[k].beta_hypers.psi

                model.state.lik_params[k].beta, beta_upd_stats, iters_eslice = ellipSlice(
                    model.state.lik_params[k].beta,
                    prior_mean_beta, prior_var_beta,
                    llik_k_forEsliceBeta,
                    TargetArgs_EsliceBetas(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k],
                        model.state.lik_params[k], model.state.Xstats[k], model.state.similarity),
                        sliceiter
                    )

                ## update beta hypers (could customize a function here to accommodate different shrinkage priors)
                model.state.lik_params[k].beta_hypers.psi = update_ψ(model.state.lik_params[k].beta_hypers.phi,
                    model.state.lik_params[k].beta ./ model.state.baseline.tau0 ./ model.state.lik_params[k].sig,
                    model.state.lik_params[k].beta_hypers.tau
                )

                model.state.lik_params[k].beta_hypers.tau = update_τ(model.state.lik_params[k].beta_hypers.phi,
                    model.state.lik_params[k].beta ./ model.state.baseline.tau0 ./ model.state.lik_params[k].sig,
                    1.0/model.p
                )

                model.state.lik_params[k].beta_hypers.phi = update_ϕ(model.state.lik_params[k].beta ./ model.state.baseline.tau0 ./ model.state.lik_params[k].sig,
                    1.0/model.p
                )

            #state.prior_mean_beta =  ..... FILL IN WITH N-N updater
            elseif ( :beta in update ) & (typeof(model.y) <: Matrix)
                # [ ] Might need to have update stats and iters_eslice
                model.state.lik_params[k].beta = sampleMultiNorm(model.X[indx_k, :], model.y[indx_k, :])

            else
                beta_upd_stats = llik_k(model.y[indx_k], model.X[indx_k,:], model.obsXIndx[indx_k],
                    model.state.lik_params[k], model.state.Xstats[k], model.state.similarity)
            end

            ## update sig, which preserves means to be modified in the update for means
            if (:sig in update)
                model.state.lik_params[k].sig, sig_upd_stats, iters_sslice = shrinkSlice(model.state.lik_params[k].sig,
                    model.state.baseline.sig_lower, model.state.baseline.sig_upper,
                    llik_k_forSliceSig_Reg,
                    TargetArgs_sliceSig_Reg(model.y[indx_k], beta_upd_stats[:means], beta_upd_stats[:vars], model.state.lik_params[k].sig,
                                        model.state.lik_params[k].beta, model.state.baseline.tau0, model.state.lik_params[k].beta_hypers.tau,
                                        model.state.lik_params[k].beta_hypers.phi, model.state.lik_params[k].beta_hypers.psi),
                    sliceiter
                    ) # sig_old doesn't need to be updated during intermediate proposals of slice sampler--vars isn't updated either,
                    # so each step evaluates against the same (original) set of target args. This allows us to use the generic slice sampler code.
            else
                sig_upd_stats = deepcopy(beta_upd_stats) # means (indx 2) and vars (indx 3) that get used haven't changed
            end
        
        elseif typeof(model.state.lik_params[k]) <: LikParams_PPMxMean
            if (:sig in update)
                model.state.lik_params[k].sig, sig_upd_stats, iters_sslice = shrinkSlice(model.state.lik_params[k].sig,
                    model.state.baseline.sig_lower, model.state.baseline.sig_upper,
                    llik_k_forSliceSig_noReg,
                    TargetArgs_sliceSig_noReg(model.y[indx_k], model.state.lik_params[k]),
                    sliceiter
                    ) # sig_old doesn't need to be updated during intermediate proposals of slice sampler--vars isn't updated either,
                    # so each step evaluates against the same (original) set of target args. This allows us to use the generic slice sampler code.
                    # I don't think this is an issue with PPMxMean either.
            end
            
            ## always caluclate this because sig_upd_stats doesn't output running means, vars with LikParams_PPMxMean
            sig_upd_stats = Dict(:means => fill(model.state.lik_params[k].mu, length(indx_k)), 
                                 :vars => fill(model.state.lik_params[k].sig^2, length(indx_k)))
        end

        ## update mu (generic)
        model.state.lik_params[k].mu =0.0 
        # if (:mu in update)
        #     yy = model.y[indx_k] - sig_upd_stats[:means] .+ model.state.lik_params[k].mu
        #     one_div_var = 1.0 ./ sig_upd_stats[:vars]
        #     yy_div_var = yy .* one_div_var
        #     v1 = 1.0 / (1.0/model.state.baseline.sig0^2 + sum(one_div_var))
        #     m1 = v1 * (model.state.baseline.mu0 / model.state.baseline.sig0^2 + sum(yy_div_var))
        #     model.state.lik_params[k].mu = randn()*sqrt(v1) + m1
        #     model.state.lik_params[k].mu =0.0 
        # end

    end

    return nothing
end
