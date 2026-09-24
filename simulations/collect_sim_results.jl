using CSV, DataFrames, Statistics, StatsBase, DataFramesMeta

"""
Collect all simulation result CSVs from results/v2.0/ and produce
paper-ready summary tables.
"""
function collect_sim_results(results_dir="results/v2.0"; output_prefix="paper_sim_")
    files = filter(f -> endswith(f, ".csv"), readdir(results_dir, join=true))
    isempty(files) && error("No CSV files found in $results_dir")

    dfs = DataFrame[]
    for f in files
        try
            df = CSV.read(f, DataFrame; silencewarnings=true)
            push!(dfs, df)
        catch e
            @warn "Failed to read $f" exception=e
        end
    end
    isempty(dfs) && error("No valid CSV files read")
    alldf = vcat(dfs...; cols=:union)

    # 1. Full raw results (all replicates)
    CSV.write("$(output_prefix)raw.csv", alldf)
    @info "Wrote $(output_prefix)raw.csv ($(nrow(alldf)) rows)"

    # 2. Summary by condition (group by simulation parameters)
    grp_cols = [:N, :nc, :variance, :interEffect, :common, :xdiff, :dims, :prec, :alph, :bet, :massParams]
    existing_grp = [c for c in grp_cols if c in names(alldf)]
    if !isempty(existing_grp)
        summary = @chain alldf begin
            groupby(existing_grp)
            combine(
                nrow => :n_reps,
                # PPMx-common (mixDPM=true)
                :midMix => median => :rmse_mix_med,
                :midMixoos => median => :rmse_mixoos_med,
                :adjrind_Mix => median => :ari_mix_med,
                :adjrindMixoos => median => :ari_mixoos_med,
                :lpsMixoos => median => :lps_mixoos_med,
                :meanBeta1 => median => :beta1_mix_med,
                :meanBeta2 => median => :beta2_mix_med,
                :commonCovAll => mean => :coverage_mix,
                # Standard PPMx (mixDPM=false)
                :midDPM => median => :rmse_dpm_med,
                :midDPMoos => median => :rmse_dpmoos_med,
                :adjrind_DPM => median => :ari_dpm_med,
                :adjrind_DPMoos => median => :ari_dpmoos_med,
                :lpsDPMoos => median => :lps_dpmoos_med,
                # K-means
                :kmean_MSE => median => :rmse_km_med,
                :kmean_MSEoos => median => :rmse_kmoos_med,
                :adjrind_K => median => :ari_km_med,
                :adjrind_Koos => median => :ari_kmoos_med,
                # SLR
                :slrRMSE => median => :rmse_slr_med,
                :slrRMSEoos => median => :rmse_slroos_med,
                # DP-GMM
                :dpmRMSE => median => :rmse_dpmm_med,
                :dpmRMSEoos => median => :rmse_dpmmoos_med,
                :dpmARI => median => :ari_dpmm_med,
                :dpmARIoos => median => :ari_dpmmoos_med,
                # SALSO ARI
                :salsoBinderARI_Mix => median => :salso_binder_mix_med,
                :salsoVIARI_Mix => median => :salso_vi_mix_med,
                :salsoBinderARI_Mixoos => median => :salso_binder_mixoos_med,
                :salsoVIARI_Mixoos => median => :salso_vi_mixoos_med,
                # Cluster counts
                :ncMix => median => :k_mix_med,
                :ncDPM => median => :k_dpm_med,
                :dpmnclusts => median => :k_dpmm_med,
            )
        end
        CSV.write("$(output_prefix)summary_by_condition.csv", summary)
        @info "Wrote $(output_prefix)summary_by_condition.csv ($(nrow(summary)) conditions)"
    end

    # 3. Key comparison table for paper (simplified, xdiff × dims × common)
    key_cols = [:xdiff, :dims, :common, :interEffect, :N]
    existing_key = [c for c in key_cols if c in names(summary)]
    if !isempty(existing_key)
        paper_tbl = select(summary, existing_key...,
            :rmse_mix_med, :rmse_dpm_med, :rmse_km_med, :rmse_slr_med, :rmse_dpmm_med,
            :rmse_mixoos_med, :rmse_dpmoos_med, :rmse_kmoos_med, :rmse_slroos_med, :rmse_dpmmoos_med,
            :ari_mix_med, :ari_dpm_med, :ari_km_med, :ari_dpmm_med,
            :ari_mixoos_med, :ari_dpmoos_med, :ari_kmoos_med, :ari_dpmmoos_med,
            :lps_mixoos_med, :lps_dpmoos_med,
            :beta1_mix_med, :beta2_mix_med,
            :coverage_mix,
            :k_mix_med, :k_dpm_med, :k_dpmm_med)
        CSV.write("$(output_prefix)paper_comparison.csv", paper_tbl)
        @info "Wrote $(output_prefix)paper_comparison.csv"
    end

    # 4. SALSO-specific table
    salso_cols = [c for c in names(summary) if startswith(c, "salso")]
    if !isempty(salso_cols)
        salso_tbl = select(summary, existing_key..., salso_cols...)
        CSV.write("$(output_prefix)salso.csv", salso_tbl)
        @info "Wrote $(output_prefix)salso.csv"
    end

    return alldf, summary
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    collect_sim_results()
end