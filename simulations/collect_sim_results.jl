using CSV, DataFrames, Statistics, StatsBase, DataFramesMeta

"""
Collect all simulation result CSVs from results/v3-dl-centered/ and produce
paper-ready summary tables. Fails loudly if results are missing/incomplete.
"""
function collect_sim_results(results_dir="results/v3-dl-centered"; output_prefix="paper_sim_")
    files = filter(f -> endswith(f, ".csv"), readdir(results_dir, join=true))
    isempty(files) && error("No CSV files found in $results_dir")

    grp_cols = [:N, :nc, :variance, :interEffect, :common, :xdiff, :dims, :prec, :alph, :bet, :massParams]
    required_metric_cols = [
        :midMix, :midMixoos, :adjrind_Mix, :adjrindMixoos, :lpsMixoos,
        :meanBeta1, :meanBeta2, :commonCovAll,
        :midDPM, :midDPMoos, :adjrind_DPM, :adjrind_DPMoos, :lpsDPMoos,
        :kmean_MSE, :kmean_MSEoos, :adjrind_K, :adjrind_Koos,
        :slrRMSE, :slrRMSEoos,
        :dpmRMSE, :dpmRMSEoos, :dpmARI, :dpmARIoos,
        :dpmzeroIn1, :dpmcommonIn1, :dpmzeroIn2, :dpmcommonIn2,
        :salsoBinderARI_Mix, :salsoVIARI_Mix, :salsoBinderARI_Mixoos, :salsoVIARI_Mixoos,
        :ncMix, :ncDPM, :dpmnclusts,
    ]

    dfs = DataFrame[]
    for f in files
        df = CSV.read(f, DataFrame)
        if !hasproperty(df, :N)
            error("Incomplete results in $f: missing required column :N")
        end
        bad_n = unique(df.N[df.N .!= 1000])
        if !isempty(bad_n)
            error("Incomplete/mismatched results in $f: expected N=1000 for every row; found N=$(join(bad_n, ", "))")
        end
        push!(dfs, df)
    end
    isempty(dfs) && error("No valid simulation CSV files read")
    alldf = vcat(dfs...; cols=:union)

    missing_grp = setdiff(grp_cols, names(alldf))
    if !isempty(missing_grp)
        error("Incomplete simulation schema: missing required columns $(join(missing_grp, ", ")) in $(results_dir)")
    end

    # Full raw results (all replicates)
    summary = @chain alldf begin
        groupby(grp_cols)
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
            # DPMM coverage (for cluster-specific slopes)
            :dpmzeroIn1 => mean => :cov_dpmm_zero1,
            :dpmcommonIn1 => mean => :cov_dpmm_common1,
            :dpmzeroIn2 => mean => :cov_dpmm_zero2,
            :dpmcommonIn2 => mean => :cov_dpmm_common2,
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

    missing_summary = setdiff(required_metric_cols, names(summary))
    isempty(missing_summary) || error("Incomplete simulation summary: missing output columns $(join(missing_summary, ", "))")

    key_cols = [:xdiff, :dims, :common, :interEffect, :N]
    missing_key = setdiff(key_cols, names(summary))
    isempty(missing_key) || error("Incomplete simulation summary: missing key columns $(join(missing_key, ", "))")

    paper_cols = [key_cols...,
        :rmse_mix_med, :rmse_dpm_med, :rmse_km_med, :rmse_slr_med, :rmse_dpmm_med,
        :rmse_mixoos_med, :rmse_dpmoos_med, :rmse_kmoos_med, :rmse_slroos_med, :rmse_dpmmoos_med,
        :ari_mix_med, :ari_dpm_med, :ari_km_med, :ari_dpmm_med,
        :ari_mixoos_med, :ari_dpmoos_med, :ari_kmoos_med, :ari_dpmmoos_med,
        :lps_mixoos_med, :lps_dpmoos_med,
        :beta1_mix_med, :beta2_mix_med,
        :coverage_mix,
        :cov_dpmm_zero1, :cov_dpmm_common1, :cov_dpmm_zero2, :cov_dpmm_common2,
        :k_mix_med, :k_dpm_med, :k_dpmm_med]
    missing_paper = setdiff(paper_cols, names(summary))
    isempty(missing_paper) || error("Incomplete simulation comparison: missing columns $(join(missing_paper, ", "))")
    paper_tbl = select(summary, paper_cols...)

    missing_salso = setdiff([:salsoBinderARI_Mix, :salsoVIARI_Mix, :salsoBinderARI_Mixoos, :salsoVIARI_Mixoos], names(summary))
    isempty(missing_salso) || error("Incomplete simulation summary: missing SALSO columns $(join(missing_salso, ", "))")
    salso_tbl = select(summary, key_cols..., :salsoBinderARI_Mix, :salsoVIARI_Mix, :salsoBinderARI_Mixoos, :salsoVIARI_Mixoos)

    CSV.write("$(output_prefix)raw.csv", alldf)
    @info "Wrote $(output_prefix)raw.csv ($(nrow(alldf)) rows)"
    CSV.write("$(output_prefix)summary_by_condition.csv", summary)
    @info "Wrote $(output_prefix)summary_by_condition.csv ($(nrow(summary)) conditions)"
    CSV.write("$(output_prefix)paper_comparison.csv", paper_tbl)
    @info "Wrote $(output_prefix)paper_comparison.csv"
    CSV.write("$(output_prefix)salso.csv", salso_tbl)
    @info "Wrote $(output_prefix)salso.csv"

    return alldf, summary
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    collect_sim_results()
end
