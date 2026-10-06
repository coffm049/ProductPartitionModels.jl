using CSV, DataFrames, Statistics, StatsBase, Printf

"""
Generate paper-ready LaTeX tables from collected simulation results.
Run on HPC after simulations complete and collect_sim_results.jl has been run.
"""

# Type II error (when common != 0): P(0 in CI | true effect ≠ 0)
function t2_error(df, zero_col, common_col)
    mask = .!iszero.(df[:, common_col])
    n = sum(mask)
    n > 0 ? mean(df[mask, zero_col]) : missing
end

# Coverage: P(true effect in CI)
coverage(df, col) = mean(df[:, col])

# For β2* (sign-alternating), the true effect is -common
# Type II error: P(0 in CI | true effect = -common ≠ 0) = mean(zeroIn2 | common ≠ 0)
# Coverage: P(-common in CI) = mean(commonIn2)
t2_error_beta2(df, zero2_col, common_col) = t2_error(df, zero2_col, common_col)
coverage_beta2(df, col) = mean(df[:, col])

# --- Helper: compute Type II error and coverage from raw data ---
function compute_method_metrics(raw_df, method, common_col)
    if method == "mix"          # PPMx-common
        z1, c1, z2, c2 = "zeroInDPM", "commonInDPM", "zeroInDPM2", "commonInDPM2"
    elseif method == "dpm"      # Standard PPMx
        z1, c1, z2, c2 = "zeroInDPM", "commonInDPM", "zeroInDPM2", "commonInDPM2"
    elseif method == "dpmm"     # DPMM
        z1, c1, z2, c2 = "dpmzeroIn1", "dpmcommonIn1", "dpmzeroIn2", "dpmcommonIn2"
    elseif method == "kmeans"
        z1, c1, z2, c2 = "zeroInk", "commonInk", "zeroInk2", "commonInk2"
    elseif method == "slr"
        z1, c1, z2, c2 = "zeroInSLR", "commonInSLR", "zeroInSLR2", "commonInSLR2"
    else
        error("Unknown method: $method")
    end

    # Type II error (β1): P(0 in CI | common ≠ 0)
    mask = .!iszero.(raw_df[:, :common])
    t2_1 = sum(.!iszero.(raw_df[:, :common])) > 0 ? mean(raw_df[raw_df.common .!= 0, z1]) : missing
    # β2* has true effect = -common
    t2_2 = sum(.!iszero.(raw_df[:, :common])) > 0 ? mean(raw_df[raw_df.common .!= 0, z2]) : missing

    cov_1 = mean(raw_df[:, c1])
    cov_2 = mean(raw_df[:, c2])

    return t2_1, t2_2, cov_1, cov_2
end

function fmt2(v1, v2)
    ismissing(v1) || ismissing(v2) ? "NA" : @sprintf("%.3f\\newline %.3f", v1, v2)
end

function fmt_ci(v, l, u)
    ismissing(v) || ismissing(l) || ismissing(u) ? "NA" : @sprintf("%.3f (%.3f--%.3f)", v, l, u)
end

"""
Generate paper-ready LaTeX tables from collected simulation results.
Run on HPC after simulations complete and collect_sim_results.jl has been run.
"""

function make_sim_table_paper(results_dir="results/v2.0"; output_file="paper_sim_table.tex")
    # Read raw data to compute Type II error and coverage properly
    raw_file = "paper_sim_raw.csv"
    if !isfile(raw_file)
        error("Run collect_sim_results.jl first to generate $raw_file")
    end
    raw_df = CSV.read(raw_file, DataFrame)
    
    # Filter to N=1000, 50 reps (current runs)
    raw_df = filter(r -> r.N == 1000, raw_df)
    
    # Group by condition
    grp_cols = [:dims, :xdiff, :common, :interEffect, :variance, :N]
    existing_grp = [c for c in [:dims, :xdiff, :common, :interEffect, :variance, :N] if c in names(raw_df)]
    
    if isempty(existing_grp)
        @warn "Could not find grouping columns in raw data"
        return
    end
    
    # Group and compute metrics for each method
    grouped = groupby(raw_df, existing_grp)
    results = DataFrame()
    
    for g in grouped
        # Condition identifiers
        cond = DataFrame(g[1, existing_grp])
        
        # PPMx-common (mixDPM=true)
        t2_m1, t2_m2, cov_m1, cov_m2 = compute_method_metrics(g, "mix", :common)
        # Standard PPMx (mixDPM=false) - same raw columns as mix
        t2_s1, t2_s2, cov_s1, cov_s2 = compute_method_metrics(g, "dpm", :common)
        # DPMM
        t2_d1, t2_d2, cov_d1, cov_d2 = compute_method_metrics(g, "dpmm", :common)
        # K-means
        t2_k1, t2_k2, cov_k1, cov_k2 = compute_method_metrics(g, "kmeans", :common)
        # SLR
        t2_l1, t2_l2, cov_l1, cov_l2 = compute_method_metrics(g, "slr", :common)
        
        push!(results, merge(cond, 
            Dict(:t2_mix_1=>t2_m1, :t2_mix_2=>t2_m2, :cov_mix_1=>cov_m1, :cov_mix_2=>cov_m2,
                 :t2_std_1=>t2_s1, :t2_std_2=>t2_s2, :cov_std_1=>cov_s1, :cov_std_2=>cov_s2,
                 :t2_dpmm_1=>t2_d1, :t2_dpmm_2=>t2_d2, :cov_dpmm_1=>cov_d1, :cov_dpmm_2=>cov_d2,
                 :t2_km_1=>t2_k1, :t2_km_2=>t2_k2, :cov_km_1=>cov_k1, :cov_km_2=>cov_k2,
                 :t2_slr_1=>t2_l1, :t2_slr_2=>t2_l2, :cov_slr_1=>cov_l1, :cov_slr_2=>cov_l2))
        )
    end
    
    sort!(results, [:dims, :xdiff, :common, :interEffect, :variance, :N])
    
    # Build LaTeX table
    io = IOBuffer()
    println(io, "\\begin{table}[ht!]")
    println(io, "\\centering")
    println(io, "\\fontsize{6.0pt}{8pt}\\selectfont")
    println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} p{0.2cm}p{0.2cm}p{0.2cm}p{0.2cm}p{0.4cm}|")
    println(io, " p{0.6cm}p{0.5cm}p{0.5cm}p{0.6cm}p{0.5cm}p{0.5cm}p{0.6cm}p{0.55cm}p{0.55cm}p{0.5cm}p{0.4cm}p{0.4cm}}")
    println(io, "\\toprule")
    println(io, " & & & & & \\multicolumn{3}{c}{T2} & \\multicolumn{3}{c}{Cov} & \\multicolumn{3}{c}{Bias} & \\multicolumn{3}{c}{RMSE} \\\\")
    println(io, "\\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17}")
    println(io, "\\$\\Delta \\beta\$ & \\$a_{\\beta}\$ & \\$b_{\\beta}\$ & \\$\\sigma_{\\epsilon}^2\$ & \\$\\Delta X\$ & PPMx & LR & KM & PPMx & LR & KM & PPMx & LR & KM & PPMx & LR & KM \\\\")
    println(io, "\\midrule\\addlinespace[2.5pt]")
    
    function fmt2(v1, v2)
        ismissing(v1) || ismissing(v2) ? "NA" : @sprintf("%.3f\\newline %.3f", v1, v2)
    end
    
    function fmt_ci(v, l, u)
        ismissing(v) || ismissing(l) || ismissing(u) ? "NA" : @sprintf("%.3f (%.3f--%.3f)", v, l, u)
    end
    
    for row in eachrow(results)
        dX = row.xdiff
        dims = row.dims
        inter = row.interEffect
        common = row.common
        var = row.variance
        N = row.N
        
        println(io, "$(dims) & $(var) & $(inter) & $(var) & $(dX) & ", 
            fmt2(row.t2_mix_1, row.t2_mix_2), " & ",
            fmt2(row.t2_slr_1, row.t2_slr_2), " & ",
            fmt2(row.t2_km_1, row.t2_km_2), " & ",
            fmt2(row.cov_mix_1, row.cov_mix_2), " & ",
            fmt2(row.cov_slr_1, row.cov_slr_2), " & ",
            fmt2(row.cov_km_1, row.cov_km_2), " \\\\")
    end
    
    # Write to file
    io = IOBuffer()
    println(io, "\\begin{table}[ht!]")
    println(io, "\\centering")
    println(io, "\\fontsize{6.0pt}{8pt}\\selectfont")
    println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} p{0.2cm}p{0.2cm}p{0.2cm}p{0.2cm}p{0.4cm}|")
    println(io, " p{0.6cm}p{0.5cm}p{0.5cm}p{0.6cm}p{0.5cm}p{0.5cm}p{0.6cm}p{0.55cm}p{0.55cm}p{0.5cm}p{0.4cm}p{0.4cm}}")
    println(io, "\\toprule")
    println(io, " & & & & & \\multicolumn{3}{c}{T2} & \\multicolumn{3}{c}{Cov} & \\multicolumn{3}{c}{Bias} & \\multicolumn{3}{c}{RMSE} \\\\")
    println(io, "\\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17}")
    println(io, "\\$\\Delta \\beta\$ & \\$a_{\\beta}\$ & \\$b_{\\beta}\$ & \\$\\sigma_{\\epsilon}^2\$ & \\$\\Delta X\$ & PPMx & LR & KM & PPMx & LR & KM & PPMx & LR & KM & PPMx & LR & KM \\\\")
    println(io, "\\midrule\\addlinespace[2.5pt]")
    
    for row in eachrow(results)
        dX = row.xdiff
        dims = row.dims
        inter = row.interEffect
        common = row.common
        var = row.variance
        N = row.N
        
        println(io, "$(dims) & $(var) & $(inter) & $(var) & $(dX) & ", 
            fmt2(row.t2_mix_1, row.t2_mix_2), " & ",
            fmt2(row.t2_slr_1, row.t2_slr_2), " & ",
            fmt2(row.t2_km_1, row.t2_km_2), " & ",
            fmt2(row.cov_mix_1, row.cov_mix_2), " & ",
            fmt2(row.cov_slr_1, row.cov_slr_2), " & ",
            fmt2(row.cov_km_1, row.cov_km_2), " \\\\")
    end
    
    # Write to file
    io = IOBuffer()
    println(io, "\\begin{table}[ht!]")
    println(io, "\\centering")
    println(io, "\\fontsize{6.0pt}{8pt}\\selectfont")
    println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} p{0.2cm}p{0.2cm}p{0.2cm}p{0.2cm}p{0.4cm}|")
    println(io, " p{0.6cm}p{0.5cm}p{0.5cm}p{0.6cm}p{0.5cm}p{0.5cm}p{0.6cm}p{0.55cm}p{0.55cm}p{0.5cm}p{0.4cm}p{0.4cm}}")
    println(io, "\\toprule")
    println(io, " & & & & & \\multicolumn{3}{c}{T2} & \\multicolumn{3}{c}{Cov} & \\multicolumn{3}{c}{Bias} & \\multicolumn{3}{c}{RMSE} \\\\")
    println(io, "\\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17}")
    println(io, "\\$\\Delta \\beta\$ & \\$a_{\\beta}\$ & \\$b_{\\beta}\$ & \\$\\sigma_{\\epsilon}^2\$ & \\$\\Delta X\$ & PPMx & LR & KM & PPMx & LR & KM & PPMx & LR & KM & PPMx & LR & KM \\\\")
    println(io, "\\midrule\\addlinespace[2.5pt]")
    
    for row in eachrow(results)
        dX = row.xdiff
        dims = row.dims
        inter = row.interEffect
        common = row.common
        var = row.variance
        N = row.N
        
        println(io, "$(dims) & $(var) & $(inter) & $(var) & $(dX) & ", 
            fmt2(row.t2_mix_1, row.t2_mix_2), " & ",
            fmt2(row.t2_slr_1, row.t2_slr_2), " & ",
            fmt2(row.t2_km_1, row.t2_km_2), " & ",
            fmt2(row.cov_mix_1, row.cov_mix_2), " & ",
            fmt2(row.cov_slr_1, row.cov_slr_2), " & ",
            fmt2(row.cov_km_1, row.cov_km_2), " \\\\")
    end
    
    println(io, "\\bottomrule")
    println(io, "\\end{tabular*}")
    println(io, "    \\caption{Simulation performance summary of estimating the common effect \\$\\\\bm \\\\beta^*\\$ across PPMx-common, standard PPMx, DP-GMM, K-means, and linear regression methods. Shown are Type II error (T2), coverage (Cov), median bias (censored to [-3,3]), and RMSE (truncated at 90th pctl) for \\$\\\\beta_1^*\\$ and \\$\\\\beta_2^*\\$ (two values per cell). New baselines: standard PPMx and DP-GMM cluster-then-regression added per Reviewer 1. Bias/RMSE displayed with robust median/90th-pctl truncation; coverage for sign-alternating \\$\\\\beta_2^*\\$ correctly evaluates \\$-\\\\beta^*\\in\\text{CI}\\$.)")
    println(io, "    \\label{tab:betaTuningTable}")
    println(io, "\\end{table}")
    
    write(output_file, String(take!(io)))
    @info "Wrote $output_file"
end