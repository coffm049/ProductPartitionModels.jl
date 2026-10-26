#!/usr/bin/env python3
"""
Generate paper table (tab:v2grid) from dim=5 simulation results.
Matches the format in Response_to_Reviewers.tex Table tab:v2grid.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("C:/Users/coffm049/research/04-projects/ppmx/ProductPartitionModels.jl/simulations/results/v2.0")

# Parse filename to extract parameters
def parse_filename(fname):
    """Extract simulation parameters from filename."""
    # Pattern: N1000_c8_inter1.0_common1.0_xd0.4_v0.05_dim5_prec0.1alph1.0bet1.0_mass1.01.0_dpm1.0_500.csv
    # or: Imbal_N1000_c8_inter1.0_common1.0_xd0.4_v0.05_dim5_prec0.1alph1.0bet1.0_mass2.02.0_dpm1.0_500.csv
    is_imbalanced = fname.startswith("Imbal_")
    
    # Remove prefix
    name = fname.replace("Imbal_", "").replace(".csv", "")
    
    # Extract parameters
    params = {}
    params['imbalanced'] = is_imbalanced
    
    # N
    m = re.search(r'N(\d+)', name)
    params['N'] = int(m.group(1)) if m else 1000
    
    # nc (number of clusters)
    m = re.search(r'_c(\d+)', name)
    params['nc'] = int(m.group(1)) if m else 8
    
    # interEffect
    m = re.search(r'_inter([\d.]+)', name)
    params['interEffect'] = float(m.group(1)) if m else 1.0
    
    # common
    m = re.search(r'_common([\d.]+)', name)
    params['common'] = float(m.group(1)) if m else 1.0
    
    # xdiff
    m = re.search(r'_xd([\d.]+)', name)
    params['xdiff'] = float(m.group(1)) if m else 0.0
    
    # variance
    m = re.search(r'_v([\d.]+)', name)
    params['variance'] = float(m.group(1)) if m else 0.05
    
    # dims
    m = re.search(r'_dim(\d+)', name)
    params['dims'] = int(m.group(1)) if m else 5
    
    # prec
    m = re.search(r'_prec([\d.]+)', name)
    params['prec'] = float(m.group(1)) if m else 0.1
    
    # alph
    m = re.search(r'alph([\d.]+)', name)
    params['alph'] = float(m.group(1)) if m else 1.0
    
    # bet
    m = re.search(r'bet([\d.]+)', name)
    params['bet'] = float(m.group(1)) if m else 1.0
    
    # massParams - format is massX.XX.YY where X is massa and YY is massb, each with one decimal place
    # e.g., mass1.01.0 = massa=1.0, massb=1.0; mass2.02.0 = massa=2.0, massb=2.0
    m = re.search(r'mass(\d+\.\d)(\d+\.\d)', name)
    if m:
        params['massa'] = float(m.group(1))
        params['massb'] = float(m.group(2))
    else:
        params['massa'] = 1.0
        params['massb'] = 1.0
    
    return params

# Read all CSV files
all_data = []
for f in results_dir.glob("*.csv"):
    params = parse_filename(f.name)
    try:
        df = pd.read_csv(f)
        # Add parameter columns
        for k, v in params.items():
            df[k] = v
        all_data.append(df)
        print(f"Read {f.name}: {len(df)} rows")
    except Exception as e:
        print(f"Error reading {f.name}: {e}")

if not all_data:
    print("No data files found!")
    exit(1)

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal rows: {len(full_df)}")
print(f"Conditions: {full_df.groupby(['imbalanced', 'common', 'xdiff', 'massa', 'massb']).size()}")

# Filter for massParams=1.01.0 (massa=1.0, massb=1.0) for main analysis
# For imbalanced xdiff=0.4, 0.6 we need to use mass2.02.0 since mass1.01.0 doesn't exist
main_df = full_df[
    ((full_df['massa'] == 1.0) & (full_df['massb'] == 1.0)) |
    ((full_df['imbalanced'] == True) & (full_df['xdiff'].isin([0.4, 0.6])) & (full_df['massa'] == 2.0) & (full_df['massb'] == 2.0))
].copy()

print(f"\nMain analysis rows: {len(main_df)}")
print(f"Conditions in main analysis:")
print(main_df.groupby(['imbalanced', 'common', 'xdiff', 'massa', 'massb']).size())

# Compute summary statistics by condition
group_cols = ['imbalanced', 'common', 'xdiff']
summary = main_df.groupby(group_cols).agg(
    n_reps=('N', 'count'),
    # PPMx-common (mixDPM=true)
    ari_mix_med=('adjrind_Mix', 'median'),
    ari_mixoos_med=('adjrindMixoos', 'median'),
    lps_mixoos_med=('lpsMixoos', 'median'),
    # Standard PPMx (mixDPM=false)
    ari_dpm_med=('adjrind_DPM', 'median'),
    ari_dpmoos_med=('adjrind_DPMoos', 'median'),
    lps_dpmoos_med=('lpsDPMoos', 'median'),
    # K-means
    ari_km_med=('adjrind_K', 'median'),
    ari_kmoos_med=('adjrind_Koos', 'median'),
    # DP-GMM
    ari_dpmm_med=('dpmARI', 'median'),
    ari_dpmmoos_med=('dpmARIoos', 'median'),
).reset_index()

print("\nSummary statistics:")
print(summary.to_string())

# Create LaTeX table matching tab:v2grid format
# The paper table has:
# Design | ΔX | PPMx-common | Standard PPMx | K-means | DPMM | PPMx-common | Standard PPMx
# Balanced | 0.0 | ... | ... | ... | ... | ... | ...
# Balanced | 0.2 | ... | ... | ... | ... | ... | ...
# ...

# Paper table uses in-sample ARI for clustering and OOS LPS for prediction
# For imbalanced, Standard PPMx duplicates PPMx-common (per paper note)

# Define the order of rows as in paper
balanced_xdiffs = [0.0, 0.2, 0.4, 0.6, 1.0]
imbalanced_xdiffs = [0.0, 0.4, 0.6, 0.8, 1.0]

latex_lines = []
latex_lines.append("\\begin{table}[!htp]")
latex_lines.append("\\centering")
latex_lines.append("\\fontsize{8.0pt}{10pt}\\selectfont")
latex_lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrrr}")
latex_lines.append("\\toprule")
latex_lines.append(" & & \\multicolumn{4}{c}{Clustering (ARI)} & \\multicolumn{2}{c}{Prediction (LPS)} \\\\")
latex_lines.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-8}")
latex_lines.append("Design & $\\Delta X$ & PPMx-common & Standard PPMx & K-means & DPMM & PPMx-common & Standard PPMx \\\\")
latex_lines.append("\\midrule\\addlinespace[2.5pt]")

for design_name, xdiffs, imbalanced_flag in [
    ("Balanced", balanced_xdiffs, False),
    ("Imbalanced", imbalanced_xdiffs, True)
]:
    for xd in xdiffs:
        row = summary[(summary['imbalanced'] == imbalanced_flag) & 
                      (summary['xdiff'] == xd) & 
                      (summary['common'] == 1.0)]
        
        if len(row) == 0:
            # Try common=0.0 for balanced xdiff=0.0
            if design_name == "Balanced" and xd == 0.0:
                row = summary[(summary['imbalanced'] == imbalanced_flag) & 
                              (summary['xdiff'] == xd) & 
                              (summary['common'] == 0.0)]
        
        if len(row) == 0:
            print(f"WARNING: No data for {design_name} xdiff={xd}")
            continue
        
        r = row.iloc[0]
        
        # Clustering ARI (in-sample)
        ppmx_common_ari = r['ari_mix_med']
        ppmx_ari = r['ari_dpm_med']
        kmeans_ari = r['ari_km_med']
        dpmm_ari = r['ari_dpmm_med']
        
        # Prediction LPS (OOS)
        ppmx_common_lps = r['lps_mixoos_med']
        ppmx_lps = r['lps_dpmoos_med']
        
        # For imbalanced, Standard PPMx duplicates PPMx-common per paper
        if design_name == "Imbalanced":
            ppmx_ari = ppmx_common_ari
            ppmx_lps = ppmx_common_lps
        
        # Format values
        def fmt_ari(v):
            if pd.isna(v):
                return "NA"
            return f"{v:.3f}"
        
        def fmt_lps(v):
            if pd.isna(v):
                return "NA"
            return f"{v:.3f}"
        
        latex_lines.append(f"{design_name} & {xd:.1f} & {fmt_ari(ppmx_common_ari)} & {fmt_ari(ppmx_ari)} & {fmt_ari(kmeans_ari)} & {fmt_ari(dpmm_ari)} & {fmt_lps(ppmx_common_lps)} & {fmt_lps(ppmx_lps)} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular*}")
latex_lines.append("\\caption{Second simulation grid, 80 replicates per cell. Clustering columns report the adjusted Rand index for PPMx-common, Standard PPMx, K-means, and DPMM. Prediction columns report the test log predictive score (LPS), available only for the two Bayesian models because the point-estimator baselines have no posterior predictive distribution. LPS is higher (less negative) is better. For the imbalanced design only the common-effect model was fitted, so the Standard PPMx column duplicates PPMx-common. At $\\Delta X=0$ the covariates carry no information about the generating partition. Imbalanced has fractions fractions $0.552, 0.248, 0.111, 0.050, 0.022, 0.010, 0.005, 0.002$.")
latex_lines.append("\\label{tab:v2grid}")
latex_lines.append("\\end{table}")

# Output
output_file = results_dir.parent / "paper_table_v2grid.tex"
with open(output_file, 'w') as f:
    f.write('\n'.join(latex_lines))

print(f"\nLaTeX table written to {output_file}")
print("\n--- LaTeX Table ---")
print('\n'.join(latex_lines))

# Also save summary CSV for verification
summary.to_csv(results_dir.parent / "paper_table_v2grid_summary.csv", index=False)
print(f"\nSummary CSV written to {results_dir.parent / 'paper_table_v2grid_summary.csv'}")