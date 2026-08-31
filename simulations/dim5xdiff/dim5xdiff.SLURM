#!/bin/bash
#SBATCH --job-name=dim5xdiff
#SBATCH --output=../logs/dim5xdiff.out
#SBATCH --error=../logs/dim5xdiff.err
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
# array = xdiff index (0-5) x mass index (0-2) = 18 tasks.
# task id -> xi = id / 3, mi = id % 3; see MASSES below.
# 2026-08-29 resubmit: job 17210922_0 (COMPLETED), _4 (COMPLETED), _1/_2/_5 (TIMEOUT at 48h) — _3 not in SLURM mail (check sacct; add back if needed).
# Full grid for reference: 0-17. Subset below is only failed/timeout tasks for 72h rerun.
#SBATCH --array=1,2,5
##SBATCH --array=0-17
##SBATCH --array=0-5



PREC=0.1
ALPHA=1.0
BETA=1.0

# full xd grid 0.0-1.0, dims=5, common=1
XDIFFS=(0.0 0.2 0.4 0.6 0.8 1.0)
# DP-concentration (mass A) sweep for the sensitivity analysis (Rev 3 C2).
# Each (xdiff, mass) pair is one array task: task = xi * NM + mi.
MASSES=(1.0 2.0 5.0)
NX=${#XDIFFS[@]}
NM=${#MASSES[@]}

IDX=$SLURM_ARRAY_TASK_ID
XI=$(( IDX / NM ))
MI=$(( IDX % NM ))
XDIFF=${XDIFFS[$XI]}
MASS=${MASSES[$MI]}

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

# optional: needed only for SALSO metrics (R 'salso' package)
module unload R 2>/dev/null || true
export R_HOME="/projects/standard/gdc/public/envs/r-salso/lib/R"
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
# R (conda) needs a newer libstdc++ (CXXABI_1.3.15) than Julia bundles; preload it so
# R's libicui18n resolves correctly, while keeping LD_LIBRARY_PATH limited to R_HOME/lib
# so Julia's own Pango still loads (conda's older libpangoft2 must NOT be exposed).
export LD_PRELOAD="/projects/standard/gdc/public/envs/r-salso/lib/libstdc++.so.6"
export PATH="$R_HOME/bin:$PATH"

cd ~/software/ProductPartitionModels.jl/simulations

# read arguments from command line
# N = parse(Int, ARGS[1])
# nc = parse(Int, ARGS[2])
# variance = parse(Float64, ARGS[3])
# interEffect = parse(Float64, ARGS[4])
# common = parse(Float64, ARGS[5])
# xdiff = parse(Float64, ARGS[6])
# dims = parse(Int, ARGS[7])
# prec = parse(Float64, ARGS[8])
# alph = parse(Float64, ARGS[9])
# bet = parse(Float64, ARGS[10])
# reps = parse(Int, ARGS[11])
# niters = parse(Int, ARGS[12])
# massa = parse(Float64, ARGS[13])
# massb = parse(Float64, ARGS[14])
# runDPM = parse(Int, ARGS[15])
# DPMalpha = parse(Float64, ARGS[16])
# DPMiters = parse(Int, ARGS[17])
julia --project=. --threads=2 sim.jl 1000 8 0.05 1.0 1.0 $XDIFF 5 $PREC $ALPHA $BETA 80 1000 $MASS $MASS 1 1.0 500
