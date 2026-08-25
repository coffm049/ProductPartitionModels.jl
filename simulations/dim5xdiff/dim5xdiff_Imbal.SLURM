#!/bin/bash
#SBATCH --job-name=dim5imbal
#SBATCH --output=../logs/dim5imbal.out
#SBATCH --error=../logs/dim5imbal.err
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
# array = xdiff index (0-5) x mass index (0-2) = 18 tasks.
#SBATCH --array=0-17



PREC=0.1
ALPHA=1.0
BETA=1.0

# imbalanced cluster sizes (exp decay fractions), dims=5, common=1, full xd grid
XDIFFS=(0.0 0.2 0.4 0.6 0.8 1.0)
# DP-concentration (mass A) sweep for the sensitivity analysis (Rev 3 C2).
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate r-salso
export R_HOME="$CONDA_PREFIX/lib/R"

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
# imbalanced = parse(Int, ARGS[18])
julia --project=. --threads=2 sim.jl 1000 8 0.05 1.0 1.0 $XDIFF 5 $PREC $ALPHA $BETA 80 1000 $MASS $MASS 1 1.0 500 1
