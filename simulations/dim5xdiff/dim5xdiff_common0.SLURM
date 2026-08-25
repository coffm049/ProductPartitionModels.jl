#!/bin/bash
#SBATCH --job-name=dim5common0
#SBATCH --output=../logs/dim5common0.out
#SBATCH --error=../logs/dim5common0.err
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
# run only tasks 0-1 (xd 0.0, 0.2): tasks 2-5 (xd 0.4-1.0) already completed.
# To rerun the full grid, comment out the array line above this block and
# uncomment the "full grid" line below.
#SBATCH --array=0-1
##SBATCH --array=0-5



PREC=0.1
ALPHA=1.0
BETA=1.0

# degenerate case beta* = 0: common=0, dims=5, full xd grid
XDIFFS=(0.0 0.2 0.4 0.6 0.8 1.0)

XDIFF=${XDIFFS[$SLURM_ARRAY_TASK_ID]}

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

# optional: needed only for SALSO metrics (R 'salso' package)
module unload R 2>/dev/null || true
export R_HOME="/projects/standard/gdc/public/envs/r-salso/lib/R"
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
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
julia --project=. --threads=2 sim.jl 1000 8 0.05 1.0 0.0 $XDIFF 5 $PREC $ALPHA $BETA 80 1000 1.0 1.0 1 1.0 500
