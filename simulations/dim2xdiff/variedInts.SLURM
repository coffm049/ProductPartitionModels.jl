#!/bin/bash
#SBATCH --job-name=smaller
#SBATCH --output=../logs/smaller.out
#SBATCH --error=../logs/smaller.err
#SBATCH --cpus-per-task=2
#SBATCH --time=12:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
#SBATCH --array=0-1



PREC=0.1
ALPHA=1.0
BETA=1.0

XDIFFS=(0.8 1.0)

XDIFF=${XDIFFS[$SLURM_ARRAY_TASK_ID]}

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
julia --project=. --threads=2 sim.jl 1000 8 0.05 1.0 1.0 $XDIFF 2 $PREC $ALPHA $BETA 80 1000 1.0 1.0
