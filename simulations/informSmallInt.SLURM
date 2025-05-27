#!/bin/bash
#SBATCH --job-name=common2
#SBATCH --output=logs/common2.out
#SBATCH --error=logs/common2.err
#SBATCH --cpus-per-task=2
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
#SBATCH --account=saonli

PREC=0.01
ALPHA=3.0
BETA=0.01

echo "Task: NC=8, INTER=2, PREC=$PREC, ALPHA=$ALPHA, BETA=$BETA"

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
julia --project=. --threads=2 sim.jl 150 8 0.1 1.0 2.0 0.0 3 $PREC $ALPHA $BETA 50 5000
