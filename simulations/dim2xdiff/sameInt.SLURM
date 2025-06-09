#!/bin/bash
#SBATCH --job-name=same
#SBATCH --output=../logs/same.out
#SBATCH --error=../logs/same.err
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

PREC=0.01
ALPHA=2.0
BETA=1.0

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
julia --project=. --threads=2 sim.jl 150 8 0.01 1.0 1.0 1.0 2 $PREC $ALPHA $BETA 50 5000
