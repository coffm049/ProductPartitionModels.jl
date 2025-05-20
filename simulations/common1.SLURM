#!/bin/bash
#SBATCH --job-name=common2
#SBATCH --output=logs/common2.out
#SBATCH --error=logs/common2.err
#SBATCH --cpus-per-task=2
#SBATCH --time=12:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --mail-user=coffm049@umn.edu

PREC=0.01
ALPHA=1.0
BETA=0.1

echo "Task: NC=8, INTER=2, PREC=$PREC, ALPHA=$ALPHA, BETA=$BETA"

cd ~/software/ProductPartitionModels.jl/simulations

# read arguments from command line
# N, nc, variance, interEffect, common, xdiff
julia --project=. --threads=2 sim.jl 120 8 0.1 2.0 1.0 0.0 1 $PREC $ALPHA $BETA
