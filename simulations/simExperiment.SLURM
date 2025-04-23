#!/bin/bash
#SBATCH --job-name=sim_array
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G

# Define the parameter grid (manually or dynamically)
NCS=(5 10 10)
INTERS=(2.0 4.0 5.0)

NC=${NCS[$SLURM_ARRAY_TASK_ID]}
INTER=${INTERS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: NC=$NC, INTER=$INTER"

cd ~/software/ProductPartitionModels.jl/simulations

# read arguments from command line
# N, nc, variance, interEffect, common, xdiff
julia --project=. --threads=4 sim.jl 200 $NC 0.0 $INTER 3.0 0.0
