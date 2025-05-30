#!/bin/bash
#SBATCH --job-name=tuneCenter_array
#SBATCH --output=logs/tuneCenter_%A_%a.out
#SBATCH --error=logs/tuneCenter_%A_%a.err
#SBATCH --array=1-18
#SBATCH --cpus-per-task=2
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --account=saonli

# Define the parameter grid (manually or dynamically)
PRECS=(0.01 0.1)
ALPHAS=(0.1 1.0 2.0)
BETAS=(0.1 1.0 2.0)

len_PREC=${#PRECS[@]}
len_ALPHA=${#ALPHAS[@]}
len_BETA=${#BETAS[@]}

# Total combinations
total=$((len_PREC * len_ALPHA * len_BETA))
# Ensure task ID is in range
# if [ "$SLURM_ARRAY_TASK_ID" -ge "$total" ]; then
#     echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of parameter combinations $total"
#     exit 1
# fi

# Compute indices
id=$SLURM_ARRAY_TASK_ID
i=$((id / (len_ALPHA * len_BETA)))
j=$(( (id / len_BETA) % len_ALPHA ))
k=$(( id % len_PREC ))

# Fetch actual parameters
PREC=${PRECS[$i]}
ALPHA=${ALPHAS[$j]}
BETA=${BETAS[$k]}

echo "Task $SLURM_ARRAY_TASK_ID: NC=8, INTER=2, PREC=$PREC, ALPHA=$ALPHA, BETA=$BETA"

cd ~/software/ProductPartitionModels.jl/simulations

# read arguments from command line
# N, nc, variance, interEffect, common, xdiff
julia --project=. --threads=2 sim.jl 150 8 0.1 1.0 2.0 0.0 2 $PREC $ALPHA $BETA 25 5000
