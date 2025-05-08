#!/bin/bash
#SBATCH --job-name=tuneCenter_array
#SBATCH --output=logs/tuneCenter_%A_%a.out
#SBATCH --error=logs/tuneCenter_%A_%a.err
#SBATCH --array=1-2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G

# Define the parameter grid (manually or dynamically)
NCS=(8 8 )
INTERS=(2.0 2.0)
PRECS=(0.1 0.1)
ALPHAS=(1.0 1.0)
BETAS=(1.0 1.0)

NC=${NCS[$SLURM_ARRAY_TASK_ID]}
INTER=${INTERS[$SLURM_ARRAY_TASK_ID]}
PREC=${PRECS[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: NC=$NC, INTER=$INTER, PREC=$PREC, ALPHA=$ALPHA, BETA=$BETA"

cd ~/software/ProductPartitionModels.jl/simulations

# read arguments from command line
# N, nc, variance, interEffect, common, xdiff
julia --project=. --threads=2 tuneCenter-ppmx.jl 200 $NC 1.0 $INTER 3.0 0.0 1 $PREC $ALPHA $BETA
