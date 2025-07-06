#!/bin/bash
#SBATCH --job-name=smaller
#SBATCH --output=../logs/smaller.out
#SBATCH --error=../logs/smaller.err
#SBATCH --cpus-per-task=2
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=coffm049@umn.edu
#SBATCH --array=0-3

PREC=(0.1)
ALPHAS=(3.0 5.0)
BETAS=(3.0 5.0)

len_ALPHA=${#ALPHAS[@]}
len_BETA=${#BETAS[@]}

# Compute indices
i=$((SLURM_ARRAY_TASK_ID / len_ALPHA))
j=$((SLURM_ARRAY_TASK_ID % len_BETA))

# for ((id = 0; id < 9; id++)); do
#   i=$(( $id / len_ALPHA ))
#   j=$(( $id % len_BETA ))
#   ALPHA=${ALPHAS[$i]}
#   BETA=${BETAS[$j]}
#
#   echo "$PREC $ALPHA $BETA"
# done

# Fetch actual parameters
ALPHA=${ALPHAS[$i]}
BETA=${BETAS[$j]}

echo "$PREC $ALPHA $BETA"

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
julia --project=. --threads=2 sim.jl 1000 8 1.0 5.0 5.0 10.0 2 $PREC $ALPHA $BETA 20 1000 3.0 3.0
