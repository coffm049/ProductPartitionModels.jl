#!/bin/bash
#SBATCH --job-name=large
#SBATCH --output=../logs/large.out
#SBATCH --error=../logs/large.err
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
#SBATCH --array=0-8

# * 10

PREC=0.01
ALPHA=0.1
BETA=0.1

INTERS=(0.25 0.5 1.0)
XDIFFS=(0.25 0.5 1.0)

len_INTER=${#INTERS[@]}
len_X=${#XDIFFS[@]}

i=$(( SLURM_ARRAY_TASK_ID / len_INTER ))
j=$(( SLURM_ARRAY_TASK_ID %  len_X))

INTER=${INTERS[$i]}
XDIFF=${XDIFFS[$j]}

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
julia --project=. --threads=2 sim.jl 1000 10 0.1 $INTER 1.0 $XDIFF 2 $PREC $ALPHA $BETA 10 5000
