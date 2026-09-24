#!/bin/bash
#SBATCH --job-name=collect_sim
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-collect_sim-%j.out
#SBATCH --error=slurm-collect_sim-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate julia

cd ~/papers/ABCD-adhd/simulations
julia --project=. collect_sim_results.jl

echo "Simulation results collected"
ls -la paper_sim_*.csv