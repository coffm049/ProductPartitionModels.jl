#SBATCH --mem=1G
#SBATCH --output=slurm-%A_%a.out

# Define parameter arrays
A_vals=(a1 a2)
B_vals=(b1 b2 b3)
C_vals=(c1 c2)

# Get lengths
len_A=${#A_vals[@]}
len_B=${#B_vals[@]}
len_C=${#C_vals[@]}

# Total combinations
total=$((len_A * len_B * len_C))

# Ensure task ID is in range
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of parameter combinations $total"
    exit 1
fi

# Compute indices
id=$SLURM_ARRAY_TASK_ID
i=$((id / (len_B * len_C)))
j=$(( (id / len_C) % len_B ))
k=$(( id % len_C ))

# Fetch actual parameters
A=${A_vals[$i]}
B=${B_vals[$j]}
C=${C_vals[$k]}

for ((id=0; id<total; id++)); do
    i=$((id / (len_B * len_C)))
    j=$(( (id / len_C) % len_B ))
    k=$(( id % len_C ))

    A=${A_vals[$i]}
    B=${B_vals[$j]}
    C=${C_vals[$k]}

    echo "ID=$id: A=$A, B=$B, C=$C"
done
