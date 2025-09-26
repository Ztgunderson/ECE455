#!/usr/bin/env zsh
#
# SBATCH --partition=instruction
# SBATCH --time=00:02:00
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=8
# SBATCH --output=parallel_matmul.output

# Change to the directory where the script was submitted
cd $SLURM_SUBMIT_DIR

# Compile the C++ program with OpenMP support (-fopenmp)
g++ -fopenmp parallel_matmul.cpp -o parallel_matmul

# Set the number of threads for OpenMP to use (matches --cpus-per-task)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Execute the compiled program
./parallel_matmul