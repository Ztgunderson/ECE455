#!/usr/bin/env zsh
#
# SBATCH --partition=instruction
# SBATCH --time=00:01:00
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=8
# SBATCH --output=parallel_sum.output

# Change to the directory where the script was submitted
cd $SLURM_SUBMIT_DIR

# Compile the C++ program with OpenMP support
g++ -fopenmp parallel_sum.cpp -o parallel_sum

# Set the number of threads for OpenMP to use (matches --cpus-per-task)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Execute the compiled program
./parallel_sum