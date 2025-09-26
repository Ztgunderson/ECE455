#!/usr/bin/env zsh
#
# SBATCH --partition=instruction
# SBATCH --time=00:02:00
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=8
# SBATCH --output=parallel_sum_timing.output

# Change to the directory where the script was submitted
cd $SLURM_SUBMIT_DIR

# Compile the C++ program with OpenMP support (-fopenmp)
g++ -fopenmp parallel_sum_timing.cpp -o parallel_sum_timing

# Execute the compiled program. The program internally loops 
# from 1 to 8 threads using the num_threads clause, utilizing the 
# 8 CPUs requested by --cpus-per-task.
./parallel_sum_timing