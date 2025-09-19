#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:02:00
#SBATCH --ntask=1
#SBATCH --cpus-pre-task=4
#SBATCH --output=hello_threads.output

cd $SLURM_SUBMIT_DIR
g++ -O2 -std=c++17 hello_threads.cpp -o hello_threads -pthread ./hello_threads