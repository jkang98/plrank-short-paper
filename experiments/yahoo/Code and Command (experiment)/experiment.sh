#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=experiment_nn_yahoo_50
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=90:00:00
#SBATCH --mail-user=j.kang@uva.nl
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL


# Set-up the environment.
source ~/.bashrc
conda activate myenv

# Start the experiment.
# 50 is cutoff
python3 experiment_nn_yahoo.py 50
