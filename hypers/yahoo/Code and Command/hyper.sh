#!/bin/sh
# The following lines instruct Slurm to allocate one CPU.
#SBATCH --job-name=hyper_nn_5_yahoo
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
# 5 is cutoff
python3 yahoo_NN_hyper.py 5
