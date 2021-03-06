#!/bin/bash

#SBATCH -J train-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 4:00:00
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -A m1759
#SBATCH -q special

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
conda activate exatrkx-test

# Single GPU training
srun -u python lightning_train.py $@
