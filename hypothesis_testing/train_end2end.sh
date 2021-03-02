#!/bin/bash

#SBATCH -J train-end2end
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 7:00:00
#SBATCH -G 1
#SBATCH -o logs/%x-%j.out
#SBATCH -A m1759
#SBATCH -q special

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
conda activate exatrkx-test


# Single GPU training
srun -u python batch_train_jet.py
