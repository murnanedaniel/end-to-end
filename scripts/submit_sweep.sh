#!/bin/bash

#SBATCH -C gpu -N 1 -t 8:00:00 -c 10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -o logs/%x-%j.out
#SBATCH -J end-to-end-sweep
#SBATCH -A m1759
#SBATCH --requeue

module load esslurm
# module load pytorch/v1.2.0-gpu

conda activate exatrkx-test

echo -e "\nStarting sweeps\n"

for i in {0..7}; do
    echo "Launching task $i"
    srun -N 1 -n 1 -G 1 wandb agent murnanedaniel/EndToEnd/7jpjzq53 &
done
wait
