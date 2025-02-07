#!/bin/bash

#SBATCH --job-name=unideps
#SBATCH --output=slurm/out/%J_%x_%a_%t.out
#SBATCH --error=slurm/err/%J_%x_%a_%t.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH -N=1
#SBATCH -n=1
#SBATCH -c=2
#SBATCH --mem=32G

module load python/3.10.9-fasrc01
module load cuda/12.4
module load cudnn
conda activate relcomp

srun -N 1 -c $SLURM_CPUS_PER_TASK python main.py
