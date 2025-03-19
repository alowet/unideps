#!/bin/bash

#SBATCH --job-name=unideps
#SBATCH --output=slurm/out/%J_%x_%a_%t.out
#SBATCH --error=slurm/err/%J_%x_%a_%t.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alowet@g.harvard.edu
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem 64G

module load python/3.10.9-fasrc01
module load cuda/12.4
module load cudnn
conda activate interp

srun -N 1 -c $SLURM_CPUS_PER_TASK python refit.py
