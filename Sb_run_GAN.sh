#!/bin/bash

#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --partition=vgpu
#SBATCH --job-name="GAN"
# SBATCH --mail-user=s4744924@student.uq.edu.au
# SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=x%j.out
#SBATCH --error=x%j.err

module load cuda
conda activate pytorch2

srun python GAN_CelebA_working.py
