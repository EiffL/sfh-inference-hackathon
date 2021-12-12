#!/bin/bash
#SBATCH --job-name=TNG100_kin_train
#SBATCH --output=TNG100_kin_train%j.out
#SBATCH --error=TNG100_kin_train%j.err
#SBATCH --nodes=1
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --reservation=hackathon
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3
module purge
module load tensorflow-gpu
set -x
srun python main.py