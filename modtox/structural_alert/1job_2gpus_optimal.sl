#!/bin/sh
#SBATCH -J MD_GPU 
#SBATCH --output=namd_%j.out
#SBATCH --error=namd_%j.err
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1

module load Python/3.6.4-foss-2018a  CUDA/9.0.176 cuDNN/7.0.5-CUDA-9.0.176
python cnn_cyp.py





