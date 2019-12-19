#!/bin/sh
#SBATCH -J MD_GPU 
#SBATCH --output=namd_%j.out
#SBATCH --error=namd_%j.err
#SBATCH --ntasks=10
#SBATCH --gres=gpu:1
##SBATCH --mem=0

module load RDKit/2018.09.3-foss-2018a-Python-3.6.4 Python/3.6.4-foss-2018a  CUDA/9.0.176 cuDNN/7.0.5-CUDA-9.0.176
python new_cnn_cyp.py --just_ligands --volume






