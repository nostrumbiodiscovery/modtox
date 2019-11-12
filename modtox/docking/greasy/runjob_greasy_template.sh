#!/bin/sh
#SBATCH --job-name=ARzinc8m
#SBATCH -D .
#SBATCH --output=vs_%j.out
#SBATCH --error=vs_%j.err
#SBATCH --ntasks=$ntasks
#SBATCH --cpus-per-task=1
#SBATCH --partition=smp
#SBATCH --qos=xlong
#SBATCH --time=120:00:00

module load impi
export SCHRODINGER=/gpfs/projects/bsc72/Schrodinger_SMP3
/apps/GREASY/2.1.2.1/bin/greasy $greasy_file
