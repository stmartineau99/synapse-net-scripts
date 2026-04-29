#!/bin/bash
#SBATCH -p large96s
#SBATCH --job-name=prepare_deepict
#SBATCH -o ./slurm-%j.out
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

source ~/.bashrc
micromamba activate synapse-net

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts/prepare_datasets

cd $SCRIPT_DIR

python prepare_deepict.py