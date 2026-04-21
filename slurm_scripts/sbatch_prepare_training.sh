#!/bin/bash
#SBATCH -p large96s
#SBATCH --job-name=prepare_training
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

source ~/.bashrc
micromamba activate synapse-net

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts

cd $SCRIPT_DIR

python prepare_training.py --config $1
