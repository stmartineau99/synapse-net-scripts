#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=predict_actin_run8
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH --qos=2h

source ~/.bashrc
micromamba activate synapse-net

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/experiments/deepict/run8

cd $SCRIPT_DIR

python predict_actin_experimental.py
