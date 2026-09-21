#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=supervised_training
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH -C 80gb_vram

source ~/.bashrc
micromamba activate synapse-net

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts/training

cd $SCRIPT_DIR

python supervised_training.py --config $1
