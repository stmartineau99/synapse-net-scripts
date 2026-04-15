#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=deepict_run11
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH -C 80gb_vram

source ~/.bashrc
micromamba activate synapse-net

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/experiments/deepict/run11

cd $SCRIPT_DIR

python prepare_training_data.py && \
    python train_actin_segmentation.py
