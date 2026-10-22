#!/bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100
#SBATCH -C "80gb_vram&inet"
#SBATCH --job-name=probe_patch_depth
#SBATCH -o ./slurm_logs/slurm-%j_probe_patch_depth.out
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
micromamba activate synapse-net-backbones

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts/training/probe

cd $SCRIPT_DIR

python probe_patch_depth.py "$@"
