#!/bin/bash
#SBATCH -p large96s
#SBATCH --job-name=analysis
#SBATCH -o ./slurm_logs/slurm-%j_analysis.out
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

source ~/.bashrc
micromamba activate synapse-net

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts/analysis

cd $SCRIPT_DIR

python plot_intensity_distributions.py
