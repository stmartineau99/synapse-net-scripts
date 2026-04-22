#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=predict_atin
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH --qos=2h

source ~/.bashrc
micromamba activate synapse-net

CKPT=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run13/checkpoints/actin-deepict-run13

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts/inference

cd $SCRIPT_DIR

python predict_actin_deepict.py --checkpoint $CKPT
