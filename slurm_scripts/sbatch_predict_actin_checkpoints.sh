#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=predict_checkpoints
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH --qos=2h

source ~/.bashrc
micromamba activate synapse-net

CONFIG=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/configs/deepict/deepict_run16.toml

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/pipeline_scripts/inference

cd $SCRIPT_DIR

python predict_actin_checkpoints.py --config $CONFIG --step 2
