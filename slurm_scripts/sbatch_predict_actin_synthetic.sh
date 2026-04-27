#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=predict_actin
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH --qos=2h

source ~/.bashrc
micromamba activate synapse-net

RUNS=(run16)

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/
CONFIG_DIR=$PARENT_DIR/configs/deepict
SCRIPT_DIR=$PARENT_DIR/pipeline_scripts

cd $SCRIPT_DIR

for RUN in "${RUNS[@]}"; do
    python predict_actin_synthetic.py --config $CONFIG_DIR/deepict_${RUN}.toml
done
