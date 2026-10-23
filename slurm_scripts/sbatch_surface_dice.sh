#!/bin/bash
#SBATCH -p large96s
#SBATCH --job-name=surface_dice
#SBATCH -o ./slurm_logs/slurm-%j_surface_dice.out
#SBATCH -t 30:00
#SBATCH --qos=2h
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

source ~/.bashrc
micromamba activate synapse-net-backbones

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/

DATASET=deepict
RUNS=(run44 run45)

SCRIPT_DIR=$PARENT_DIR/synapse-net-scripts/pipeline_scripts/evaluation
cd $SCRIPT_DIR

for RUN in "${RUNS[@]}"; do
    python surface_dice.py \
        --data_dir $PARENT_DIR/data/predictions/$DATASET/subvolumes \
        --model_name actin-${DATASET}-${RUN} \
        --results_dir $PARENT_DIR/synapse-net-scripts/evaluation_results
done
