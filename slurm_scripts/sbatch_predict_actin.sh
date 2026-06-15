#!/bin/bash
#SBATCH -p grete:interactive
#SBATCH --job-name=predict_actin
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -G 1g.20gb:1
#SBATCH --qos=2h

source ~/.bashrc
micromamba activate synapse-net

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/

DATASET=deepict
RUNS=(run16 run17 run18 run19 run20)

SCRIPT_DIR=$PARENT_DIR/synapse-net-scripts/pipeline_scripts/inference
cd $SCRIPT_DIR

for RUN in "${RUNS[@]}"; do
    python predict_actin.py \
        --config $PARENT_DIR/synapse-net-scripts/configs/$DATASET/${DATASET}_${RUN}.toml \
        --data_dir $PARENT_DIR/data/experimental/deepict/h5/subvolumes/train
done