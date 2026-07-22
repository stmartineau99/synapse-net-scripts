#!/bin/bash
#SBATCH -p grete:interactive
#SBATCH --job-name=sweep_dist_threshold
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/experiments/instance_segmentation/slurm-%j_%x.out
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1g.20gb:1
#SBATCH --qos=2h

source ~/.bashrc
micromamba activate tardis-em
set -euo pipefail

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage
SCRIPT_DIR=$PARENT_DIR/synapse-net-scripts/experiments/instance_segmentation
cd "$SCRIPT_DIR"

python sweep_dist_threshold.py \
    --data_path "$PARENT_DIR/data/experimental/deepict/h5/00004.h5" \
    --seg_key labels/actin \
    --pixel_size 10 \
    --erosion 3 \
    --out_dir "$PARENT_DIR/data/predictions/deepict/csv/instances" \
    --tag greedy
