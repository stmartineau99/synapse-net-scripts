#!/bin/bash
#SBATCH -p large96s
#SBATCH --job-name=clean_pc
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/.bashrc
micromamba activate tardis-em

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage

SEG_KEY=labels/actin        # use segmentations/<model> for model predictions
TAG=gt
EROSION=3
MERGE_RADIUS=2

SCRIPT_DIR=$PARENT_DIR/synapse-net-scripts/experiments/instance_segmentation
cd $SCRIPT_DIR

python clean_pointcloud_test.py \
    --data_dir $PARENT_DIR/data/experimental/deepict/h5 \
    --output_dir $PARENT_DIR/data/predictions/deepict/csv/instances \
    --seg_key $SEG_KEY \
    --tag $TAG \
    --erosion $EROSION \
    --merge_radius $MERGE_RADIUS
