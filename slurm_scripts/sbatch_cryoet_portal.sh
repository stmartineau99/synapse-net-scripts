#!/bin/bash
#SBATCH -p large96s
#SBATCH --job-name=get_dataset
#SBATCH -o ./slurm-%j.out
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=inet

source ~/.bashrc
micromamba activate synapse-net
export IMOD_DIR=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/imod-5.1.0-ucflk2pud47w7jj27xr5zzitis7kredg
source $IMOD_DIR/IMOD-linux.sh

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage

DATA_DIR=$PARENT_DIR/data/experimental/deepict
SCRIPT_DIR=$PARENT_DIR/synapse-net-scripts/pipeline_scripts/prepare_datasets
cd $SCRIPT_DIR

python get_dataset_cryoet_portal.py 10002 \
    --output_dir $DATA_DIR/downloaded \
    --download_frames \
    --download_tiltseries
