#!/bin/bash
#SBATCH -p standard96:shared
#SBATCH --job-name=test_actin_occ
#SBATCH --array=0-9%5
#SBATCH -t 20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G

CONFIG=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/simulation/configs/s0.toml

OUT_ROOT=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/simulation/testing/run1
OUT_DIR=$OUT_ROOT/task$SLURM_ARRAY_TASK_ID
mkdir -p $OUT_DIR

source ~/.bashrc
micromamba activate simulation-main

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/source/polnet-synaptic/scripts/data_gen
cd $SCRIPT_DIR

# keep MT_PMER_OCC constant, vary ACTIN_PMER_OCC 
START_VAL=1
STEP=0.5
ACTIN_PMER_OCC=$(python3 -c "print(f'{$START_VAL + $SLURM_ARRAY_TASK_ID * $STEP:.2f}')")

MT_PMER_OCC=0.1

python all_features_argument.py \
    --config $CONFIG \
    --out_dir $OUT_DIR \
    --actin_pmer_occ $ACTIN_PMER_OCC \
    --mt_pmer_occ $MT_PMER_OCC