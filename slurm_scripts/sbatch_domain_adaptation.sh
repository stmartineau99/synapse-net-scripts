#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=deepict_run8
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=40G
#SBATCH -G A100
#SBATCH -C 80gb_vram

source ~/.bashrc
micromamba activate synapse-net

export TMPDIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/tmp_$SLURM_JOB_ID
mkdir -p $TMPDIR

echo $TMPDIR
df -h $TMPDIR

SCRIPT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/experiments/deepict/run8

cd $SCRIPT_DIR

python train_domain_adaptation.py

rm -r $TMPDIR