#!/bin/bash
#SBATCH -p grete:interactive
#SBATCH --job-name=create_env 
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --constraint=inet
#SBATCH -G 1g.20gb:1
#SBATCH --qos=2h

cd /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/source/synapse-net

micromamba create -n synapse-net -f environment.yaml

micromamba activate synapse-net
pip install -e .