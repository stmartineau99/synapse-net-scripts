#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=create_env
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -G A100
#SBATCH -C inet

set -e

source ~/.bashrc

SAGE=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage
ENV_NAME=synapse-net-backbones
SAM2_REF=2b90b9f5ceec907a1c18123530e92e794ad901a4
DINOV3_REF=6876159a11b4df116f30f667f8c9888617df0751

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep "CUDA Version"

micromamba create -y -n $ENV_NAME -f $SAGE/synapse-net-scripts/environment_backbones.yaml
micromamba activate $ENV_NAME

# micro_sam pulls the conda torch_em, which the editable install below replaces.
micromamba remove -y -n $ENV_NAME --force torch_em || true

SAM2_BUILD_CUDA=0 pip install --no-deps "git+https://github.com/facebookresearch/sam2.git@$SAM2_REF"
pip install --no-deps "git+https://github.com/facebookresearch/dinov3.git@$DINOV3_REF"
pip install --no-deps -e $SAGE/source/torch-em
pip install --no-deps -e $SAGE/source/synapse-net

python -c "
import torch, sam2, dinov3, segment_anything, timm, micro_sam
import micro_sam.v2.util
import synapse_net.training
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
"

micromamba env export -n $ENV_NAME --explicit > $SAGE/synapse-net-scripts/environment_backbones.lock.txt
echo "Environment $ENV_NAME created."
