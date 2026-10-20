#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --job-name=update_env
#SBATCH -o /projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/slurm-%j_%x.out
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -G A100
#SBATCH -C inet

# Updates an existing synapse-net-backbones env in place with the steps of sbatch_create_env.sh,
# plus monai for the cryosiam backbone. Safe to run again.

set -e

source ~/.bashrc

SAGE=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage
ENV_NAME=synapse-net-backbones
SAM2_REF=2b90b9f5ceec907a1c18123530e92e794ad901a4
DINOV3_REF=6876159a11b4df116f30f667f8c9888617df0751
DINOV2_REF=7764ea0f912e53c92e82eb78a2a1631e92725fc8
MICRO_SAM_REF=76974d9c5793b7b71f52eefee2aa8ab2c0807ce2

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep "CUDA Version"

micromamba activate $ENV_NAME

micromamba install -y -n $ENV_NAME -c conda-forge --freeze-installed monai

# micro_sam pulls the conda torch_em, which the editable install below replaces.
micromamba remove -y -n $ENV_NAME --force torch_em || true

SAM2_BUILD_CUDA=0 pip install --no-deps "git+https://github.com/facebookresearch/sam2.git@$SAM2_REF"
pip install --no-deps "git+https://github.com/facebookresearch/dinov3.git@$DINOV3_REF"

# micro-sam dev replaces the conda micro_sam; synapse-net gets the sam and dinov2 checkpoints from it.
pip install --no-deps "git+https://github.com/computational-cell-analytics/micro-sam.git@$MICRO_SAM_REF"

# dinov2 must be editable: a regular pip build drops dinov2/hub/cell_dino (no __init__.py),
# which breaks the torch.hub load in micro_sam.models.vfm.
[ -d $SAGE/source/dinov2 ] || git clone https://github.com/facebookresearch/dinov2.git $SAGE/source/dinov2
git -C $SAGE/source/dinov2 checkout $DINOV2_REF
pip install --no-deps -e $SAGE/source/dinov2

pip install --no-deps -e $SAGE/source/torch-em
pip install --no-deps -e $SAGE/source/synapse-net

# The training jobs may run without internet, so the cryosiam weights are cached here.
python -c "
from torch_em.model.cryozoo import CryoSiamEncoder
print('cryosiam checkpoint', CryoSiamEncoder.get_checkpoint())
"

python -c "
import torch, sam2, dinov2, dinov3, segment_anything, timm, micro_sam, monai
import micro_sam.v1.util, micro_sam.v2.util, micro_sam.models.vfm
import torch_em.model.cryozoo
import synapse_net.training
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
"

micromamba env export -n $ENV_NAME --explicit > $SAGE/synapse-net-scripts/environment_backbones.lock.txt
echo "Environment $ENV_NAME updated."
