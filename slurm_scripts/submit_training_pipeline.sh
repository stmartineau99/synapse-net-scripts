#!/bin/bash

# toggle on/off
RUN_PREPARE=false
RUN_SUPERVISED=true
RUN_DOMAIN_ADAPTATION=false

PARENT_DIR=/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts
SCRIPT_DIR=$PARENT_DIR/slurm_scripts

LOG_DIR=$PARENT_DIR/slurm_logs
JSON_DIR=$PARENT_DIR/slurm_metrics
mkdir -p $LOG_DIR $JSON_DIR

CONFIG=$PARENT_DIR/configs/deepict/deepict_run19.toml
CONFIG_NAME=$(basename $CONFIG .toml)

submit_job() {
    local job_name=$1
    local script=$2
    local config=$3
    local dependency=$4

    local dependency_flag=""
    [[ -n "$dependency" ]] && dependency_flag="--dependency=afterok:$dependency"

    local job_id=$(sbatch --job-name=$job_name $dependency_flag \
        --output="$LOG_DIR/slurm-%j_%x.out" \
        $script $config | awk '{print $4}')

    if [[ -z "$job_id" ]]; then
        echo "ERROR: Failed to submit $job_name. Aborting." >&2
        return 1
    fi
    echo "Submitted $job_name as job $job_id." >&2

    sbatch --job-name="metrics" \
        --partition=large96s \
        --dependency=afterany:$job_id \
        --output=/dev/null \
        --wrap="source ~/.bashrc && \
                micromamba activate synapse-net && \
                python $SCRIPT_DIR/collect_slurm_metrics.py $job_id --out_path $JSON_DIR/slurm-${job_id}_${job_name}.json" > /dev/null

    echo $job_id
}

JOB1_ID=""
JOB2_ID=""

if [[ "$RUN_PREPARE" == true ]]; then
    JOB1_ID=$(submit_job "prepare_${CONFIG_NAME}" \
        $SCRIPT_DIR/sbatch_prepare_training.sh $CONFIG "") || exit 1
fi

if [[ "$RUN_SUPERVISED" == true ]]; then
    JOB2_ID=$(submit_job "SL_${CONFIG_NAME}" \
        $SCRIPT_DIR/sbatch_supervised_training.sh $CONFIG $JOB1_ID) || exit 1
fi

if [[ "$RUN_DOMAIN_ADAPTATION" == true ]]; then
    JOB3_ID=$(submit_job "UDA_${CONFIG_NAME}" \
        $SCRIPT_DIR/sbatch_domain_adaptation.sh $CONFIG $JOB2_ID) || exit 1
fi
