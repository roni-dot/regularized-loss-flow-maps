#!/bin/bash
#SBATCH --job-name cifar10_monge_sweep_eps
#SBATCH --time 4-00:00:00
#SBATCH --output cifar10_monge_sweep_eps-%A_%a.out
#SBATCH --error=cifar10_monge_sweep_eps-%A_%a.err
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
# the site wrapper overrides --partition, so request the GPU type instead
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --qos=normal
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB

## Create output directories if they don't exist
mkdir -p /home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge_sweep_eps/logs
mkdir -p /home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge_sweep_eps/outputs

## Configuration
PY_DIR=/home/ronimaor/regularized-loss-flow-maps/py
CONFIG_PATH="configs.cifar10"
DATASET_LOCATION="/home/ronimaor/regularized-loss-flow-maps/datasets"
OUTPUT_FOLDER="/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge_sweep_eps/outputs"
PROJECT_NAME="self-distill-flow-maps-cifar10-monge_sweep-eps"


TASK_ID=${SLURM_ARRAY_TASK_ID}
SLURM_ID=0
SINKHORN_EPS_IDX=${SLURM_ARRAY_TASK_ID}

METHOD="LSD"
PFMM_TYPE="N/A"
STOPGRAD="convex"

# case ${LOSS_SLURM_ID} in
#     0)
#         METHOD="LSD"
#         PFMM_TYPE="N/A"
#         STOPGRAD="convex"
#         ;;
#     1)
#         METHOD="PSD uniform"
#         PFMM_TYPE="uniform"
#         STOPGRAD="convex"
#         ;;
#     2)
#         METHOD="PSD midpoint"
#         PFMM_TYPE="midpoint"
#         STOPGRAD="convex"
#         ;;
#     3)
#         METHOD="ESD (Eulerian)"
#         PFMM_TYPE="N/A"
#         STOPGRAD="full"
#         ;;
# esac

# Monge gap hyperparameters
LAMBDA_REG=0.1 ; LAMBDA_TAG=0p1
MONGE_NUM_PAIRS=1
MG_BATCH=512

case ${SINKHORN_EPS_IDX} in
    0) SINKHORN_EPS=0.01; EPS_TAG=0p01 ;;
esac

# Print job info
echo "=========================================="
echo "Monge Gap Regularized Training"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: ${SAMPLING}"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Config: ${CONFIG_PATH}"
echo "------------------------------------------"
echo "lambda_reg:       ${LAMBDA_REG}"
echo "sinkhorn_eps:     ${SINKHORN_EPS} (relative to mean cost)"
echo "monge_num_pairs:  ${MONGE_NUM_PAIRS}"
echo "mg_batch_size:    ${MG_BATCH}"
echo "------------------------------------------"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

# Hook conda into the shell and activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flow_maps

# Fix library path to prioritize conda environment libraries over system ones
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Change to project directory
cd $PY_DIR || exit

echo "Starting training..."
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"

# Run training with the Monge gap regularizer enabled
python -u launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --dataset_location "${DATASET_LOCATION}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --lambda_reg_override ${LAMBDA_REG} \
    --sinkhorn_eps_override ${SINKHORN_EPS} \
    --monge_num_pairs_override ${MONGE_NUM_PAIRS} \
    --mg_batch_override ${MG_BATCH} \
    --wandb_project_override "${PROJECT_NAME}" \
    --wandb_name_additional "lam${LAMBDA_TAG}_eps${EPS_TAG}_K${MONGE_NUM_PAIRS}_mgb${MG_BATCH}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: ${PROJECT_NAME}"
echo "=========================================="

exit
