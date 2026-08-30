#!/bin/bash
#SBATCH --job-name=cifar10_monge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06-00:00
#SBATCH --qos=pakman
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:rtx_6000:4
#SBATCH --mem=64GB
#SBATCH --output=/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge/loss_method_%A_%a.out
#SBATCH --error=/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge/loss_method_%A_%a.err
#SBATCH --array=0  # 4 experiments: LSD convex, PSD uniform, PSD midpoint, ESD full

## Create output directories if they don't exist
mkdir -p /home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge/logs

## Configuration
PY_DIR=/home/ronimaor/regularized-loss-flow-maps/py
CONFIG_PATH="configs.cifar10"
DATASET_LOCATION="/home/ronimaor/regularized-loss-flow-maps/datasets"
OUTPUT_FOLDER="/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_monge/outputs"
PROJECT_NAME="self-distill-flow-maps-cifar10-monge"
mkdir -p "${OUTPUT_FOLDER}"

# Monge gap hyperparameters (matching the pre-bugfix sweep in
# monge_gap_slurm_scripts/cifar10/cifar10_monge.sbatch, rerun here after the
# diag/offdiag multi-GPU interleaving fix). No OT coupling override is
# passed -- configs.cifar10 already defaults ot_coupling=None.
LAMBDA_REG=0.1
SINKHORN_EPS=1e-4 
MONGE_NUM_PAIRS=1  # currently inert: _sample_mg_pairs always uses a single
                   # fixed (tmin, tmax) pair regardless of this value
MG_BATCH=128

# Map array task ID to experiment
SLURM_ID=${SLURM_ARRAY_TASK_ID}

# Get experiment details for logging
case ${SLURM_ID} in
    0)
        METHOD="LSD"
        STOPGRAD="convex"
        SAMPLING="uniform_upper_triangle"
        ;;
    1)
        METHOD="PSD uniform"
        STOPGRAD="convex"
        SAMPLING="uniform_upper_triangle"
        ;;
    2)
        METHOD="PSD midpoint"
        STOPGRAD="convex"
        SAMPLING="uniform_upper_triangle"
        ;;
    3)
        METHOD="ESD (Eulerian)"
        STOPGRAD="full"
        SAMPLING="uniform_upper_triangle"
        ;;
esac

# Print job info
echo "=========================================="
echo "Monge Gap Regularized Training (post diag/offdiag bugfix)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: ${SAMPLING}"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Config: ${CONFIG_PATH}"
echo "------------------------------------------"
echo "lambda_reg:       ${LAMBDA_REG}"
echo "sinkhorn_eps:     ${SINKHORN_EPS}"
echo "monge_num_pairs:  ${MONGE_NUM_PAIRS} (inert, see comment above)"
echo "mg_batch_size:    ${MG_BATCH}"
echo "------------------------------------------"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

module purge
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate flow_maps || { echo "ACTIVATE FAILED"; exit 1; }
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

# Change to project directory
cd $PY_DIR || exit

echo "Starting training..."
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "relative_eps: mean"

# Run training with the Monge gap regularizer enabled (no OT coupling)
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
    --wandb_name_additional "lam${LAMBDA_REG}_eps${SINKHORN_EPS}_mgb${MG_BATCH}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: ${PROJECT_NAME}"
echo "=========================================="

exit
