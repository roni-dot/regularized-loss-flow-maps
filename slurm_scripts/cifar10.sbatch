#!/bin/bash
#SBATCH --job-name=loss_method_comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01-00:00
#SBATCH --qos=pakman
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:rtx_6000:4
#SBATCH --mem=64GB
#SBATCH --output=/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_baseline/loss_method_%A_%a.out
#SBATCH --error=/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_baseline/loss_method_%A_%a.err
#SBATCH --array=0  # 4 experiments: LSD convex, PSD uniform, PSD midpoint, ESD full

## Create output directories if they don't exist
mkdir -p /home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_baseline/logs

## Configuration
PY_DIR=/home/ronimaor/regularized-loss-flow-maps/py
CONFIG_PATH="configs.cifar10"
DATASET_LOCATION="/home/ronimaor/regularized-loss-flow-maps/datasets"
OUTPUT_FOLDER="/home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_baseline/outputs"
PROJECT_NAME="self-distill-flow-maps-cifar10-baseline"

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
echo "Loss Method Comparison Training"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: ${SAMPLING}"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Config: ${CONFIG_PATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

module purge
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate flow_maps || { echo "ACTIVATE FAILED"; exit 1; }
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Change to project directory
cd $PY_DIR || exit

echo "Starting training..."
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"

# Run training with the specific configuration
python -u launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --dataset_location "${DATASET_LOCATION}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --wandb_project_override "${PROJECT_NAME}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: ${PROJECT_NAME}"
echo "=========================================="

exit