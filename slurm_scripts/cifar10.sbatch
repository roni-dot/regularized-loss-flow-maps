#!/bin/bash
#SBATCH --job-name cifar10_baseline
#SBATCH --time 2-00:00:00
#SBATCH --output cifar10_baseline-%A_%a.out
#SBATCH --error=cifar10_baseline-%A_%a.err
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
mkdir -p /home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_baseline/logs
mkdir -p /home/ronimaor/regularized-loss-flow-maps/outputs/cifar10_baseline/outputs

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