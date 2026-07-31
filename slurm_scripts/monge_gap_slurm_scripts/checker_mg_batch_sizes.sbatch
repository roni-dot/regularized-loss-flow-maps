#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 2-00:00:00
#SBATCH --job-name checker_ms_bs_sweep
#SBATCH --output checker_ms_bs_sweep-%A_%a.out
#SBATCH --error=checker_ms_bs_sweep-%A_%a.err
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --array=0-15   # 4 losses x 4 mg_batch values
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

## Configuration
PY_DIR=/home/ronimaor/regularized-loss-flow-maps/py
CONFIG_PATH="configs.checker"
OUTPUT_FOLDER="/home/ronimaor/regularized-loss-flow-maps/outputs/checker_ms_bs_sweep"
PROJECT_NAME="self-distill-flow-maps-mg-batch-sweep"

mkdir -p "${OUTPUT_FOLDER}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
LOSS_SLURM_ID=$(( TASK_ID / 4 ))
MG_BATCH_IDX=$(( TASK_ID % 4 ))

case ${LOSS_SLURM_ID} in
    0)
        METHOD="LSD"
        PFMM_TYPE="N/A"
        STOPGRAD="convex"
        ;;
    1)
        METHOD="PSD uniform"
        PFMM_TYPE="uniform"
        STOPGRAD="convex"
        ;;
    2)
        METHOD="PSD midpoint"
        PFMM_TYPE="midpoint"
        STOPGRAD="convex"
        ;;
    3)
        METHOD="ESD (Eulerian)"
        PFMM_TYPE="N/A"
        STOPGRAD="full"
        ;;
esac

case ${MG_BATCH_IDX} in
    0) MG_BATCH=128  ;;
    1) MG_BATCH=512  ;;
    2) MG_BATCH=1024 ;;
    3) MG_BATCH=2048 ;;
esac

LAMBDA_REG=0.1
EXP_NAME="${METHOD// /_}_mgbatch${MG_BATCH}"

echo "=========================================="
echo "Checker Monge Gap Sweep: loss x mg_batch"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${TASK_ID}  (loss_idx=${LOSS_SLURM_ID}, mg_batch_idx=${MG_BATCH_IDX})"
echo "Method: ${METHOD}"
echo "PFMM Type: ${PFMM_TYPE}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: uniform_upper_triangle (half plane)"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Batch Size: 100,000"
echo "Total Steps: 100,000"
echo "mg_batch: ${MG_BATCH}"
echo "lambda_reg: ${LAMBDA_REG}"
echo "Experiment name: ${EXP_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

module purge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flow_maps

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $PY_DIR || exit

echo "Starting training..."
echo "Time limit: 2 days"
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Dataset: Checker (2D, 10M samples)"
echo "Network: 4-layer MLP with 512 neurons"
echo "GPU: Single RTX 6000"

python launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${LOSS_SLURM_ID} \
    --dataset_location "" \
    --output_folder "${OUTPUT_FOLDER}" \
    --lambda_reg_override ${LAMBDA_REG} \
    --mg_batch_override ${MG_BATCH} \
    --wandb_project_override "${PROJECT_NAME}" \
    --wandb_name_additional "${MG_BATCH}_mg_batch_size"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: ${PROJECT_NAME}"
echo "=========================================="

exit