#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 2-00:00:00
#SBATCH --job-name checker_baseline
#SBATCH --output checker_baseline-%A_%a.out
#SBATCH --error=checker_baseline-%A_%a.err
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --account=ronimaor
#SBATCH --array=0-3   # 4 experiments: LSD convex, PSD uniform, PSD midpoint, ESD full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

## Configuration
PY_DIR=/path/to/flow-maps/py            # <-- update to your actual path
CONFIG_PATH="configs.checker"
OUTPUT_FOLDER="/path/to/outputs/checker_baseline"   # <-- update to your actual path

mkdir -p "${OUTPUT_FOLDER}"

SLURM_ID=${SLURM_ARRAY_TASK_ID}

case ${SLURM_ID} in
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

echo "=========================================="
echo "Checker Baseline Reproduction (Monge gap disabled)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "PFMM Type: ${PFMM_TYPE}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: uniform_upper_triangle (half plane)"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Batch Size: 100,000"
echo "Total Steps: 100,000"
echo "lambda_reg: 0.0 (Monge gap disabled, true no-op)"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

module purge
module load gcc/12.2.0-fasrc01          # <-- adjust if your cluster module differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax_3_17_25              # <-- adjust to your env name

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
    --slurm_id ${SLURM_ID} \
    --dataset_location "" \
    --output_folder "${OUTPUT_FOLDER}" \
    --lambda_reg_override 0.0

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: fmm2"
echo "=======================================