#!/bin/bash
# What this file is for:
# The single entry point for Whisper and FLAN-T5. Submit with:
#
#   sbatch bash_scripts/train.sh
#
# Each run records Whisper candidates and FLAN-T5 predictions.

#SBATCH --job-name=gensec
#SBATCH --partition=gpu
#SBATCH --qos=gpu
# Request an H100 for consistent training throughput and BF16 support.
# A generic GPU request can land on a T4, forcing FP32 training and running
# roughly 7.5x slower than the previous H100 run. Wait for an H100 if needed.
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/gdhakal/gensec-asr
#SBATCH --output=/home/gdhakal/gensec-asr/training.txt
#SBATCH --error=/home/gdhakal/gensec-asr/training.txt
#SBATCH --open-mode=truncate

set -euo pipefail
trap 'echo "===== GENSEC JOB FAILED: $(date) =====" >&2' ERR

PROJECT_ROOT="${GENSEC_PROJECT_ROOT:-/home/gdhakal/gensec-asr}"
CONFIG_FILE="$PROJECT_ROOT/configs/baseline.yaml"
LOG_FILE="$PROJECT_ROOT/training.txt"

# Slurm already sends both streams to the log. This makes a direct
# `bash train.sh` behave the same way.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  exec > >(tee "$LOG_FILE") 2>&1
fi

echo "===== GENSEC PIPELINE STARTED ====="
echo "Repository: $PROJECT_ROOT"
echo "Started: $(date)"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing config: $CONFIG_FILE" >&2
  exit 1
fi

module load miniconda3/base
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="gensec_env"
echo "===== PREPARING ENVIRONMENT: $ENV_NAME ====="
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pip
fi
if ! conda run -n "$ENV_NAME" python -c "import torch, transformers, datasets, pyarrow, numpy; assert torch.__version__.split('+')[0] == '2.4.1'; assert transformers.__version__ == '4.44.2'; assert pyarrow.__version__ == '17.0.0'; assert numpy.__version__ == '1.26.4'" >/dev/null 2>&1; then
  echo "Installing or repairing pinned dependencies in $ENV_NAME"
  conda run -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/envs/requirements.txt"
fi
conda activate "$ENV_NAME"
set -u
echo "Environment ready: $ENV_NAME"

export OMP_NUM_THREADS=8
# Beam search allocates and frees large, unevenly sized blocks every step, which
# fragments the caching allocator badly enough to OOM with GBs nominally free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_NUM_THREADS=8
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

echo "===== DEPENDENCY CHECK ====="
python -c "import torch, transformers, datasets, pyarrow, numpy, librosa, pandas, sklearn, yaml, scipy, statsmodels, matplotlib; assert torch.cuda.is_available(), 'CUDA GPU required'; print('GPU:', torch.cuda.get_device_name(0)); print('Transformers:', transformers.__version__); print('PyArrow:', pyarrow.__version__, 'NumPy:', numpy.__version__)"

echo "===== PYTHON PIPELINE ====="
python -u "$PROJECT_ROOT/scripts/run_pipeline.py"

echo "Finished: $(date)"
