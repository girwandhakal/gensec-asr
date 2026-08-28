#!/bin/bash
# What this file is for:
# The single entry point for the GenSEC pipeline. Submit with:
#
#   sbatch bash_scripts/train.sh
#
# It activates the environment, checks dependencies, and runs the one Python
# pipeline that builds the reference map, generates Whisper n-best output,
# builds the correction dataset, fine-tunes FLAN-T5, runs inference,
# postprocesses, and scores WER.

#SBATCH --job-name=gensec
#SBATCH --partition=gpu
#SBATCH --qos=gpu
# Pin the GPU model. A bare `gpu:1` is a lottery: one run landed an H100 80GB,
# the next a Tesla T4, which has a quarter of the memory, no bf16 support (so
# training silently falls back to fp32), and ran 8x slower.
# This partition offers: v100, t4, l4, a100-80, h100-80 (sinfo -p gpu -o "%N %G").
# Only a100-80 and h100-80 have both bf16 and the memory for beam search;
# swap to a100-80 if the two h100 nodes are busy.
#SBATCH --gres=gpu:h100-80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/gdhakal/gensec-asr/gensec_training.txt
#SBATCH --error=/home/gdhakal/gensec-asr/gensec_training.txt
#SBATCH --open-mode=truncate

set -euo pipefail

PROJECT_ROOT="${GENSEC_PROJECT_ROOT:-/home/gdhakal/gensec-asr}"
CONFIG_FILE="$PROJECT_ROOT/configs/baseline.yaml"
LOG_FILE="$PROJECT_ROOT/gensec_training.txt"

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
conda activate gensec_env
set -u

export OMP_NUM_THREADS=8
# Beam search allocates and frees large, unevenly sized blocks every step, which
# fragments the caching allocator badly enough to OOM with GBs nominally free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_NUM_THREADS=8
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

echo "===== DEPENDENCY CHECK ====="
if python -c "import torch, transformers, datasets, librosa, pandas, sklearn, yaml" >/dev/null 2>&1; then
  echo "Dependencies are already installed."
else
  echo "Dependencies missing or broken; installing the pinned requirements..."
  python -m pip install -r "$PROJECT_ROOT/envs/requirements.txt"
  python -c "import torch, transformers, datasets, librosa, pandas, sklearn, yaml"
  echo "Dependencies installed successfully."
fi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "===== PYTHON PIPELINE ====="
python -u "$PROJECT_ROOT/scripts/run_pipeline.py"

echo "Finished: $(date)"
