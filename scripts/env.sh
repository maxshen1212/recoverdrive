# Source this before running project commands: `source scripts/env.sh`.
# Keeps the uv cache off the nearly full root disk and pins shared GPUs.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/$USER/.cache/uv}"
export NUSCENES_ROOT="${NUSCENES_ROOT:-$PROJECT_ROOT/data/nuscenes}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Set up by scripts/setup_env.sh on a driver < 580: keep `uv run` from reinstalling the CUDA 13 builds.
if [ -f "$PROJECT_ROOT/.venv/.cuda12" ]; then export UV_NO_SYNC=1; fi
