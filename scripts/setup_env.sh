#!/usr/bin/env bash
# Set up both environments (.venv and serving/.venv) on this machine. The same command works on every server:
# - NVIDIA driver >= 580 (CUDA 13): the normal `uv sync` from uv.lock and serving/uv.lock.
# - Older driver, such as gvl2 (565, CUDA 12.7): the same package versions, but CUDA 12 builds of torch/torchvision
#   (main env: cu126) and of vLLM/torch (serving env: the vLLM 0.32.0 cu129 release wheel; torchcodec 0.16.0, because
#   the locked 0.17.0 has only a CUDA 13 build and vLLM fails to start without it). The marker .venv/.cuda12
#   makes scripts/env.sh set UV_NO_SYNC=1, so `uv run` does not reinstall the CUDA 13 builds from the lock.
#   After a dependency change in pyproject.toml, run this script again.
# uv picks a torch build from the driver only in `uv pip` (--torch-backend auto), not in `uv sync`, and vLLM's CUDA 12
# wheel is on no package index, so this script makes the choice.
# Usage: ./scripts/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

if [ "${driver%%.*}" -ge 580 ]; then
  echo "== driver $driver: CUDA 13 builds from the lock files"
  rm -f .venv/.cuda12 serving/.venv/.cuda12
  unset UV_NO_SYNC
  uv sync
  uv sync --project serving
else
  echo "== driver $driver: CUDA 12 builds"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  cuda_pkgs='^(torch|torchvision|torchaudio|triton|nvidia-|cuda-|cupy|flashinfer|quack-kernels|vllm)'
  pt=https://download.pytorch.org/whl

  echo "== main env (.venv): torch 2.14.1+cu126"
  uv export --frozen --no-hashes --no-emit-project --no-header --no-annotate | grep -vE "$cuda_pkgs" > "$tmp/main.txt"
  uv venv --allow-existing --python 3.12 .venv
  uv pip install --python .venv -r "$tmp/main.txt" "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" \
    --extra-index-url "$pt/cu126" --index-strategy unsafe-best-match
  uv pip install --python .venv --no-deps -e .
  touch .venv/.cuda12

  echo "== serving env (serving/.venv): vLLM 0.32.0+cu129, torch 2.13.0+cu129"
  uv export --project serving --frozen --no-hashes --no-emit-project --no-header --no-annotate \
    | grep -vE "$cuda_pkgs" > "$tmp/serving.txt"
  uv venv --allow-existing --python 3.12 serving/.venv
  uv pip install --python serving/.venv \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" "torchcodec==0.16.0+cu129" -c "$tmp/serving.txt" \
    --extra-index-url "$pt/cu129" --index-strategy unsafe-best-match
  touch serving/.venv/.cuda12
fi

.venv/bin/python -c "import torch; print('main torch', torch.__version__, 'gpu ok', torch.cuda.is_available())"
serving/.venv/bin/python -c "import torch, vllm; print('serving torch', torch.__version__, 'vllm', vllm.__version__)"
