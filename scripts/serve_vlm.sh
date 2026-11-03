#!/usr/bin/env bash
# Serve the diagnosis VLM through vLLM's OpenAI-compatible API (separate env in serving/).
# Usage: CUDA_VISIBLE_DEVICES=0 ./scripts/serve_vlm.sh [extra vllm args]
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
SERVED_NAME="${VLM_SERVED_NAME:-qwen3-vl-8b-instruct}"
PORT="${VLM_PORT:-8000}"
mkdir -p "$PROJECT_ROOT/outputs/logs"
# The server's system nvcc is CUDA 11.5, too old for FlashInfer's JIT sampling kernels; use PyTorch sampling.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec uv run --project "$PROJECT_ROOT/serving" vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "${VLM_GPU_UTIL:-0.6}" \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  "$@"
