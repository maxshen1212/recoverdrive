#!/usr/bin/env bash
# Download the model weights into the Hugging Face cache ($HF_HOME, default ~/.cache/huggingface).
# Usage: ./scripts/download_models.sh [--ablation]
#   default     SAM 3 (~3.4 GB, gated) and Qwen3-VL-8B-Instruct (~17.5 GB): everything the main experiments need
#   --ablation  also Qwen3-VL-4B-Instruct (~9 GB) and Qwen3-VL-8B-Thinking (~17.5 GB), used only by the E5 ablation
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"

if ! uv run hf auth whoami > /dev/null 2>&1; then
  echo "Not logged in to Hugging Face. Run: uv run hf auth login" >&2
  exit 1
fi
# SAM 3: only the files transformers loads (the repo also holds an extra 3.5 GB checkpoint).
uv run hf download facebook/sam3 config.json model.safetensors processor_config.json tokenizer.json \
    tokenizer_config.json special_tokens_map.json vocab.json merges.txt || {
  echo "SAM 3 is gated: request access at https://huggingface.co/facebook/sam3 and wait for approval." >&2
  exit 1
}
uv run hf download Qwen/Qwen3-VL-8B-Instruct
if [ "${1:-}" = "--ablation" ]; then
  uv run hf download Qwen/Qwen3-VL-4B-Instruct
  uv run hf download Qwen/Qwen3-VL-8B-Thinking
fi
echo "Done. Models are in ${HF_HOME:-$HOME/.cache/huggingface}/hub"
