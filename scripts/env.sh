# Source this before running project commands: `source scripts/env.sh`.
# Keeps the uv cache off the nearly full root disk and pins shared GPUs.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/$USER/.cache/uv}"
if [ -z "${NUSCENES_ROOT:-}" ]; then
  NUSCENES_ROOT="$PROJECT_ROOT/data/nuscenes"
  # data/ is gitignored, so a git worktree has none: use the main checkout's copy.
  _rd_git="$(git -C "$PROJECT_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [ ! -d "$NUSCENES_ROOT" ] && [ -n "$_rd_git" ] && [ -d "$(dirname "$_rd_git")/data/nuscenes" ]; then
    NUSCENES_ROOT="$(dirname "$_rd_git")/data/nuscenes"
  fi
  unset _rd_git
fi
export NUSCENES_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
