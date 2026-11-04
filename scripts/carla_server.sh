#!/usr/bin/env bash
# Start the CARLA 0.9.16 server headless (spec §1.4: -RenderOffScreen). Stop it with:
#   pkill -INT -f "[C]arlaUE4-Linux-Shipping"   (the brackets stop pkill from matching its own shell)
# The server has Mesa Vulkan drivers next to NVIDIA's; without VK_ICD_FILENAMES the render thread can pick a
# non-NVIDIA device and hang ("GameThread timed out waiting for RenderThread"). Vulkan adapter numbering differs
# from CUDA's: check `nvidia-smi` after start to see which GPU it uses.
# Usage: [CARLA_ROOT=...] [CARLA_PORT=2000] [CARLA_ADAPTER=1] ./scripts/carla_server.sh
set -euo pipefail
CARLA_ROOT="${CARLA_ROOT:-/data/$USER/carla/CARLA_0.9.16}"
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for f in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$f" ] && export VK_ICD_FILENAMES="$f" && break
  done
fi
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  echo "No NVIDIA Vulkan driver file (nvidia_icd.json) found; set VK_ICD_FILENAMES to its path." >&2
  exit 1
fi
[ -x "$CARLA_ROOT/CarlaUE4.sh" ] || { echo "CARLA not found at $CARLA_ROOT; set CARLA_ROOT." >&2; exit 1; }
exec "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port="${CARLA_PORT:-2000}" -graphicsadapter="${CARLA_ADAPTER:-1}" "$@"
