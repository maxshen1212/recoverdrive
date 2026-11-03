#!/usr/bin/env bash
# Download and extract nuScenes v1.0-mini (~4.2 GB) into $NUSCENES_ROOT.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$NUSCENES_ROOT"
cd "$NUSCENES_ROOT"
if [ ! -f v1.0-mini.tgz ] || [ "$(stat -c %s v1.0-mini.tgz)" != "4167696325" ]; then
  wget -q -c https://www.nuscenes.org/data/v1.0-mini.tgz
fi
tar -xzf v1.0-mini.tgz
rm v1.0-mini.tgz
ls "$NUSCENES_ROOT"
