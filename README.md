# RecoverDrive

A fault-aware monitor for autonomous-driving perception: **detect → diagnose → recover → verify → fallback**.
Deterministic checks compare the LiDAR with six cameras, Qwen3-VL decides which sensor is faulty, a recovery
controller isolates it, and a rule-based shield picks *continue*, *slow_down* or *stop*. Evaluated on nuScenes-mini
with injected faults and in CARLA.

## Requirements

| | Minimum |
|---|---|
| OS / GPU | Linux, one NVIDIA GPU with ≥ 32 GB (Qwen3-VL-8B server ~24 GB + SAM 3 ~3 GB). Tested on RTX 6000 Ada 48 GB. |
| Driver | NVIDIA driver with CUDA 13 support (tested: 580) |
| Tools | [uv](https://docs.astral.sh/uv/) ≥ 0.12 (installs Python 3.12) |
| Disk | ~40 GB: Python envs ~12 GB, nuScenes-mini ~5 GB, model weights ~21 GB |
| Access | Hugging Face account with access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3) |
| Optional | [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases) (~28 GB) for the closed-loop experiment |

## Repository layout

```
src/recoverdrive/   pipeline package: checks, faults, VLM client, recovery, shield, methods, metrics, runner
  adapters/         nuScenes and CARLA → Frame
  detectors/        SAM 3 (default) and Qwen3-VL grounding (ablation)
scripts/            entry points: run_eval.py (E1/E2), run_e3.py (CARLA), calibrate.py, latency_report.py,
                    replay_ablation.py, make_figures.py, serve_vlm.sh, env.sh, download_nuscenes_mini.sh
configs/            eval.yaml (frozen settings), splits.yaml (dev/test scenes), e3.yaml (CARLA overrides)
serving/            separate uv project that only holds vLLM
tests/              unit tests (pytest)
docs/superpowers/   design spec (in Traditional Chinese), plans, run logs
```

## Setup

```bash
git clone git@github.com:maxshen1212/recoverdrive.git && cd recoverdrive
export UV_CACHE_DIR=~/.cache/uv        # only if you have no /data disk (scripts/env.sh defaults to /data/$USER)
source scripts/env.sh                  # run this in every new shell
uv sync                                # main env (.venv)
uv sync --project serving              # vLLM env (serving/.venv)
uv run hf auth login                   # needed once for the gated SAM 3 weights
./scripts/download_nuscenes_mini.sh    # nuScenes v1.0-mini → data/nuscenes
uv run pytest -q                       # should pass
```

Model weights (Qwen3-VL-8B, SAM 3) download on first use.

## Quick start

**1. Start the VLM server** and wait until it answers:

```bash
CUDA_VISIBLE_DEVICES=0 VLM_GPU_UTIL=0.5 ./scripts/serve_vlm.sh > vllm.log 2>&1 &
curl http://127.0.0.1:8000/health      # HTTP 200 when ready (~2 min)
```

**2. Quick check** (one dev scene, one fault, ~5 min):

```bash
uv run python scripts/run_eval.py --split dev --scenes scene-0061 --faults C1 \
    --methods B0 ours_detect ours_full --seeds 0 --tag quick
```

It prints the E1 (detection) and E2 (fallback) tables and saves them to `outputs/eval/quick/summary.md`.

**3. Main experiment** (test split, all methods and faults, 3 seeds; ~3 h with two VLM servers):

```bash
uv run python scripts/run_eval.py --split test --faults all \
    --methods B0 B1 B2 B3 ours_detect ours_full --seeds 0 1 2 --tag test_final
```

- The test split runs only with `frozen: true` in `configs/eval.yaml` (already set) and a clean git tree.
- Runs are resumable: rerun the same command to finish missing runs.
- A second server (`VLM_PORT=8001` on another GPU) doubles throughput; servers that are not running are skipped.

**4. Stop the server** when done (it holds ~24 GB): `pkill -INT -f "vllm serve"`

## Other experiments

| Experiment | Command |
|---|---|
| Dev split, faults C1/L1/A1 | `uv run python scripts/run_eval.py --split dev --faults C1 L1 A1 --methods ours_detect --seeds 0 --tag p1` |
| Threshold calibration (dev) | `uv run python scripts/calibrate.py --tag calib_dev` |
| E3 closed loop (CARLA) | `CARLA_ROOT=<path to CARLA_0.9.16> ./scripts/carla_server.sh &`, then `uv run python scripts/run_e3.py --seeds 0 1 2 --tag e3_final` |
| E4 latency | two controlled runs tagged `e4_ours` / `e4_baselines` with `--workers 1 --sequential --no-precompute --cache <new file>`, then `uv run python scripts/latency_report.py` |
| E5 recovery ablations (offline) | `uv run python scripts/replay_ablation.py --tag test_final` |
| E5 component ablations | add `--ablation no_checks` / `qwen_grounding`, or `--vlm-model` to the test command |
| Figures and tables | `uv run python scripts/make_figures.py` (writes `report_figures/`) |

Results go to `outputs/` and `report_figures/` (not tracked). Lint: `uv run ruff check src tests scripts`.
