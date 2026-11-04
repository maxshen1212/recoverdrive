# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

RecoverDrive is a course term project. It builds a fault-aware monitor for autonomous driving that runs detect → diagnose → recover → verify → fallback. The presentation is due Dec 1, 2026 and the final report Dec 10, 2026.

Phases 0–5 are done: the full detect → diagnose → recover → verify → fallback pipeline, baselines B0–B3, the nuScenes evaluation (E1/E2, test split run once from frozen commit `5d5283c`), CARLA closed loop (E3), latency (E4) and ablation (E5). Results are in `REPORT.md`. Phase 6 (slides, demo video) remains.

| Path | What it is |
|---|---|
| `docs/superpowers/specs/2026-11-03-recoverdrive-design.md` | Design spec, written in Traditional Chinese. Source of truth for architecture, fault list, metrics, experiments (E0–E5), and phase plan. Status: pending review. |
| `src/recoverdrive/` | The pipeline package (layout below). |
| `configs/` | `splits.yaml` (fixed dev/test scene split) and per-experiment configs such as `e0.yaml`. |
| `scripts/` | Shell helpers and experiment entry points. |
| `serving/` | Separate uv project that only holds vLLM. |
| `REPORT.md`, `report_figures/` | Experiment report for Phases 1–5 and its figures/tables (built by `scripts/make_figures.py`). |
| `project_proposal.tex` / `.pdf` | Local only (not tracked). Course proposal. The `.tex` is a fragment with no `\documentclass` or preamble, so it does not compile on its own. |
| `2505.02123v1.pdf` | Local only (not tracked). The DriveAgent paper this project starts from: https://arxiv.org/abs/2505.02123 |
| `DriveAgent/` | Local only (not tracked). Clone of the upstream DriveAgent code (`origin` = github.com/Paparare/DriveAgent). It is its own git repo. |

The repo (github.com/maxshen1212/recoverdrive, private) tracks code, configs, `docs/` (design spec, plans, run logs) and the report. `outputs/`, `data/`, the PDFs/`.tex`, and `DriveAgent/` are gitignored and exist only on this server.

## Commands

Always `source scripts/env.sh` first. It points `UV_CACHE_DIR` at `/data/$USER/.cache/uv` (the root disk is ~95% full), sets `NUSCENES_ROOT` (default `data/nuscenes`), and defaults `CUDA_VISIBLE_DEVICES=0`. The GPUs are shared: check `nvidia-smi` and override `CUDA_VISIBLE_DEVICES` per command.

```bash
uv sync                                         # main env (.venv, Python 3.12, torch cu130)
uv sync --project serving                       # vLLM env (serving/.venv)
./scripts/download_nuscenes_mini.sh             # nuScenes v1.0-mini -> data/nuscenes
uv run pytest                                   # all unit tests
uv run pytest tests/test_checks.py::test_match_boxes_threshold   # one test
uv run ruff check src tests scripts

CUDA_VISIBLE_DEVICES=1 VLM_GPU_UTIL=0.5 ./scripts/serve_vlm.sh   # Qwen3-VL on 127.0.0.1:8000
uv run python scripts/smoke_test.py             # one frame through SAM 3 + Qwen + Rerun
uv run python scripts/run_e0.py [--limit N] [--no-vlm]   # E0 -> outputs/e0/<timestamp>/
```

Stop the vLLM server when you finish; it holds ~24 GB on a shared GPU.

Unattended runs (`/goal`) follow the project skill `.claude/skills/autonomous-run/SKILL.md`: it covers planning without approval, checkpoint commits, deferring minor issues, and when to stop with `BLOCKED:`.

## Environment gotchas

- **Two uv environments on purpose.** vLLM pins its own `torch` and `transformers`. The main env talks to it only through the OpenAI-compatible API, so never add `vllm` to the root `pyproject.toml`.
- **numpy override.** `nuscenes-devkit` pins `numpy<2`, but `rerun-sdk` needs `numpy>=2`. The root `pyproject.toml` forces `numpy>=2` through `[tool.uv] override-dependencies`. Only the devkit's loaders and geometry helpers are used, and `tests/test_geometry.py` covers them.
- **System `nvcc` is CUDA 11.5.** Anything that JIT-compiles CUDA (FlashInfer, for example) fails. `serve_vlm.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` for this reason.
- **SAM 3** loads through transformers' `Sam3Model`/`Sam3Processor`. `facebook/sam3` is gated; the server's HF account already has access. `Qwen/Qwen3-VL-8B-Instruct` sits in the default HF cache.
- **Rerun.** The server is headless, so recordings are saved as `.rrd` files. View them locally with a viewer that matches `rerun-sdk` 0.41.

## Code layout (src/recoverdrive)

- `types.py`: the spec §2.2 dataclasses. Each camera and the LiDAR carry their own `ego_pose`, because sensor timestamps differ and projection needs the pose at each sensor's own time.
- `adapters/nuscenes.py`: builds a `Frame` for a scene keyframe. GT boxes are stored in the global frame, and only categories mapped in the config are kept.
- `faults.py`: seeded fault episodes (at least 2 frames each, with a clean frame between episodes) and the injectors for C1–C4, L1, L2, L4; A1–A3 are applied to agent calls in `agent.py`. L3 is not implemented.
- `geometry.py`: transforms, points-in-box counts, and 3D → 2D projection with near-plane clipping.
- `checks.py`: IoU, one-to-one Hungarian matching (class-agnostic), and building the `CheckReport`. `match_rate` is `None` when no GT boxes project into the camera.
- `detectors/sam3.py`: SAM 3 detector that runs all text prompts in one batched pass.
- `vlm.py`: the diagnosis agent. Uses `response_format=json_schema` and validates the reply again in `parse_diagnosis`.
- `pipeline.py`: `DetectDiagnosePipeline.step` runs one frame and records per-component latency.
- `build.py`: builds the components from a YAML config.
- `recording.py`: JSONL logs, Rerun output, and overlay PNGs.
- `cache.py` (SQLite cache for SAM 3 and VLM results), `render.py` (camera mosaic, LiDAR top view), `agent.py` (diagnosis-call wrapper with agent-fault injection), `recovery.py` (re-ask, vote, re-observe, isolate, verify), `shield.py` (rule-based actions and the acceptable-action labels), `methods.py` (B0–B3, Ours-detect, Ours-full), `metrics.py` (E1/E2), `runner.py` (parallel, resumable evaluation runs), `detectors/qwen_grounding.py` (E5 ablation), `adapters/carla_adapter.py` (CARLA → `Frame`).
- `pipeline.py` and `build.py` serve the E0 scripts only.
- Scripts: `run_eval.py` (E1/E2 and the online ablations), `calibrate.py`, `run_e3.py` + `carla_server.sh` (CARLA), `latency_report.py` (E4), `replay_ablation.py` (offline E5 ablations), `make_figures.py` (report figures/tables).

## DriveAgent/ (upstream reference)

Upstream: https://github.com/Paparare/DriveAgent (the file names below refer to that repo; the local clone is not tracked). Treat it as read-only reference. The spec does not try to reproduce the paper's numbers: the code does not run, and part of the data (R2/R3) was never released. Released data: Hugging Face dataset `Parechan/driveagent`.

How it works:
- Two scripts, each an LLM text chain over "bags": folders named `hou-*` under `--data-root`, each with `bag_information/object_descriptions.csv` and images in `img/`.
- `vehicle.py` (sensor abnormality) runs first. It reads CSV columns `object_description` (LiDAR text) and `description` (visual text) and writes `vehicle_*` columns, including `vehicle_expert_output`.
- `enviroment.py` (environment change analysis; the filename typo is upstream) runs second. It captions the image named in `env_picture_title` with `gpt-4o-mini`, then compares frame t-1 with frame t using those captions plus `vehicle_expert_output`.
- The CSV is the cache: each LLM step writes its own column, rows with a filled column are skipped, and the CSV is rewritten after every row.
- `--framework zero-shot | cot | self-refine | full` selects which chains run. These map to baselines B1–B3 in the spec.
- Prompts are in `vehicle_prompts.json` (nested under key `vehicle`) and `environment_prompts.json` (nested under key `environment`). Models are hard-coded to OpenAI `gpt-4o` / `gpt-4o-mini`.

Known breakage, relevant when reusing it:
- `enviroment.py` calls functions that are never defined: `self_refine_objects`, `lidar_changes_analysis`, `cause1_analysis`, `cause2_analysis`, `cause_aggregation`, `aggregation`. Their prompts do exist in the JSON.
- `enviroment.py` looks up prompt keys at the top level, but the JSON nests them under `environment`, so every lookup raises `KeyError`. `vehicle.py` handles the nesting correctly.
- Resume is broken in both scripts: pandas reads empty cells back as `NaN`, which is truthy, so unfinished rows are skipped on a rerun.
- Both scripts parse CLI args at import time, so they cannot be imported as modules. Copy the logic and prompts instead.

## Planned architecture (summary of the spec)

Read the spec before writing pipeline code. Points that span modules:

- **Hybrid design:** geometry and math in deterministic Python; judgment in a VLM (Qwen3-VL-8B-Instruct served by vLLM through an OpenAI-compatible API). All VLM calls go through one client, so switching models (4B/8B/32B, Instruct/Thinking, or GPT-4o) is a config change only. JSON outputs use vLLM schema-constrained decoding.
- **Data flow:** source adapter (nuScenes / CARLA) → fault injector (eval only) → LiDAR objects (GT boxes + in-box point count) ‖ SAM 3 camera detection (2D boxes) → consistency checks → diagnosis agent (Qwen3-VL) → recovery controller (re-query / re-observe / isolate, then verify) → rule-based safety shield → action + logs (JSONL + Rerun).
- **Interfaces:** each module is a function from dataclass to dataclass, testable with fake data: `Frame`, `FaultLabel`, `CheckReport`, `Diagnosis`, `RecoveryResult`, `Action`. Both data adapters emit `Frame`, so everything downstream is source-agnostic.
- **Safety shield uses no LLM.** The only actions are `continue < slow_down < stop`. This order defines the "unnecessary" and "missed" fallback metrics.
- **Fault IDs:** camera C1–C4, LiDAR L1–L4, agent A1–A3. MVP covers C1, L1, A1. Detecting L3 (ghost objects) is out of scope.
- **Methods compared:** B0 (no monitor), B1–B3 (DriveAgent-style text chain re-implemented on the same Qwen3-VL: zero-shot, CoT, Self-Refine), Ours-detect, Ours-full. B1–B3 feed the same safety shield, so results differ only in detection and recovery.
- **Evaluation integrity:** nuScenes-mini is split by scene, 6 dev and 4 test. Tune prompts and thresholds on dev only, then freeze them; run the test split once for the report. Fault injection is seeded; stochastic settings use 3 seeds (mean ± std).

## Compute environment

Shared, headless lab server: 4 × RTX 6000 Ada (48 GB), driver 580 / CUDA 13.0. Always set `CUDA_VISIBLE_DEVICES` to pin GPUs. CARLA (0.9.x) must run with `-RenderOffScreen`. SAM 3 weights require accepting the license on Hugging Face before download. If a package does not support CUDA 13, the spec's fallback is Docker or pinned versions.
