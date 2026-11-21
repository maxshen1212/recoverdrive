# Run: course requirements, Phase B (GPU experiments)
- Plan: docs/superpowers/plans/2026-11-20-course-requirements.md (Phase A ledger: `.superpowers/sdd/2026-11-20-course-requirements/progress.md`)
- Goal condition: "please keep monitoring the GPUs and run the experiment once some GPUs are available."
- Started: 2026-11-20 16:40
- pre-authorized: "keep monitoring the GPUs and run the experiment once some GPUs are available" — waiting for a free
  GPU longer than 30 minutes is part of the goal, not a blocker.

## Status
Current: Phase B moves to gvl2 (8 x RTX 6000 Ada, 6 free). Both environments are built there with
`scripts/setup_env.sh` (CUDA 12 builds, same versions; 168 tests pass on gvl2), and the weights are in gvl2's default
Hugging Face cache. Next: smoke test (one SAM 3 frame, one VLM request), then B3 (C2/C3 campaigns), B4-B6.

Tasks:
- B1 (Task 8 code): `Scenario` for CARLA (defaults reproduce E3), per-step `gap_m` and `collision`, `falsify_carla.py`.
- B2 (Task 7c code): `--confirm K` mode of `falsify_nuscenes.py` (replay the worst trials with Ours-full).
- B3 (Task 7b run): C2/C3 campaigns with SAM 3 on a GPU.
- B4 (Task 7c run): Ours-full confirmation (vLLM 8B).
- B5 (Task 8 run): CARLA regression cell, then CARLA falsification.
- B6 (Task 9): triage of the GPU campaigns; repairs only for real bugs.
- B7 (Task 11): final numbers into `docs/requirements-and-verification.md` and the assurance figure.

## Checkpoints
| Task | Commit | Verification | Notes |
|---|---|---|---|
| Phase A | de57395 | 162 passed, ruff clean; replay 4830/4830 | specs, proofs, CPU falsification, triage, figure (see the ledger) |
| B1 | 2c086f3 | 167 passed, ruff clean; run_e3 imports | `Scenario` (defaults = E3, tests), run_e3 `scenario=`, per-step `gap_m`/`collision`, `falsify_carla.py`; CARLA regression cell runs in B5 |
| B2 | (this commit) | 168 passed, ruff clean; dry run exits at the VLM check | `select_worst` (test), `--confirm K --from-tag --vlm-url` replays the worst trials with Ours-full |

## Deferred questions
- D1 (B1): CARLA falsification objective. Chose: the robustness of R5 (TTC braking and gap ≥ 0), which mixes action
  ranks and meters in one minimum. If you prefer otherwise: separate campaigns per R4 and R5 (twice the CARLA time).
- D2 (B7): the Google Doc. Chose: keep the report text in `docs/requirements-and-verification.md` until Phase B ends,
  then insert it in place. If you prefer otherwise: insert the Phase A text now and update it later.

## Blockers
- 2026-11-21 00:15 (B3): BLOCKED. The user stopped the GPU watcher on this server and moved the project to gvl2,
  where 6 of 8 GPUs are free. Running Phase B there needs decisions outside the goal's scope: GPU jobs on another
  machine, rebuilding the environments there, and about 21 GB of model weights (Qwen3-VL-8B and the gated SAM 3),
  either copied from this server or downloaded with a Hugging Face login. No GPU job is running; all code is
  committed (`619343b`, plus this log).
- 2026-11-21 00:25 (B3): BLOCKED, severe (core assumption). gvl2's NVIDIA driver is 565.57.01 (CUDA 12.x at most),
  but the environments pin CUDA 13 builds (`torch==2.14.1` cu130 in the main env, vLLM 0.32 in `serving/`), which
  need driver >= 580. The stack cannot use gvl2's GPUs as it is. Options: upgrade the driver (lab admin), build a
  CUDA 12 environment for gvl2 (new pins, recorded in provenance), or wait for the old server's GPUs.
- 2026-11-21 00:40: checked the PyTorch index and wrongly concluded that `torch==2.14.1` has no CUDA 12 x86_64 wheel
  (the search stopped at the first match, an aarch64 wheel).
- 2026-11-21 (correction): CUDA 12 builds of the exact pinned versions exist. Main env: `torch==2.14.1+cu126`
  (x86_64, cp312), which driver 565 (CUDA 12.7) supports natively. `serving/`: vLLM 0.32.0 needs `torch==2.13.0`, and
  the vLLM GitHub release has `vllm-0.32.0+cu129` for x86_64, with `torch==2.13.0+cu129` on the PyTorch index. CUDA
  12.9 on a 12.7 driver relies on CUDA minor-version compatibility, so a smoke test (one SAM 3 frame, one VLM request)
  must pass first. Only the CUDA build changes, not the library versions.

## Waiting log
- 2026-11-20 16:44–22:45: watcher 1 timed out after 6 h; no card ever had >= 8 GB free with <= 50% load (all 4 at
  100%, 13 processes of other users). Watcher 2 started 22:45 (until about 04:45).
- 2026-11-21 00:10: watcher 2 stopped by the user. The project moves to the gvl2 server (8 x RTX 6000 Ada, mostly
  free): copied with rsync to `maxshen@10.137.32.108:/data/maxshen/recoverdrive` without the `.venv` folders. Phase B
  continues there. Before running: `uv sync` (main, `--project serving`, and in this worktree) and the model weights
  (Qwen3-VL-8B, SAM 3) in the Hugging Face cache.
