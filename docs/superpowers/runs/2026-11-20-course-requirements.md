# Run: course requirements, Phase B (GPU experiments)
- Plan: docs/superpowers/plans/2026-11-20-course-requirements.md (Phase A ledger: `.superpowers/sdd/2026-11-20-course-requirements/progress.md`)
- Goal condition: "please keep monitoring the GPUs and run the experiment once some GPUs are available."
- Started: 2026-11-20 16:40
- pre-authorized: "keep monitoring the GPUs and run the experiment once some GPUs are available" — waiting for a free
  GPU longer than 30 minutes is part of the goal, not a blocker.

## Status
Current: waiting for a GPU (all code for Phase B is written and tested). GPU watcher running (`wait_gpu.sh`: a card
with >= 8 GB free and <= 50% load for 5 minutes). Next: B3 (C2/C3 campaigns) when the watcher fires, then B4-B6.

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

## Waiting log
- 2026-11-20 16:44–22:45: watcher 1 timed out after 6 h; no card ever had >= 8 GB free with <= 50% load (all 4 at
  100%, 13 processes of other users). Watcher 2 started 22:45 (until about 04:45).
- 2026-11-21 00:10: watcher 2 stopped by the user. The project moves to the gvl2 server (8 x RTX 6000 Ada, mostly
  free): copied with rsync to `maxshen@10.137.32.108:/data/maxshen/recoverdrive` without the `.venv` folders. Phase B
  continues there. Before running: `uv sync` (main, `--project serving`, and in this worktree) and the model weights
  (Qwen3-VL-8B, SAM 3) in the Hugging Face cache.
