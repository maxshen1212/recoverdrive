# Finish Phase B, VLM-centric variants, report and release — Plan

> Unattended run (`/goal`), following `.claude/skills/autonomous-run/SKILL.md`. Run log:
> `docs/superpowers/runs/2026-11-21-finish-and-vlm-variants.md`.

**Goal (user, 2026-11-21, translated):** (1) run the results that are still missing; (2) then let the VLM do more of
the reasoning, with as little hand-made engineering as possible: try combinations such as the VLM extracting features
from the raw sensor data first, combining them with the fixed numbers of the pipeline, and a final VLM call that
analyses everything; (3) organize all results into data figures, a demo video, a table of experiment variables and a
system data-flow block diagram; (4) update the report and data on Google Docs; (5) merge the branch into `main` and
push.

**Pre-authorized by the goal:** new method variants (2); updating the Google Doc and the Drive folder
`1iJn5UZNgCUXQtAErjc7FvjrqkfstP44a` (4); merging into `main` and pushing (5).

**Still not authorized (Severe if needed):** any use of the test split for the new variants (spec §5.4: tune on dev,
freeze, run test once; the user has not frozen a new variant), editing `configs/eval.yaml` or the frozen methods.

## Why the variants (from FutureWork Step 1)

On sensor faults, checks-only matches Ours-full (F1 0.714 for both): the VLM confirms the deterministic flags and adds
no detection quality of its own. The Ours prompt tells the VLM "The deterministic checks are reliable", so it defers to
them. The falsification triage (Phase A) found the remaining errors in the checks' evidence: false alarms on clean
frames (219 of 605 violating trials), intermittent evidence (118), faults below the detection limit (86) and too few
reference objects (A3: 89). A VLM that judges the images itself could cover some of these classes.

## Variants (new methods; the frozen methods do not change)

Every variant has a `_detect` output (first answer, no recovery) and a `_full` output (the same recovery controller
and shield as Ours-full: re-ask, 3-sample vote, re-observe, isolate, verify). Agent faults (A1–A3) hit the final
decision call only, as they hit Ours' diagnosis call.

| Variant | VLM calls per frame | Inputs of the deciding call | Idea |
|---|---|---|---|
| `think` | 1 | images + check report | Neutral prompt (the checks can be wrong), free reasoning before the verdict |
| `think_nochecks` | 1 | images only | The same reasoning prompt without any deterministic evidence |
| `perceive` (two-stage) | 2 | stage-1 observations + check report + images | Stage 1: the VLM describes every sensor from the raw data (no checks). Stage 2: it weighs its own observations against the check report |
| `perceive_nochecks` | 2 | stage-1 observations + images | Two-stage, VLM only |
| `percam` | 6 cameras + 1 LiDAR + 1 | per-sensor observations (full-resolution single images) + check report | Feature extraction per sensor at higher resolution, then one synthesis call |

DEFAULT (D3): the variant prompts are written once and not tuned (the goal asks to minimize engineering); one smoke
check on 2 dev scenes is allowed to fix format errors only.

## Tasks

### T1: Finish the CARLA falsification (Phase B, B5)
- Resume the 4 campaigns `v1_<hazard>_<sampler>` (40 runs each; 27 of 160 done) with `--resume`; then replay the 5
  lowest-R5 scenarios per hazard with `ours_full` (`--confirm 5`).
- Verification: `summary.md` of each campaign has 40 runs; `carla_v1_confirm/summary.md` has 10 replays.

### T2: Triage of the GPU campaigns (B6)
- `scripts/triage_falsification.py` on `nuscenes_gpu_v1`; read `confirm_cpu_v1` / `confirm_gpu_v1` / CARLA results;
  append to `docs/superpowers/runs/2026-11-20-falsification.md`. Repairs only for real bugs (class c).
- Verification: every violating class has a cause; the bug list is stated.

### T3: Implement the variants
- `src/recoverdrive/vlm.py`: prompts and schemas; `src/recoverdrive/methods.py`: `VLMVariant` (subclass of `Ours`,
  overrides the messages) and the names in `build_methods`.
- Tests with a fake client: message content per variant, stage-1 call shared by the samples of one frame, agent fault
  only on the deciding call.
- Verification: `uv run pytest -q`, ruff, and `scripts/replay_ablation.py --check` still 4830/4830 on `test_final`.

### T4: Dev screening (seed 0)
- `run_eval.py --split dev --faults all --seeds 0` for the 5 variants and the references `ours_detect ours_full
  checks_only B0`.
- Verification: `summary.md` per tag with E1/E2 tables, 0 frame errors.

### T5: Best variants with 3 seeds, and on the falsification counterexamples
- DEFAULT (D4): pick the 2 variants with the best pooled `_full` F1 on dev seed 0 whose unnecessary + missed fallback
  is not worse than Ours-full by more than 0.02. Run seeds 1 and 2 for them and for the references.
- Replay the falsification counterexamples (`--confirm 10 --from-tag cpu_v1` and `gpu_v1`) with the best variant.
- Verification: mean ± std tables; confirm summaries.

### T6: Figures, video, tables, diagram
- Data figures (variant comparison, falsification random vs TPE, CARLA), experiment-variable table, data-flow block
  diagram (`scripts/make_dataflow_figure.py`), demo video (MP4 from the recorded E3 demo frames).
- Verification: files exist and each figure was opened and checked.

### T7: Report text (B7 and the variants)
- Final numbers in `docs/requirements-and-verification.md`; new `docs/vlm-variants.md`; provenance rows; README and
  FutureWork lines; assurance figure redrawn with the final numbers.
- Verification: every number traced to a generated file.

### T8: Google Doc and Drive
- Update the report Doc (`1TejQjwoDeW_gQMlVCWMl-G1R87ICXqNJk1hCnP4fHA8`) in place; upload figures, video and a small
  results zip to the Drive folder.
- Verification: read the Doc back.

### T9: Merge and push
- Merge `worktree-course-requirements` into `main` (in the main checkout), run the tests there, push `main` and the
  branch.
- Verification: `git log origin/main -1` shows the merge.
