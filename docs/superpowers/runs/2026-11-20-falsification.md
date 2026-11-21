# Falsification log (Phase A, CPU only)

Plan: `docs/superpowers/plans/2026-11-20-course-requirements.md`, Tasks 7a and 9.

## Run

- `scripts/falsify_nuscenes.py --cache-only --faults C1 C4 L1 L2 L4 --specs R1 R2 R3 R6 --budget 60 --workers 10
  --tag cpu_v1` at commit `fbd56ff` (clean tree), with `CUDA_VISIBLE_DEVICES=""`. All 4 GPUs were busy.
- Output: `outputs/falsify/nuscenes_cpu_v1/` (`summary.md`, `trials.jsonl`, `triage.md`, one folder per trial).
- System under test: checks-only (no VLM) on the 6 dev scenes. Every SAM 3 detection came from the cache, with no
  miss: these faults reuse original images.
- Scale: 40 campaigns, one per fault × requirement × sampler, with 60 trials each, 2,400 trials in total. About 3.5 s
  per trial on 10 CPU workers.

## Random vs. guided search

| | Random | Guided (TPE) |
|---|---|---|
| Violating trials | 157 / 1,200 (13%) | 448 / 1,200 (37%) |
| Violating trials after the 10 shared start-up trials | 129 | 420 |

- TPE's first 10 trials are random samples with the same seed, so both samplers share those trials.
- Per (fault, requirement) pair, TPE found more violations in 16 of 20 pairs and the same number in 4: L2/R1 (19 each)
  and three pairs where neither search found any (L1/R1, L4/R1, L4/R6).
- R3 (recovery within 2 steps) violations were found **only** by the guided search, for all five faults.

## Triage

`scripts/triage_falsification.py` classifies the first violated obligation of every violating trial (605 trials) from
the logged frame:

| Cause | Trials | Class |
|---|---|---|
| False alarm of the checks on a clean frame | 219 | known false-alarm rate of the calibrated checks (8.6% of clean frames on dev); the search learned which scene windows contain them |
| Evidence intermittent: the check misses one frame of an active fault | 118 | detection limit; the controller restores correctly on the evidence it gets (C3 is proven) |
| Missed: fault below the detection limit of the checks | 86 | detection limit (for example C1 where the rear view happens to match the projected objects, small L2 shifts) |
| Evidence lost: < 2 reference objects | 54 | assumption A3 violated |
| First clean frame after the episode (isolation not yet re-observed) | 51 | by design: one step of recovery lag, which R2 counts and R3 allows |
| Missed: camera has < 2 reference objects | 35 | assumption A3 violated |
| The faulty sensor stays flagged after the episode | 32 | temporal checks compare with the still-faulty previous frame |
| Missed: level near the low/medium label boundary | 7 | label boundary |
| Other sensors flagged after the episode | 2 | false alarm |
| Cautious on a fault frame (other sensors flagged too) | 1 | false alarm on a second sensor |

**Bug candidates: 0.** Every class traces to one of three causes:
- a stated assumption (A3);
- the detection limit or the known false-alarm rate of the deterministic checks;
- a severity label boundary.

None traces to the logic of the recovery controller or the shield, which Tasks 4–5 proved.

## Decision

- No repair (Task 9, class c is empty).
- The residual risk is in the monitor's evidence, not in its logic. This matches the assurance case: G1 (shield) is
  proven, and G2 (detection) is the weak link.
- These results are also the starting point for the method work: a VLM that judges the images itself could cover the
  A3 and detection-limit classes.

## Phase B: C2/C3 campaigns (GPU)

- `scripts/falsify_nuscenes.py --faults C2 C3 --specs R1 R2 R3 R6 --budget 60 --tag gpu_v1` at commit `89d95d7`
  (clean tree) on gvl2, SAM 3 on a GPU for the new faulted images. Output: `outputs/falsify/nuscenes_gpu_v1/`.
- Scale: 16 campaigns (2 faults × 4 requirements × 2 samplers) with 60 trials each, 960 trials.

| | Random | Guided (TPE) |
|---|---|---|
| Violating trials | 43 / 480 (9%) | 166 / 480 (35%) |
| Violating trials after the 10 shared start-up trials | 34 | 157 |

- TPE found more violations in 6 of 8 (fault, requirement) pairs. Random found more in 2: C3/R1 (7 vs 0) and C3/R6
  (1 vs 0). Guided search is not better everywhere: when violations are rare and scattered (a C3 level just above
  the detection limit), the model TPE builds from the first trials can lead it away from them.

Triage (`scripts/triage_falsification.py --run outputs/falsify/nuscenes_gpu_v1`, 209 violating trials):

| Cause | Trials | Class |
|---|---|---|
| False alarm of the checks on a clean frame | 88 | known false-alarm rate of the checks |
| Missed: fault below the detection limit of the checks | 30 | detection limit (small rotations or shifts, mild blur) |
| Other sensors flagged after the episode | 26 | false alarm |
| Evidence intermittent: the check misses one frame of an active fault | 23 | detection limit |
| First clean frame after the episode | 14 | by design (one step of recovery lag) |
| Missed: camera has < 2 reference objects | 9 | assumption A3 violated |
| Missed: level near the low/medium label boundary | 8 | label boundary |
| Evidence lost: < 2 reference objects | 7 | assumption A3 violated |
| The faulty sensor stays flagged after the episode | 4 | temporal checks compare with the still-faulty previous frame |

**Bug candidates: 0.** The classes are the same as in Phase A.

## Phase B: replay of the worst counterexamples with Ours-full (VLM)

`--confirm 10 --method ours_full` replays the 10 lowest-robustness violating trials of each (fault, requirement)
pair with the full system (Qwen3-VL-8B, recovery, shield).

| Source campaign | Replayed | Still violated with Ours-full |
|---|---|---|
| `cpu_v1` (C1, C4, L1, L2, L4) | 147 | 145 |
| `gpu_v1` (C2, C3) | 67 | 67 |

The VLM repairs 2 of 214 counterexamples of the deterministic checks. This matches Step 1 ("the VLM confirms the
flags"): the Ours prompt tells the VLM that the checks are reliable, so it follows them, including their false alarms
and misses. This is the starting point of the VLM-centric variants (plan `2026-11-21-finish-and-vlm-variants.md`).

## Phase B: closed-loop falsification in CARLA

- `scripts/falsify_carla.py --budget 40` per (hazard, sampler), 4 campaigns, 160 runs, checks-only, Town01, on gvl2.
  The campaigns stopped at 27 runs on 2026-11-21 01:10 and were resumed with `--resume` (finished trials read back
  from their folders; two cut-off trials set aside as `trial_*.partial-*`). Outputs:
  `outputs/falsify/carla_v1_{pedestrian,lead_brake}_{random,tpe}/`.
- Search variables: route, trigger offset (±5 m), walker speed (1.0–2.5 m/s) or lead braking time (1–4 s), front
  fault (C1, black camera, rotated camera C2, LiDAR dropout in the front sector), level, window, seed.

| hazard | sampler | runs | R4 violations | R5 violations | collisions |
|---|---|---|---|---|---|
| lead car braking | random | 40 | 5 | 0 | 0 |
| lead car braking | TPE | 40 | 17 | 0 | 0 |
| pedestrian | random | 40 | 3 | 1 | 1 |
| pedestrian | TPE | 40 | 4 | 0 | 0 |

Triage (from the step logs):
- C1 (31 runs) and the black front camera (18 runs): the front camera is isolated in every run, and R4 always holds.
- LiDAR dropout in the front sector (37 runs): no R4 violation.
- Rotated front camera (C2, 5–12°, 74 runs): the checks isolate the camera in only 7 runs. All 29 R4 violations
  (no slow-down within 3 s of the fault) are undetected rotations.
- The one collision (pedestrian, random trial 31): a 7.1° rotation (medium) that started 4 m before the trigger
  raised no flag, so the monitor kept "healthy / continue" at 7.3 m/s. The TTC brake uses camera-confirmed objects
  while the camera is not isolated; it fired at TTC 0.35 s, and braking to 1.2 m/s left a gap of −0.1 m.

**Bug candidates: 0.** Every violation is a missed detection (the S6b class of the shield proof): a rotation the
camera–LiDAR cross-check does not see. Misalignment (C2) is the weak spot both on nuScenes (test F1 0.47) and in
closed loop, and the closed-loop search found a collision that E3 (C1 only) could not show.
