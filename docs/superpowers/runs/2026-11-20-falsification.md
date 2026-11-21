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

## Phase B (waits for free GPUs)

- C2 and C3 campaigns (new images, SAM 3).
- Replay of the worst trials with Ours-full (VLM).
- CARLA closed-loop falsification (R4, R5).
