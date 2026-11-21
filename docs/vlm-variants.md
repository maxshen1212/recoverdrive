# VLM-centric variants of the diagnosis agent

Run: `docs/superpowers/runs/2026-11-21-finish-and-vlm-variants.md`. Code: `VLMVariant` in
`src/recoverdrive/methods.py`, prompts at the end of `src/recoverdrive/vlm.py`, tests in `tests/test_variants.py`.

## Why

Two earlier results show that, in the frozen system, the VLM adds little of its own:
- **Step 1 (test split):** on sensor faults, checks-only and Ours-full have the same F1 (0.714). The VLM confirms the
  deterministic flags.
- **Falsification replays:** Ours-full still violates 212 of the 214 worst counterexamples of the checks.

The frozen prompt tells the VLM "The deterministic checks are reliable" and asks it to use the images only to
confirm a flagged sensor, so it follows the checks, including their false alarms and misses. The falsification
triage puts the remaining errors in the checks' evidence: false alarms on clean frames, faults below the detection
limit, and cameras with fewer than two reference objects (assumption A3). A VLM that judges the raw images itself
could, in principle, cover these cases.

## The variants

The goal was to let the VLM reason more and to use less hand-made engineering. Each variant changes only the
diagnosis agent. The recovery controller, the 3-sample vote, the shield, the model (Qwen3-VL-8B-Instruct) and the
inputs (camera mosaic of the current and previous frame, LiDAR top view, check report) are the same as Ours-full.
Every variant has a `_detect` output (first answer, no recovery) and a `_full` output (with recovery).

| Variant | Calls per frame (before the vote) | What the deciding call sees | Idea |
|---|---|---|---|
| Ours (frozen) | 1 | images + check report; "the checks are reliable" | the VLM confirms the flags |
| `think` | 1 | images + check report; "the checks can be wrong" | reason step by step before the verdict |
| `think_nochecks` | 1 | images only | the same reasoning, no deterministic evidence |
| `perceive` | 2 | its own notes + check report + images | stage 1 describes every sensor from the raw data (no checks); stage 2 weighs the notes against the report |
| `perceive_nochecks` | 2 | its own notes + images | two stages, VLM only |
| `think_trust` (ablation, added under D7) | 1 | images + check report; the frozen prompt ("the checks are reliable") | Ours plus a reasoning field written before the verdict |
| `percam` | 8 | per-sensor notes + check report + current mosaic and top view | stage 1 looks at each camera alone at 960×540 with its previous image (6 calls) and at the LiDAR top views with the point count (1 call); stage 2 decides |
| `perceive_trust` (round 2) | 2 | its own notes + check report + images; the frozen prompt ("the checks are reliable") | the three-step flow (VLM features first, then the pipeline's checks, then one final VLM call) with the frozen guidance kept, plus a reasoning field |
| `imgflag` (round 2) | 2 | images + check report extended by the VLM image flag; the frozen prompt | stage 1 asks only about image quality (ok or abnormal per camera); an abnormal camera becomes one more flag `camera:<C>:vlm_image` in the report and the recovery; the frozen Ours call decides |
| `imgflag_checks` (round 2) | 1 | — (no deciding VLM call) | the same VLM image flag added to the checks-only monitor: the VLM is used only as a feature extractor |

Round 2 was added after the first results (plan section "Round 2"): `perceive_trust` tests the three-step flow (VLM features, then the checks, then one final call)
without the "checks can be wrong" instruction that hurt round 1, and the two `imgflag` variants test the direction round
1 pointed to (the VLM judges image quality only, as one more flag rather than a veto). The image check sees the current
and the previous camera mosaic and answers, per camera, whether the image is black, washed out, blurred, noisy, tilted
or rotated, shifted, or frozen; night, rain, glare, motion blur and empty roads count as normal. Its call is a feature
call, not a diagnosis call, so the agent faults A1–A3 do not reach it (as for the stage-1 calls of round 1); for
`imgflag_checks`, which has no other VLM call, the agent faults therefore have no effect.

Output of the deciding call (JSON schema): `analysis` (free reasoning, written first), `verdict` (ok, uncertain or
fault:<sensor>), `fault_type`, `confidence`, `evidence`. Stage-1 calls are not diagnosis calls: the agent faults
A1–A3 hit only the deciding call, as for Ours. The prompts were written once and not tuned (deferred question D3);
one smoke run on one dev scene checked only the output format.

## Protocol

- Dev split only (6 scenes). A new variant on the test split needs the user's approval (spec §5.4: tune on dev,
  freeze, run test once).
- Screening: all variants and the references (B0, checks-only, Ours-detect, Ours-full) over the same frames: sensor
  faults C1–C4, L1, L2, L4, seed 0 (round 1: `outputs/eval/vlm_screen_s0`, `vlm_screen_s0_trust`; round 2:
  `vlm_screen_s0_round2`).
- Changed under D7: since no variant beat Ours-full on seed 0, block-bootstrap intervals over fault episodes measure
  the gaps. Round 2 then added seeds 1–2 for the references, `think_trust` and the image-flag variants
  (`vlm_trust_s12`, `vlm_imgflag_s12`), the agent faults for the variants that keep the frozen call
  (`vlm_agent_s0`), replays of the nuScenes counterexamples, and CARLA replays of the closed-loop violations.

## Results

Screening on the dev split: seed 0, the 7 sensor faults (C1–C4, L1, L2, L4), 42 runs, 1,701 frames, 0 frame errors in
either round (numbers from `report_figures/results_2026-11-21.json`). All methods see the same frames; every `_full`
method has the recovery controller and shield of Ours-full. Rows in the order of `docs/media/vlm_variants.png`.

| Method | F1 | P | R | Right sensor | False alarm | Unnecessary fallback | Missed fallback | VLM calls per frame |
|---|---|---|---|---|---|---|---|---|
| Checks-only (no VLM) | 0.727 | 0.876 | 0.621 | 0.948 | 0.086 | 0.044 | 0.075 | 0 |
| Ours-full (frozen) | 0.726 | 0.871 | 0.621 | 0.950 | 0.089 | 0.050 | 0.075 | 2.06 |
| Think, trusting the checks | 0.732 | 0.814 | 0.665 | 0.880 | 0.149 | 0.085 | 0.062 | 2.29 |
| Perceive → decide, trusting the checks (round 2) | 0.617 | 0.817 | 0.495 | 0.856 | 0.108 | 0.058 | 0.117 | 3.31 |
| VLM image flag → Ours (round 2) | 0.726 | 0.739 | 0.714 | 0.848 | 0.246 | 0.152 | 0.056 | 3.47 |
| VLM image flag → checks-only (round 2) | 0.723 | 0.727 | 0.719 | 0.843 | 0.264 | 0.137 | 0.054 | 1.00 |
| Think | 0.599 | 0.661 | 0.548 | 0.776 | 0.274 | 0.149 | 0.112 | 2.78 |
| Think, no checks | 0.613 | 0.529 | 0.729 | 0.474 | 0.633 | 0.328 | 0.071 | 3.30 |
| Perceive → decide | 0.516 | 0.800 | 0.381 | 0.856 | 0.093 | 0.050 | 0.165 | 3.31 |
| Perceive → decide, no checks | 0.403 | 0.664 | 0.289 | 0.638 | 0.143 | 0.072 | 0.190 | 2.72 |
| Per-sensor → decide | 0.508 | 0.757 | 0.382 | 0.888 | 0.120 | 0.069 | 0.157 | 9.34 |

Block-bootstrap 95% intervals over fault episodes (2,000 draws, paired) for Ours-full minus each variant:
- Every variant that adds the VLM's own first look to the deciding call, or is told the checks can be wrong, has a
  lower F1, and every interval excludes 0: per-sensor +0.218 [+0.173, +0.264], perceive +0.209 [+0.163, +0.258],
  perceive without checks +0.323 [+0.271, +0.372], perceive trusting the checks +0.109 [+0.070, +0.149], think +0.127
  [+0.077, +0.176], think without checks +0.113 [+0.072, +0.152].
- `think_trust` has the same F1 (difference −0.007 [−0.026, +0.012]) at a different operating point: fewer missed
  fallbacks (0.062 against 0.075; difference +0.012 [+0.004, +0.021]) but more false alarms (0.149 against 0.089)
  and unnecessary fallbacks (0.085 against 0.050), and it names the right sensor less often (0.880 against 0.950).
- The image-flag variants have the same F1 as well (image flag → Ours −0.001 [−0.028, +0.025]; image flag →
  checks-only +0.003 [−0.025, +0.028]), at a much more cautious operating point: fewer missed fallbacks (0.056
  against 0.075; +0.019 [+0.009, +0.029]) for almost three times the false alarms (0.246 against 0.089; −0.157
  [−0.194, −0.123]) and unnecessary fallbacks (0.152 against 0.050), and the right sensor on 0.848 of the alarms
  instead of 0.950.

F1 per fault (`docs/media/vlm_variants_per_fault.png`):

| Method | C1 | C2 | C3 | C4 | L1 | L2 | L4 |
|---|---|---|---|---|---|---|---|
| Checks-only | 0.74 | 0.49 | 0.48 | 0.74 | 0.95 | 0.49 | 0.97 |
| Ours-full | 0.74 | 0.49 | 0.47 | 0.74 | 0.95 | 0.49 | 0.97 |
| Think, trusting the checks | 0.84 | 0.54 | 0.53 | 0.73 | 0.92 | 0.42 | 0.96 |
| Perceive → decide, trusting the checks | 0.64 | 0.37 | 0.57 | 0.57 | 0.73 | 0.23 | 0.97 |
| VLM image flag → Ours | 0.70 | 0.57 | 0.64 | 0.74 | 0.88 | 0.55 | 0.93 |
| VLM image flag → checks-only | 0.70 | 0.57 | 0.64 | 0.74 | 0.87 | 0.55 | 0.91 |
| Think | 0.36 | 0.47 | 0.66 | 0.54 | 0.72 | 0.39 | 0.91 |
| Think, no checks | 0.55 | 0.59 | 0.63 | 0.54 | 0.61 | 0.60 | 0.74 |
| Perceive → decide | 0.07 | 0.18 | 0.58 | 0.64 | 0.58 | 0.16 | 0.97 |
| Perceive → decide, no checks | 0.21 | 0.19 | 0.56 | 0.19 | 0.20 | 0.20 | 0.91 |
| Per-sensor → decide | 0.07 | 0.09 | 0.64 | 0.38 | 0.80 | 0.11 | 0.90 |

What the VLM image check sees (round 2, `imgflag_checks` outputs of the screen; one answer per frame for all cameras):

| Frames | The VLM marks the right camera abnormal |
|---|---|
| C3 degraded image, high severity | 48 / 55 (87%) |
| C3, medium | 16 / 44 (36%) |
| C3, low | 2 / 21 |
| C2 misalignment (rotation or shift), all severities | 11 / 120 (9%) |
| C1 wrong view | 1 / 120 |
| C4 frozen image | 3 / 120 |
| Clean frames: any camera marked abnormal | 163 / 861 (19%): night scenes 122 / 287 (43%), day scenes 41 / 574 (7%) |

On clean frames its stated reasons are noise (76 frames), dark or black (65), blur (61), glare or washed out (56),
frozen (20) and tilt or shift (9). On the fault frames where the image flag raises an alarm that Ours-full does not
(78), the VLM names the faulty camera in 34 and another camera in 44: about half of the recall gain is a coincidence
of a false flag with a fault elsewhere (for example 14 alarms on L2 frames, a LiDAR fault, from camera flags).

### What this shows

- **Letting the VLM overrule the checks makes detection worse.** When the prompt says the checks can be wrong, the
  VLM explains correct flags away. Two logged examples (scene-0061): on a C1 frame the stage-1 note calls the
  front tile "plausible for front-facing view" (it shows the rear view) and the decision calls the low-match flag
  "a false positive"; on an L2 frame it sees "a dense, uniform point cloud" and calls the camera–LiDAR
  disagreement noise. A swapped viewpoint with plausible content and a calibration shift are visible to geometry,
  not to the eye.
- **The VLM's own first look hurts even with the frozen guidance.** `perceive_trust` keeps "the checks are reliable"
  and only adds the VLM's notes on the raw data before the final call; recall still drops from 0.621 to 0.495
  (missed fallbacks 0.117 against 0.075). So round 1's drop was not only the "checks can be wrong" sentence: the
  notes themselves ("plausible", "consistent") outweigh the flags.
- **Reasoning itself is not the problem.** With the frozen guidance kept, reasoning first (`think_trust`) gives the
  same F1 and moves the monitor toward caution: fewer missed fallbacks, more false alarms. Its largest per-fault
  gain is on C1 (0.84 against 0.74). The Thinking model in E5 showed the same pattern on the test split.
- **As a feature extractor, the VLM sees strong image degradation and little else.** Asked only about image quality,
  it marks 87% of high-severity degraded images (C3), which raises F1 on C3 from 0.47 to 0.64, but it almost never
  sees a rotated or shifted camera (9%), a swapped view (1%) or a frozen image (3%), and it calls 43% of clean night
  frames abnormal. Pooled F1 stays at 0.726: the C2/C3/L2 gains are paid for by false alarms everywhere else. The
  deciding VLM call adds nothing over the rules here (image flag → checks-only: F1 0.723 with one call per frame).
- **More calls are not better.** The per-sensor variant uses 9.3 calls per frame (4.5 times Ours-full) and has the
  second-lowest F1 (0.508; 0.07 on C1).

### Agent faults (round 2)

Dev split, seed 0, faults A1 (corrupted output), A2 (wrong answer: ok and fault swapped) and A3 (timeout), 729 frames,
0 frame errors (`outputs/eval/vlm_agent_s0`). Every sensor is healthy on these frames, so any fallback is unnecessary.
Recovery success = the final answer is valid and matches the model's fault-free answer (spec §5.2).

| Method | Recovery A1 | Recovery A2 | Recovery A3 | Recovery, all | Unnecessary fallback | VLM calls per frame |
|---|---|---|---|---|---|---|
| Ours-full (frozen) | 0.667 | 0.758 | 0.558 | 0.661 | 0.056 | 2.21 |
| Think, trusting the checks | 0.633 | 0.692 | 0.542 | 0.622 | 0.089 | 2.31 |
| VLM image flag → Ours | 0.667 | 0.758 | 0.558 | 0.661 | 0.154 | 3.52 |

The variants keep the re-ask recovery: the image-flag variant recovers exactly as often as Ours-full, because its
deciding call is the frozen one, and `think_trust` slightly less often. On these healthy-sensor frames the image flag
nearly triples the unnecessary fallbacks (0.154 against 0.056). `imgflag_checks` is not in the table: its only VLM call
is the image check, which the agent faults do not reach (deferred question D10).

### Replays of the falsification counterexamples

The 214 worst counterexamples of the checks (10 per fault and requirement from `nuscenes_cpu_v1` and
`nuscenes_gpu_v1`) replayed with `--confirm 10` (`outputs/falsify/nuscenes_confirm_{cpu,gpu}_v1[_<method>]/`;
`docs/media/vlm_replays.png`):

| Method | Still violated | R1 missed | R2 unnecessary | R3 recovery | R6 early restore |
|---|---|---|---|---|---|
| Ours-full | 212 / 214 | 47 / 47 | 70 / 70 | 46 / 46 | 49 / 51 |
| Think, trusting the checks | 179 / 214 | 38 / 47 | 68 / 70 | 42 / 46 | 31 / 51 |
| Think, no checks | 105 / 214 | 16 / 47 | 70 / 70 | 7 / 46 | 12 / 51 |

Without the check report, the VLM avoids most missed fallbacks, slow recoveries and early restores of these cases:
its decisions do not depend on the check evidence that failed there (a lost reference object, an intermittent
flag). It fixes none of the 70 unnecessary fallbacks, because it raises false alarms of its own. The counterexamples
were selected to break the checks, so any method whose errors are independent of the checks avoids many of them by
construction; the screening table above, on all frames, is the fair comparison. Together they suggest a monitor in
which the VLM's own judgment of image quality adds evidence that the checks lack, without a veto over them.

### CARLA replays with the VLM image flag (round 2)

Every closed-loop violation of the CARLA falsification came from a rotated front camera that the checks missed
(`docs/superpowers/runs/2026-11-20-falsification.md`). We replayed the 5 lowest-R4 scenarios per hazard and the
lowest-R5 scenario per hazard (12 scenarios, front camera rotated by 5.2°–11.3°) with `imgflag_checks_full`, the
checks-only rules plus the VLM image flag (`outputs/falsify/carla_v2_imgflag_{r4,r4_ped,r5}/`; table in
`report_figures/results_2026-11-21.md`).

| | checks-only (source runs) | image flag → checks-only |
|---|---|---|
| Front camera isolated during the fault | 0 of the fault steps | 0 / 153 fault steps |
| Clean steps with a sensor isolated | 1 | 65 / 308 (the rear camera, mostly "frozen") |
| R4 violated, the 10 lowest-R4 scenarios | 10 / 10 | 7 / 10 |
| R4 violated, the 2 lowest-R5 scenarios | 0 / 2 | 1 / 2 |
| Collisions | 1 / 12 | 1 / 12 (the same pedestrian scenario) |

The VLM never marked the rotated front camera. It marked the rear camera abnormal on 65 clean steps; its cached
answers say "frozen (identical to previous frame while scene moves)": a rear view changes little within 0.5 s,
especially when the car starts or slows. The three R4 "fixes" are coincidences: in each, a slow-down for a false
rear-camera alarm happened at the start of the front fault (speed ≤ 3.06 m/s), while the front camera stayed in use. In
the lead-car scenario where R4 newly fails, an earlier false-alarm slow-down left the car farther behind the lead car,
so the time-to-collision brake that had satisfied R4 for checks-only no longer fired within 3 s of the fault.
