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

Output of the deciding call (JSON schema): `analysis` (free reasoning, written first), `verdict` (ok, uncertain or
fault:<sensor>), `fault_type`, `confidence`, `evidence`. Stage-1 calls are not diagnosis calls: the agent faults
A1–A3 hit only the deciding call, as for Ours. The prompts were written once and not tuned (deferred question D3);
one smoke run on one dev scene checked only the output format.

## Protocol

- Dev split only (6 scenes). A new variant on the test split needs the user's approval (spec §5.4: tune on dev,
  freeze, run test once).
- Screening: all variants and the references (B0, checks-only, Ours-detect, Ours-full) in one run over the same
  frames: sensor faults C1–C4, L1, L2, L4, seed 0 (`outputs/eval/vlm_screen_s0`).
- Planned: the best variants with seeds 1–2 and all 10 faults. Changed under D7: since no variant beats Ours-full on
  seed 0, block-bootstrap intervals over fault episodes measure the gaps instead, and two variants replay the
  falsification counterexamples.

## Results

Screening on the dev split: seed 0, the 7 sensor faults (C1–C4, L1, L2, L4), 42 runs, 1,701 frames, 0 frame errors
(`outputs/eval/vlm_screen_s0`, `outputs/eval/vlm_screen_s0_trust`; numbers from
`report_figures/results_2026-11-21.json`). All methods see the same frames; every `_full` method has the recovery
controller and shield of Ours-full.

| Method | F1 | P | R | Right sensor | False alarm | Unnecessary fallback | Missed fallback | VLM calls per frame |
|---|---|---|---|---|---|---|---|---|
| Checks-only (no VLM) | 0.727 | 0.876 | 0.621 | 0.948 | 0.086 | 0.044 | 0.075 | 0 |
| Ours-full (frozen) | 0.726 | 0.871 | 0.621 | 0.950 | 0.089 | 0.050 | 0.075 | 2.06 |
| Think, trusting the checks | 0.732 | 0.814 | 0.665 | 0.880 | 0.149 | 0.085 | 0.062 | 2.29 |
| Think | 0.599 | 0.661 | 0.548 | 0.776 | 0.274 | 0.149 | 0.112 | 2.78 |
| Think, no checks | 0.613 | 0.529 | 0.729 | 0.474 | 0.633 | 0.328 | 0.071 | 3.30 |
| Perceive → decide | 0.516 | 0.800 | 0.381 | 0.856 | 0.093 | 0.050 | 0.165 | 3.31 |
| Perceive → decide, no checks | 0.403 | 0.664 | 0.289 | 0.638 | 0.143 | 0.072 | 0.190 | 2.72 |
| Per-sensor → decide | 0.508 | 0.757 | 0.382 | 0.888 | 0.120 | 0.069 | 0.157 | 9.34 |

Block-bootstrap 95% intervals over fault episodes (2,000 draws, paired) for Ours-full minus each variant:
- Every variant except `think_trust` has a lower F1, and every interval excludes 0: per-sensor +0.218
  [+0.173, +0.264], perceive +0.209 [+0.163, +0.258], perceive without checks +0.323 [+0.271, +0.372], think +0.127
  [+0.077, +0.176], think without checks +0.113 [+0.072, +0.152].
- `think_trust` has the same F1 (difference −0.007 [−0.026, +0.012]) at a different operating point: fewer missed
  fallbacks (0.062 against 0.075; difference +0.012 [+0.004, +0.021]) but more false alarms (0.149 against 0.089)
  and unnecessary fallbacks (0.085 against 0.050), and it names the right sensor less often (0.880 against 0.950).

F1 per fault (`docs/media/vlm_variants_per_fault.png`):

| Method | C1 | C2 | C3 | C4 | L1 | L2 | L4 |
|---|---|---|---|---|---|---|---|
| Checks-only | 0.74 | 0.49 | 0.48 | 0.74 | 0.95 | 0.49 | 0.97 |
| Ours-full | 0.74 | 0.49 | 0.47 | 0.74 | 0.95 | 0.49 | 0.97 |
| Think, trusting the checks | 0.84 | 0.54 | 0.53 | 0.73 | 0.92 | 0.42 | 0.96 |
| Think | 0.36 | 0.47 | 0.66 | 0.54 | 0.72 | 0.39 | 0.91 |
| Think, no checks | 0.55 | 0.59 | 0.63 | 0.54 | 0.61 | 0.60 | 0.74 |
| Perceive → decide | 0.07 | 0.18 | 0.58 | 0.64 | 0.58 | 0.16 | 0.97 |
| Perceive → decide, no checks | 0.21 | 0.19 | 0.56 | 0.19 | 0.20 | 0.20 | 0.91 |
| Per-sensor → decide | 0.07 | 0.09 | 0.64 | 0.38 | 0.80 | 0.11 | 0.90 |

### What this shows

- **Letting the VLM overrule the checks makes detection worse.** When the prompt says the checks can be wrong, the
  VLM explains correct flags away. Two logged examples (scene-0061): on a C1 frame the stage-1 note calls the
  front tile "plausible for front-facing view" (it shows the rear view) and the decision calls the low-match flag
  "a false positive"; on an L2 frame it sees "a dense, uniform point cloud" and calls the camera–LiDAR
  disagreement noise. A swapped viewpoint with plausible content and a calibration shift are visible to geometry,
  not to the eye.
- **Reasoning itself is not the problem.** With the frozen guidance kept, reasoning first (`think_trust`) gives the
  same F1 and moves the monitor toward caution: fewer missed fallbacks, more false alarms. Its largest per-fault
  gain is on C1 (0.84 against 0.74).
- **The VLM's own perception helps on image-quality faults.** Without any check report, `think_nochecks` reaches
  F1 0.59 on misalignment (C2) and 0.63 on degraded images (C3), above the checks (0.49, 0.48), and every
  image-centric variant beats the checks on C3. These two faults have the lowest test F1 of Ours-full (C2 0.47,
  C3 0.45), and misalignment caused every closed-loop violation the falsification found. The price is a 63%
  false-alarm rate and weak LiDAR and viewpoint diagnosis.
- **More calls are not better.** The per-sensor variant uses 9.3 calls per frame (4.5 times Ours-full) and has the
  second-lowest F1 (0.508; 0.07 on C1).

A direction that follows from this, not tested here: keep the checks in charge and ask the VLM only about image
quality (C2, C3), where it sees more than the geometry, with its answer as one more flag rather than a veto.

Not done (deferred question D7): seeds 1–2 (no variant beats Ours-full on seed 0), the agent faults A1–A3 for the
variants, and the test split (a new variant needs the user's approval under spec §5.4).

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
