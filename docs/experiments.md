# Experiments: questions and variables

One table for every experiment of the project. **Changed** = the independent variables (what the experiment varies);
**fixed** = the controlled settings; **measured** = the dependent variables. Results and the commit of each run are in
`docs/superpowers/runs/results-provenance.md`.

Common to all nuScenes experiments unless stated otherwise: nuScenes v1.0-mini, scene-level split (6 dev scenes for
tuning, 4 test scenes run once from the frozen commit `5d5283c`), keyframes at 2 Hz, SAM 3 camera detector (6 classes,
score ≥ 0.5), LiDAR objects = annotated boxes with ≥ 5 points within 50 m (assumption A4), Qwen3-VL-8B-Instruct on
vLLM, the frozen thresholds of `configs/eval.yaml`, the same recovery controller (re-ask ≤ 2, 3-sample vote at T = 0.7,
vote threshold 0.6) and the same rule-based shield (continue < slow_down < stop). Fault episodes: seeded, 2–6 frames,
fault ratio 0.5, at most one fault at a time.

## Detection, fallback and system experiments

| Experiment | Question | Changed | Fixed | Measured | Scale |
|---|---|---|---|---|---|
| **E1/E2** main comparison | Does the monitor detect faults and choose a safe action? | method (B0, B1–B3, Ours-detect, Ours-full); fault type (C1–C4, L1, L2, L4, A1–A3); severity (sampled per episode) | test split; frozen config; seeds 0–2 | E1: precision, recall, F1, right-sensor rate, false-alarm rate. E2: recovery success, episodes recovered, frames to recover, unnecessary fallback, missed fallback | 4 scenes × 10 faults × 3 seeds × 6 methods = 120 runs, 4,830 frames |
| Checks-only baseline (Step 1) | How much does the VLM add over the deterministic checks? | method (checks-only vs. Ours-full) | as E1/E2 | as E1/E2; bootstrap intervals over fault episodes | test, 3 seeds |
| Easy vs. hard faults (Step 1) | Is the gap between methods real on the hard faults? | fault difficulty (easy: L4, high-severity L1, A1, A3; hard: the rest) | as E1/E2 | recall, right-sensor rate, missed fallback, recovery | test, 3 seeds |
| **E3** CARLA closed loop | Does the monitor prevent collisions when a front-camera fault hits before a hazard? | method (B0, B1, Ours-full); condition (clean, C1); hazard (pedestrian crossing, lead car braking); route (4) | CARLA 0.9.16 Town01, cruise 30 km/h, slow-down 10 km/h, perception every 0.5 s, TTC threshold 2 s | collision rate, route completion, unnecessary stops | 4 routes × 2 conditions × 3 seeds × 3 methods = 72 runs |
| **E4** latency | How long does one monitoring step take, and can it get near real time? | model size (8B, 4B); VLM gate (call the VLM only on flagged frames) | dev scenes, faults C1 and L1, one method at a time, fresh cache | seconds per frame per component (SAM 3, checks, render, VLM calls, recovery) | dev, seed 0 |
| **E5** ablation | Which parts matter? | camera detector (SAM 3 vs. Qwen grounding); check report (with/without); recovery parts (vote, re-observe, isolate; offline replay); model (4B, 8B, 8B-Thinking) | test split; everything else as Ours-full | E1/E2 metrics | test, 3 seeds (Thinking: seed 0) |

## Verification experiments (course requirements)

| Experiment | Question | Changed | Fixed | Measured | Scale |
|---|---|---|---|---|---|
| Requirements on logs | Do the logged runs satisfy R1–R6? | requirement (R1, R2, R3, R6 on nuScenes; R4, R5 in CARLA); method | the logged E1/E2 and E3 runs | violations per requirement, STL robustness, Clopper–Pearson bounds per fault type | all test and E3 logs |
| Formal proofs | Is the safety logic correct for all inputs? | — (exhaustive) | the real Python shield and recovery controller | shield properties S1–S7 over every input combination; recovery properties C1–C4 by explicit-state model checking | all reachable states |
| **Falsification, nuScenes** | Which fault settings break a requirement, and does guided search find them faster than random injection? | sampler (random vs. guided TPE); fault (C1–C4, L1, L2, L4); requirement (R1, R2, R3, R6). Search variables: dev scene, start frame, duration (2–6), target camera, subtype, continuous severity level | budget 60 trials per campaign, seed 0, checks-only (no VLM), dev scenes only | violating trials, lowest robustness, trial of the first violation, failure types, triage cause | 56 campaigns, 3,360 trials |
| Replays of the counterexamples | Does a VLM method fix the cases where the checks fail? | method (Ours-full; the best VLM-centric variant) | the 10 worst violating trials per (fault, requirement): 214 counterexamples | share still violated | 214 replays per method |
| **Falsification, CARLA** | Which scenarios make the closed loop violate R4/R5? | sampler (random vs. TPE); hazard (pedestrian, lead car braking). Search variables: route, trigger offset (±5 m), walker speed (1.0–2.5 m/s) or lead braking time (1–4 s), fault type (C1, black front camera, rotated front camera, LiDAR dropout in the front sector), fault level, fault window (starts 0–30 m before the hazard trigger, 10–60 m long), seed (0–99) | budget 40 runs per campaign, checks-only, Town01 | R5 robustness (TTC braking, gap ≥ 0), R4 robustness (slow-down after a front fault), collisions | 4 campaigns, 160 runs; replay of the 5 worst per hazard with Ours-full |

## VLM-centric variants (2026-11-21)

| Experiment | Question | Changed | Fixed | Measured | Scale |
|---|---|---|---|---|---|
| Variant screening | Does the VLM help more when it reasons before it answers and looks at the raw data itself, with less hand-made evidence? | diagnosis agent, round 1: Ours (frozen), think, think without checks, perceive → decide, perceive → decide without checks, per-sensor → decide, think trusting the checks; round 2: perceive → decide trusting the checks, VLM image flag → Ours, VLM image flag → checks-only (see `docs/vlm-variants.md`) | dev split (6 scenes), sensor faults C1–C4, L1, L2, L4, seed 0; same recovery and shield; same model | E1/E2 metrics, VLM calls per frame; block-bootstrap intervals over fault episodes against Ours-full | 42 runs per round, all methods on the same 1,701 frames |
| Seeds 1–2 | Is the result stable across fault draws? | method: B0, checks-only, Ours-detect/full, think trusting the checks, VLM image flag → Ours | dev split, sensor faults C1–C4, L1, L2, L4, seeds 0–2 | E1/E2 metrics, mean ± std over seeds | 126 runs, 5,103 frames per method |
| Agent faults | Do the variants keep the re-ask recovery when the VLM itself fails? | method: Ours-detect/full, think trusting the checks, VLM image flag → Ours; fault A1 (invalid output), A2 (flipped verdict), A3 (timeout) | dev split, seed 0 | recovery success, unnecessary and missed fallback | 18 runs |
| Replays of the counterexamples | Does a variant fix the cases where the checks fail? | method: Ours-full, think trusting the checks, think without checks | the 214 worst nuScenes counterexamples of the checks | share still violated per requirement | 214 replays per method |
| CARLA replays with the VLM image flag | Does the VLM image flag catch the rotated front camera that the checks miss in closed loop? | method: checks-only vs. VLM image flag → checks-only | the 5 lowest-R4 and the lowest-R5 scenario per hazard from the CARLA falsification | R4, R5, collision | 12 runs |
