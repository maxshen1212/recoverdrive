# Future work

The current results come from a course project: the full pipeline, six methods, ten fault types, a closed loop in
CARLA, latency and ablations, all on nuScenes-mini (see the README). The next goal is a paper. This file lists what
is missing and plans it in three steps, from cheap to expensive.

- Step 1 makes the current report complete.
- Step 2 makes the comparison with the baselines fair.
- Step 3 adds the experiments a paper needs.

Codes such as E1 or C1 are explained in the glossary at the end.

## Where the current results are weak

- **Oracle LiDAR detector.** LiDAR objects are the ground-truth boxes with at least 5 points inside, not the output
  of a detector. This makes the camera–LiDAR cross-check cleaner than it would be on a real car.
- **Small test set.** The test split has 4 scenes. Two of them (scene-0757 and scene-1094) are in the nuScenes
  *train* split.
- **Unequal tuning.** Our prompt and thresholds were tuned on the dev scenes; the DriveAgent-style baselines were not.
- **The VLM adds little over the rules.** A rules-only (checks-only) baseline, added in Step 1, matches Ours-full on
  sensor faults (see Step 1).
- **Fixed thresholds.** One global threshold per check, although camera/LiDAR agreement depends on distance and
  object class (see `docs/related-work.md`).
- **Small closed loop.** E3 uses one fault type (C1), four routes and 12 runs per cell.

## Step 1: make the report complete

**Done (2026-11-15).** Results in `report_figures/tables.md` (S1–S5), the report Doc, and the provenance file.
- Checks-only baseline: on sensor faults it matches Ours-full (F1 0.714 for both, difference 0.000 [−0.001, 0.002];
  same false alarms; sensor accuracy 0.933 vs 0.918). Ours-full's higher pooled F1 (0.77 vs 0.60) is the agent faults,
  which a monitor without a VLM cannot have. **On this benchmark the VLM confirms the flags and adds no detection or
  fallback quality of its own.**
- Bootstrap: every Ours-full − B1/B2/B3 interval excludes 0 (F1 +0.15 to +0.18; false alarm −0.23 to −0.73).
- Easy vs. hard: on hard faults B3 names the right sensor 23 % of the time, Ours-full 79 %, checks-only 88 %.
- 4B + VLM gate: the same results as 8B + VLM gate (F1 0.601, unnecessary 0.065, missed 0.053) at 1.9 s per frame
  (8B 3.6 s, 8B + gate 2.4 s, test-weighted).

**What this changes for the paper.** The finding "evidence matters more than model size" now reads "the
deterministic evidence does the detection; the VLM defers to it". Before investing in Step 2 and Step 3, decide the
claim: either present the monitor (checks + recovery + shield) as the contribution with the VLM as an optional
explainer, or find faults the checks cannot see where the VLM helps (subtle image faults, ghost objects (L3), real
adverse weather), which Step 3's new data would have to show. Step 2's text-chain baselines with the check numbers
will likely land near checks-only for the same reason.

**Update (2026-11-21): letting the VLM reason more did not help on sensor faults.** Nine VLM-centric variants were
screened on the dev split in two rounds (`docs/vlm-variants.md`). When the prompt says the checks can be wrong, or the
VLM first writes its own notes on the raw data, it explains correct flags away and F1 drops by 0.11–0.32; with the
frozen guidance, reasoning first gives the same F1 with fewer missed and more unnecessary fallbacks. Round 2 tested the
suggested next step, the VLM as an image-quality flag next to the checks: it catches 87% of strongly degraded images
(C3 F1 0.47 → 0.64) but only 9% of rotated or shifted cameras, and it flags 43% of clean night frames, so F1 stays the
same with three times the false alarms; in CARLA it never caught the rotated front camera behind the closed-loop
violations and called the slow-changing rear view "frozen". So the second option above, faults the checks cannot see,
still needs a VLM that sees image geometry better or is calibrated against false alarms (for example a few labeled
night frames, or a stricter abnormal threshold tuned on dev).

Cheap and mostly offline: these reuse the logs of the existing runs.

| Item | What to do | Question it answers |
|---|---|---|
| Checks-only baseline | A new method with no VLM: the deterministic flags isolate sensors directly, then the shield acts. | How much does the VLM add over the rules? |
| Easy vs. hard faults | Report easy faults (L4, high-severity L1, A1, A3) separately from the rest. | Is the gap between methods real on the hard faults? |
| Confidence intervals | Bootstrap over fault episodes (4 scenes are too few to bootstrap over scenes); compare Ours-full with B3. | Is the gap stable or luck? |
| 4B model + VLM gate | Replay the gate on the logged 4B answers; time one short 4B run. | Can the monitor get close to real time (about 1–2 s per frame)? |

Also done (2026-11-15): the E3 and E4 scripts now record the git commit, as `run_eval.py` does.

## Step 2: make the comparison fair

Medium cost: new VLM runs.

| Item | What to do | Question it answers |
|---|---|---|
| Small-classifier baseline | Train a small model (for example, gradient boosting) on the check-report numbers of the dev scenes to name the faulty sensor. | Why not a small learned model instead of a VLM? |
| B1–B3 with check numbers | Give the DriveAgent-style baselines the same check report as Ours. | How much comes from the evidence, and how much from prompt engineering? |
| B1–B3 prompts tuned | Tune the baseline prompts on the dev scenes with the same budget as Ours (4 versions). | Answers "only Ours was tuned". |
| PRAM-R-style baseline (**done 2026-11-22**) | The VLM gets per-sensor quality indicators (brightness, point density) instead of cross-sensor checks. Test: F1 0.344 vs 0.765 for Ours-full at the same false-alarm rate; recall 0 on C1, C2, C4 and L2 (report Doc, Section 6.3). | How do we compare with the closest related work? |
| Larger E3 | Add a second front-camera fault (C3: an all-black front camera during the hazard) and double the runs to 24 per cell. | Does the closed-loop result hold beyond one fault and 12 runs? |

Detailed implementation plans: `docs/superpowers/plans/2026-11-15-step1-report-completeness.md` (Step 1) and
`docs/superpowers/plans/2026-11-15-step2-fair-comparison.md` (Step 2).

## Step 3: paper-level experiments

High cost. These change the data and the LiDAR side, so they need **new dev and test scenes** (see below).

| Item | What to do | Question it answers |
|---|---|---|
| New dev and test scenes | Pick both from the nuScenes *val* split; use 30–50 unseen scenes for testing, including night and rain. The current numbers become preliminary results. | Do the results hold on more scenes? Does the monitor raise false alarms in hard but fault-free conditions? |
| Real LiDAR detector | Replace the ground-truth boxes with the output of a pretrained 3D detector (CenterPoint). | Does the method still work without the oracle? |
| LiDAR noise curve | Add position noise, missed detections and false boxes to the ground-truth boxes at 0, 0.5×, 1× and 2× strength. | How fast does the monitor degrade as the detector gets worse? |
| Cross-model study | Qwen3-VL 4B / 8B / 32B and a second model family. | Does "evidence matters more than model size" hold across models? |
| Ours v2 | Distance- and class-aware thresholds and multi-frame smoothing, tuned on the new dev scenes only. | Fewer false alarms from fixed thresholds. |
| Larger closed loop | More fault types, routes and towns in CARLA. | Does the closed-loop result generalize? |

### A real LiDAR detector needs no LiDAR hardware

The nuScenes point clouds are already real: a roof-mounted 32-beam LiDAR recorded them on public roads. What is
missing is only the detection step. Today the LiDAR objects are the annotated boxes. The plan is to run a pretrained
3D detector (CenterPoint) on the GPU: it reads the recorded point clouds and outputs 3D boxes, the same way SAM 3
reads the recorded camera images on the camera side.

- Run it in Docker. The container brings its own CUDA toolkit, which avoids the system `nvcc` (CUDA 11.5).
- Only LiDAR faults (L1, L2, L4) change the point cloud, so only those frames need a new detection pass. Camera
  faults reuse the detections of the clean frame.
- Expected cost: about 1–3 days of setup and little GPU time.
- It also shows something the oracle cannot: after a calibration shift (L2), a real detector outputs boxes that are
  all shifted, which may be misread as a camera fault.

### Why the new scenes must come from nuScenes val

Public CenterPoint weights are trained on the nuScenes *train* split. Two of the current test scenes and four of the
six dev scenes are in that split, so the detector has seen them and would look better there than on new data.
Thresholds calibrated on such scenes would also be too strict for unseen scenes. Picking both dev and test scenes
from *val* (150 scenes) avoids this.

### Fallback if the detector is not ready

The LiDAR noise curve is cheap and offline. If CenterPoint cannot be set up in time, the noise curve plus a clear
limitation is the fallback, and a workshop is the more realistic venue.

## Rules that keep the evaluation honest

- Steps 1 and 2 still use the current test split. They only add baselines and offline analyses and do not change
  Ours, so the "run the test split once" rule still holds.
- Step 3 uses the new scenes. Any change to Ours is tuned on the new dev scenes only, frozen, and then run once on
  the new test scenes.
- Every result keeps a record of the commit that produced it (see `docs/superpowers/runs/results-provenance.md`
  and the `results/*` tags).

## Timeline and target

| When | What |
|---|---|
| Nov 15 – Nov 24, 2026 | Step 1 (done Nov 15); course requirements and VLM-centric variants (done Nov 20–21); PRAM-R-style baseline (done Nov 22); slides and demo (done) |
| Dec 1 – Dec 10, 2026 | Course presentation (Dec 1) and final report (Dec 10); start Step 2 with the classifier and B1–B3 with check numbers |
| Dec 11, 2026 – Jan 31, 2027 | Rest of Step 2; set up the new scenes and CenterPoint |
| Feb 1 – Feb 15, 2027 | Step 3 experiments; run the new test split once |
| Feb 15 – Mar 1, 2027 | Write the paper |

Target: IROS 2027 (submission deadline about March 1, 2027; check the official site). RA-L, which accepts
submissions at any time, is the backup.

**Paper framing.** The main claim: grounding a VLM in deterministic cross-sensor evidence matters more than making
the model bigger. The system is the means to show it. The small-classifier baseline, the real LiDAR detector and the
cross-model study are the experiments this claim depends on most.

## Glossary

| Group | Codes |
|---|---|
| Experiments | E0 pilot, E1 detection accuracy, E2 fallback quality, E3 CARLA closed loop, E4 latency, E5 ablation |
| Methods | B0 no monitor; B1 / B2 / B3 DriveAgent-style text chain (zero-shot / chain-of-thought / self-refine); Ours-detect (first VLM answer, no recovery); Ours-full (with recovery); checks-only (Ours-full without the VLM); PRAM-R-style (the VLM scores each sensor from per-sensor quality indicators) |
| Camera faults | C1 front camera shows the rear view, C2 misalignment, C3 degraded image, C4 frozen image |
| LiDAR faults | L1 point dropout, L2 calibration shift, L3 ghost objects (not evaluated), L4 total failure |
| Agent faults | A1 corrupted output, A2 wrong answer, A3 timeout |
