# Related work survey (2025–2026)

- Date: 2026-11-15
- Goal: check whether the problem RecoverDrive addresses has already been solved, and judge how large our
  contribution is.
- Method: 4 alphaXiv searches and 1 web search. The 6 closest papers were read in detail; the rest only by title
  and abstract.
- Limits: the search is not exhaustive. Papers from after June 2026 are known only through this search, and work
  that is not on arXiv (for example, conference-only papers) may be missing. A "first to ..." claim in the slides
  or report should read "among the work we found".

## 1. What RecoverDrive is

RecoverDrive is a monitor that runs beside the driving stack; it does not drive.

1. **Detect:** deterministic camera–LiDAR cross-checks in Python (2D/3D match rate, point counts, sectors).
2. **Diagnose:** Qwen3-VL-8B reads the check report and the images and names the faulty sensor.
3. **Recover:** re-ask, 3-sample vote, re-observe on the next frame, and isolate the faulty sensor (degraded mode).
4. **Verify:** check that the recovery worked.
5. **Fallback:** a rule-based shield with no LLM picks `continue < slow_down < stop`.

The VLM itself is also treated as a component that can fail: A1 corrupted output, A2 wrong answer, A3 timeout.

Faults: C1–C4 (camera), L1/L2/L4 (LiDAR), A1–A3 (agent). L3 (ghost objects) is out of scope.

## 2. Papers read in detail

| Paper | Date / institution | What it does | Difference from ours | Closeness |
|---|---|---|---|---|
| [PRAM-R](https://arxiv.org/abs/2603.04222) | 2026-03, TUM | Qwen3-VL-8B as a modality router. Input: per-sensor quality indicators (camera brightness, contrast, edge density; LiDAR point density, noise, reflectivity; radar) plus scene context. Output: a reliability score and an on/off decision per sensor. Hysteresis and EMA keep switching stable; fast/slow dual loop; hierarchical memory. | Goal is lower compute: on 10 nuScenes scenes it switches off 6.22 % of modalities, with ADE/FDE slightly worse than all sensors on (1.124/2.182 vs 1.013/2.026; the paper calls this comparable). Only synthetic stress tests (gradual degradation, abrupt failure, high-frequency noise); no injected faults with detection F1. No recovery, no verify, no closed loop, no agent faults, latency not measured. | **Highest** (same model; both use a VLM to decide which sensor to trust) |
| [AutoAgent0](https://arxiv.org/abs/2610.18866) | 2026-10, UCLA / CMU / UT Dallas / Stanford | Agentic runtime around existing driving policies: an independent detection module builds scene context → a rule-based verifier checks proposed trajectories → when no trajectory passes, an agent builds recovery maneuvers from action primitives and refines them with verifier feedback. Closed-loop evaluation on Fail2Drive and NavSafe-∞. | Same architecture shape as ours (independent checks → rule-based verification → recovery), but it targets planning errors and long-tail scenarios, not sensor faults. (Abstract and introduction read; results not read.) | High (architecture) |
| [CRUISE](https://arxiv.org/abs/2608.09202) | 2026-08, UC Irvine / US Army DEVCOM GVSC | AnomalyGPT, LoRA-fine-tuned on driving data, produces pixel-level uncertainty maps that guide camera/LiDAR fusion, with a cross-modal adaptation module. Under sensor corruption: +2.99 % AP on aiMotive 3D detection; +4.25 % / +1.56 % mIoU on DeLiVER / MFNet. | Robust fusion, not monitoring: no detect / diagnose / recover / fallback and no safety action. (Abstract and introduction read.) | Medium |
| [CMPG: When Camera and LiDAR Disagree](https://arxiv.org/abs/2610.16906) | 2026-10, Queen Mary University of London | Labels every ground-truth object by which modality detected it (StreamPETR vs TransFusion-L), about 0.4M objects across nuScenes, KITTI and Waymo. Analyses the labels with distance-stratified Shapley attribution and injects the resulting Modality Reliability Map into fusion detectors (up to +1.3 mAP). | Offline analysis of recall only, not runtime fault diagnosis. **Worth noting:** camera/LiDAR agreement depends on object class and distance (traffic cones are camera-only 16.8 % of the time, trucks 2.9 %; camera-only detections peak at 10–30 m). Our `match_rate` threshold is a single global value and ignores this. | Medium |
| [EvoEye](https://arxiv.org/abs/2607.03755) | 2026-07, SMU / U Ottawa / Lero | Runtime monitor that predicts an upcoming collision from internal ADS signals (Apollo + CARLA) and improves itself by searching for scenarios it gets wrong. 2.49 ms per inference on CPU. | Different problem: it predicts collisions; it does not say which sensor failed. Useful evaluation ideas: F1 at a fixed false-positive rate and warning lead time. | Low |
| [Counterfactual FDI for Robot Perception](https://arxiv.org/abs/2509.18460) | 2025-09, Caltech | Reliability tests plus a structural causal model; each fault hypothesis is evaluated by counterfactual reasoning. Passive version: likelihood-ratio test and Bayesian weight updates. Active version: MCTS-UCB picks the control input that best separates the fault hypotheses (Effective Information). Shown on a single-camera space robot. | No VLM and not driving. **Worth noting:** it shows that fixed thresholds are sensitive to scene context and cause false alarms, a weakness we share. Its active FDI is a more principled version of our re-observe step. | Medium (concept) |
| [DriveAgent](https://arxiv.org/abs/2505.02123) | 2025-05 | Multi-agent LLM text chain that compares LiDAR and visual text descriptions to flag sensor abnormalities. | Diagnosis only, no recovery; the released code does not run. Our B1–B3 re-implement it. | Starting point |

## 3. Earlier foundations (found in reference lists, not read)

- **Antonante, Nilsen, Carlone,** "Monitoring of perception systems: Deterministic, probabilistic, and
  learning-based fault detection and identification," *Artificial Intelligence* 325, 2023. Fault detection and
  isolation (FDI) from consistency checks between perception modules (diagnostic graphs). The closest precursor of
  our consistency checks.
- **Sinha, Schmerling, Pavone,** "Closing the loop on runtime monitors with fallback-safe MPC," CDC 2023. A
  runtime monitor triggers a fallback, and the planner keeps that fallback feasible. A precursor of our shield.
- **Goelles et al.,** "Fault detection, isolation, identification and recovery (FDIIR) methods for automotive
  perception sensors," *Sensors*, 2020. Survey of FDIIR methods for automotive sensors, including LiDAR.
- **Simplex architecture (general idea):** when a complex controller misbehaves, switch to a simple, verified
  backup. Our isolate → degraded mode → shield chain follows this idea.

## 4. Related papers seen by title or abstract only

| Paper | Relation to ours |
|---|---|
| [Drive the Thoughts](https://arxiv.org/abs/2608.29583) (2026-08, UVA) | Runtime monitoring of whether a VLA's reasoning matches its trajectory; related to our agent faults (A2) |
| [Towards Reliable VLMs for Autonomous Driving](https://arxiv.org/abs/2610.01531) (2026-10, TUM / Mercedes-Benz / UMass) | Survey of VLM reliability in driving; useful to cite in related work |
| [MiCaDe](https://arxiv.org/abs/2611.04912) (2026-11, Politecnico di Milano) | Online detection of LiDAR–camera miscalibration; related to L2 / C2 |
| [DepthVision](https://arxiv.org/abs/2509.07463) | Synthesizes RGB from LiDAR so a VLM keeps working when the camera degrades. A different recovery route (fill in data instead of isolating) |
| [Sensor-Level Fault Diagnosis with LLMs](https://arxiv.org/abs/2608.05921) | LLMs classify sensor faults during hardware-in-the-loop testing (ISO 26262 workflow), not on the vehicle; notes that prior work mostly covers single faults from small fault lists |
| [Modular Fault Diagnosis Framework](https://arxiv.org/abs/2411.09643) (2024-11, FZI / KIT) | Modular fault diagnosis for driving stacks, no VLM |
| [Testing the Fault-Tolerance of Multi-Sensor Fusion](https://arxiv.org/abs/2504.13420) (2025-04) | Tests how fusion stacks tolerate sensor faults (testing, not a monitor) |
| [Reliability-Driven LiDAR-Camera Fusion](https://arxiv.org/abs/2502.01856) (2025-02) | Weights fusion by per-sensor reliability |
| [Synergistic Simplex](https://arxiv.org/abs/2605.08190), [Mission-Level Runtime Assurance](https://arxiv.org/abs/2606.06996) (2026) | Runtime assurance / Simplex-style methods |
| [VLM-CASE](https://arxiv.org/abs/2607.05180) (2026-07, Penn State) | A VLM adapts the safety envelope to weather and context |

## 5. The gap

Every component exists somewhere, but no paper we found does all of the following together:

1. A VLM names **which** camera or LiDAR failed, using deterministic cross-sensor evidence as input.
2. Explicit recovery (vote, re-observe, isolate) followed by verification.
3. A safety shield with no LLM, measured by unnecessary and missed fallbacks.
4. The VLM's own failures (A1–A3) treated as faults to defend against.
5. Seeded evaluation over many injected faults (10 fault types × 3 seeds) plus a closed loop in CARLA.

## 6. How large is the contribution

**Strengths**

- **The combination is new.** No prior work found covers all five points of §5.
- **A clear empirical finding:** evidence matters more than model size. Removing the checks drops F1 from 0.77 to
  0.50; 4B is close to 8B (0.75 vs 0.77); the Thinking model gives the same F1 at about ten times the latency.
  PRAM-R and the others do not measure this.
- **Stricter evaluation than comparable work:** baselines on the same model, a frozen test split run once,
  3 seeds, fallback metrics and a closed loop. PRAM-R reports efficiency only; CRUISE and CMPG report mAP/mIoU only.

**Weaknesses**

- **No new component.** Consistency checks follow Antonante 2023, voting is self-consistency, isolate + fallback is
  the Simplex / fallback-safe MPC idea. The contribution is integration and evaluation, not a new algorithm.
- **No checks-only + shield baseline,** so we cannot say how much the VLM adds over the rules.
- **Idealized setup:** ground-truth LiDAR boxes (oracle detector), synthetic faults, 4 test scenes, E3 with C1 only
  and 12 runs per cell. 5 s per frame (2.6 s with the VLM gate) is far from real time.
- **Fixed thresholds.** CMPG and the Caltech paper both show that sensor agreement varies with distance, class and
  scene; our thresholds are global constants.

**Verdict**

- **As a course project:** solid and clearly above the usual level.
- **As a paper:** roughly workshop level today. A main-conference submission would need at least a real LiDAR
  detector, real fault data, and the checks-only baseline.

## 7. Suggestions for the slides and report

- **Related-work slide:** contrast with PRAM-R (same model, but routing only and efficiency metrics only) and
  AutoAgent0 (similar architecture, but planning errors instead of sensor faults). DriveAgent is the starting point
  and the source of the baselines.
- **Wording:** "among the work we found, no study combines ...", not "the first".
- **Limitations:** state the fixed-threshold issue up front and cite CMPG and the Caltech paper for future
  directions (distance/class-aware thresholds, active re-observation).
- **Future work:** PRAM-R's fast/slow dual loop for latency; DepthVision's fill-in approach as an alternative
  recovery.
- **Extra experiments:** the checks-only baseline is a new baseline, so running it on the test split does not break
  the run-once rule. A change to Ours itself (for example, distance-aware thresholds) must be tuned on dev only and
  reported as post-hoc.
