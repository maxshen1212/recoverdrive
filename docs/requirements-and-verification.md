# Requirements, verification and assurance (report text)

Source text for the report sections that answer the course requirements in `REQUIREMENT.md`: formal requirements,
test generation by falsification, formal reasoning, and an assurance strategy. Every number comes from a generated
file:
- `report_figures/requirements.md` (`scripts/check_requirements.py`)
- `report_figures/formal.md` (`scripts/verify_logic.py`)
- `outputs/falsify/*/summary.md` (`scripts/falsify_nuscenes.py`)

Items marked **[Phase B]** wait for free GPUs.

## Requirements (after "Assumptions and Constraints")

We state the requirements in Signal Temporal Logic (STL) over discrete steps of 0.5 s (nuScenes keyframes at 2 Hz;
CARLA runs perception every 0.5 s). The signals are:
- `a`: the rank of the monitor's action (continue 0, slow down 1, stop 2).
- `need`, `allow`: the ranks of the least and the most cautious acceptable action for the true situation (§5.2).
- `fault`, `fault_s`: whether a fault is active, overall and on sensor s.
- `iso_s`: whether sensor s is isolated.
- In CARLA only: speed, time-to-collision (TTC), and the gap to the hazard.

| ID | Requirement | STL |
|---|---|---|
| R1 | No missed fallback: when the situation needs caution, the action is cautious enough within one step | G( need ≥ 1 → F[0,1] (a ≥ need) ) |
| R2 | No unnecessary fallback | G( a ≤ allow ) |
| R3 | Recovery: back to "continue" within 2 steps after a fault ends | G( fault_end → F[0,2] (a = continue ∨ fault) ) |
| R4 | Front-camera fault: slow to ≤ 3.06 m/s within 3 s (CARLA) | G( front_onset → F[0,6] (speed ≤ 3.06) ) |
| R5 | TTC below 2 s: stop within one step; no collision (CARLA) | G( (TTC < 2 → F[0,1] (a_applied = stop)) ∧ gap ≥ 0 ) |
| R6 | No sensor is restored while it is still faulty | G( (fault_s ∧ iso_s) → X (iso_s ∨ ¬fault_s) ), for every sensor s |

**Assumptions:**
- A1: at most one sensor fault at a time.
- A2: calibration and timestamps are correct.
- A3: a camera can be cross-checked only when at least two LiDAR-confirmed objects (≥ 5 points, within 50 m) project
  into it.
- A4: LiDAR objects come from ground-truth boxes, an oracle detector.

The requirements go beyond reach-avoid:
- R1 and R2 together bound the action from both sides, so the monitor must be neither too permissive nor too cautious.
- R1, R4 and R5 are bounded-response properties.
- R3 is a recovery (liveness) property, and R6 a stability property of the isolation logic.

We use the quantitative semantics of STL. Robustness is positive when a requirement holds and negative when it is
violated; its size says by how much. Integer and Boolean signals are compared at half steps, so a violation by one
action level has robustness −0.5. Obligations whose time window runs past the end of a log are dropped rather than
counted as violations.

## Verification and falsification (new subsection of §6)

### Formal verification of the safety logic

The shield and the recovery controller have finite input domains once the sensors are fixed. We therefore verify them
by exhaustive enumeration: every case is run through the real Python functions. There is no hand-written model, so
nothing can drift between model and code.

| Property | Statement | Cases | Result |
|---|---|---|---|
| S1 | Assume-guarantee: a single fault that is detected and isolated correctly leads to an acceptable action | 77 | holds |
| S2 | Isolating more sensors never makes the action less cautious | 19,683 | holds |
| S3 | With no front perception (front camera and LiDAR both isolated), the shield stops | 64 | holds |
| S4 | Unresolved leads to stop; healthy leads to continue | 512 | holds |
| S5 | TTC < 2 s leads to stop, whatever the monitor chose | 15 | holds |
| S6a | Without correct detection: any outcome on a clean frame is acceptable | 257 | fails (false alarm) |
| S6b | Without correct detection: "healthy" on a fault frame is acceptable | 1 | fails (missed C1 → continue) |
| S7 | On a fault frame, isolating any single sensor, even the wrong one, gives an acceptable action | 608 | holds |

S1 and S6 split system safety into two parts:
- **The shield, proven correct (S1):** if the monitor detects the fault, the action is acceptable.
- **The monitor's detection, measured:** without correct detection, the guarantee breaks, both on false alarms (S6a)
  and on missed faults (S6b). The counterexample of S6b, a front camera showing the rear view reported as healthy, is
  exactly the failure that caused the collisions of the unmonitored car in CARLA.

S7 adds a less obvious result: naming the wrong sensor is never unsafe on its own. Any single isolation leads to
slowing down, and slowing down is acceptable on every fault frame. Detection, not attribution, is what safety depends
on.

For the recovery controller, we explored every reachable state with the frozen settings (τ = 0.6). The sensors were
two front sensors (front camera, LiDAR) and two non-front cameras. The controller never looks at sensor names, and
the shield only checks membership in the front set, so these four sensors cover every role combination of the seven.

The controller has 568 reachable states and 172,672 transitions, and all six properties hold:
- C1: a confident fault isolates the sensor in the same step.
- C2: a sensor suspected in two consecutive steps is isolated.
- C3: an isolated sensor that is neither flagged nor suspected is restored.
- C4: when the VLM gives no valid answer, the deterministic flags decide.
- C5 and C6: "degraded" and "healthy" are only reported when they are consistent.

As a sanity check, switching off re-observation makes C2 fail with a counterexample.

### Requirements on the logged runs

R1–R6 are checked on every logged run: 120 test-split runs per method, and 72 CARLA runs. With no grace step, R1
flags exactly the frames that the E2 metric counts as missed fallbacks, for every method (Ours-full: 256 of 4,830
frames). This confirms that the STL encoding and the metric agree.

| Method (nuScenes test) | R1 runs ok | R2 runs ok | R3 runs ok | R6 runs ok | R6 violation rate |
|---|---|---|---|---|---|
| B0 (no monitor) | 37/120 | 120/120 | 120/120 | 120/120 | 0.000 |
| B3 (self-refine) | 108/120 | 0/120 | 12/120 | 120/120 | 0.000 |
| Checks-only | 82/120 | 8/120 | 112/120 | 87/120 | 0.014 |
| Ours-full | 82/120 | 7/120 | 112/120 | 86/120 | 0.014 |

- R1 and R2 restate the fallback results as specifications. B3 avoids missed fallbacks by being cautious everywhere,
  so it violates R2 in every run.
- R6 is violated only by the monitors that isolate sensors: in 1.4% of obligations, an isolated sensor is restored
  while its fault is still active. This happens when the faulty camera loses its reference objects (assumption A3).
- **CARLA:** Ours-full meets R2, R4 and R5 in every run, and B0 collides in 7 of 12 faulty runs (R5).
  - R6 fails in 2 of the 12 C1 runs of Ours-full. These are the restorations seen in the demo, when nothing is in view
    to cross-check.
  - R1 also fails in 2 of the 12 runs.

Treating each fault episode as one sample, the 95% Clopper–Pearson upper bound on the share of episodes with a missed
fallback (Ours-full) is:
- ≤ 4.6% for L1, L4 and the agent faults A1–A3 (no violation in 78–97 episodes each);
- 16% for C4, 26–28% for C1–C3, and 34% for L2.

These bounds hold for the tested fault distribution only.

### Falsification

Random fault injection, as in E1/E2, samples faults blindly. Falsification instead searches for the fault settings
that violate a requirement. For each fault type and requirement we ran two searches with the same budget of 60
trials:
- random sampling (today's practice);
- a guided search with Optuna's tree-structured Parzen estimator (TPE), which uses its first 10 trials as random
  samples and then focuses where the robustness was lowest.

**Search space:** the dev scene, the start frame and duration of the episode, the target camera, the subtype, and a
continuous severity level. Severity labels follow the level tables of §5.2, so some violations sit at the label
boundaries.

**Trials:**
- Each trial runs a window around one fault episode and scores it by the robustness of one requirement.
- The system under test is checks-only, which needs no VLM.
- The faults C1, C4, L1, L2 and L4 reuse only original images, so their trials run on the CPU from cached SAM 3
  detections.

**Results (C1, C4, L1, L2, L4; 2,400 trials):**

| | Random search | Guided search (TPE) |
|---|---|---|
| Trials that violate a requirement | 157 / 1,200 (13%) | 448 / 1,200 (37%) |
| (fault, requirement) pairs with more violations found | — | 16 of 20 (the other 4 tie) |
| Recovery (R3) violations found | 0 | 34, across all five faults |

At equal budget, the guided search finds about three times as many violating fault settings as random injection. It
also finds a class that random injection never hit: slow recovery after an episode ends (R3). No search found an R1
violation for LiDAR dropout or LiDAR failure (L1, L4); those faults are always caught.

**Triage.** We classified each of the 605 violating trials from its logged frame. Every one traces to one of three
causes, and none to the recovery logic or the shield (both proven above):
- **The detection limit or the known false-alarm rate of the checks:**
  - false alarms on clean frames, 219 trials;
  - a check that misses one frame of an active fault, 118;
  - faults too weak for the checks, 86.
- **A violated assumption A3,** a camera with fewer than two reference objects: 89 trials.
- **Design choices or label boundaries:**
  - one step of recovery lag, 51 trials;
  - temporal checks that compare with the still-faulty previous frame, 32;
  - levels at the low/medium label boundary, 7.

No repair was needed. The residual risk is in the monitor's evidence, not in its logic.

**[Phase B]** C2 and C3 (new images, SAM 3 on a GPU); replay of the worst cases with Ours-full (VLM); closed-loop
falsification in CARLA (R4, R5) over pedestrian timing and speed, lead-vehicle braking, fault type and fault window.

## Assurance case (before the Conclusion)

The figure shows the assurance case in Goal Structuring Notation (`docs/media/assurance_case.png`). The top claim
(G0) is that, under the faults C1–C4, L1, L2, L4 and A1–A3 and within assumptions A1–A4, the vehicle meets R1–R6.

The argument splits into four parts:
- **G1, the shield:** proven by exhaustive checking (S1–S5, S7). It holds under the assumption that the fault is
  detected; S6a and S6b show that this assumption is needed.
- **G2, the monitor's detection:** measured on the frozen test split. Missed fallback occurs on 5.3% of frames, and
  per fault type the episode-level bounds are given above.
- **G3, the requirements:** checked on every logged run, and searched for violations by falsification. Each violation
  class is traced to a cause: an assumption, a label boundary, or a fixed bug.
- **G4, failures of the VLM itself:** contained by the rule-based fallback (C4, proven). On the test split, the
  agent-fault episodes (A1–A3) show no missed fallback.

The weakest link is G2. The guarantee holds only as far as the monitor detects faults, and the residual risk sits in
low-severity image faults, small calibration shifts, and cameras without reference objects. The falsification results
show where these cases are.
