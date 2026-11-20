# Course Requirements (Specs, Proofs, Falsification, Assurance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Unattended runs (`/goal`) also follow the project
> skill `.claude/skills/autonomous-run/SKILL.md`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap between RecoverDrive and `REQUIREMENT.md`. Add formal requirements in Signal Temporal Logic
(STL) checked on the logs, formal verification of the shield and recovery logic, falsification (search-based test
generation) on nuScenes and in CARLA, and an assurance case, all written into the report.

**Architecture:** Four layers on top of the frozen system, which is not changed:
1. A small in-repo STL module computes robustness on logged traces.
2. Exhaustive checks of the real Python shield and recovery controller serve as the proofs.
3. A falsifier searches fault and scenario parameters with Optuna and scores each run by STL robustness. It compares
   random search with guided search (TPE) at equal budget.
4. An assurance-case figure links each claim to its evidence.

Slides and video are out of scope; they come last, after the method work.

**Tech Stack:** Python 3.12 (uv), numpy, scipy (Clopper–Pearson), Optuna ≥ 5 (new dependency; 5.1.0 is the latest on
PyPI as of 2026-11-20), matplotlib, pytest, ruff. SAM 3 and Qwen3-VL (vLLM) only in Tasks 7–9; CARLA 0.9.16 only in
Tasks 8–9.

**Spec:** `REQUIREMENT.md` (course grading points); design spec `docs/superpowers/specs/2026-11-03-recoverdrive-design.md`
(§4.5 shield, §5.2 acceptable actions, §5.4 evaluation rules); results and their commits in
`docs/superpowers/runs/results-provenance.md`.

### Which tasks need new experiment runs

| Task | New runs? | Needs | Phase |
|---|---|---|---|
| 1–6, 10 | No (CPU only, existing logs) | — | A |
| 7a nuScenes falsification, C1/C4/L1/L2/L4 | **Yes, CPU only** | cached SAM 3 detections (these faults reuse original images) | A |
| 7b nuScenes falsification, C2/C3 | **Yes** | SAM 3 GPU (about 3 GB) for the new images | B |
| 7c Ours-full confirmation | **Yes** | one vLLM 8B server (about 24 GB) | B |
| 8 CARLA falsification | **Yes** | CARLA server (about 5 GB GPU) + SAM 3; vLLM for the confirmation | B |
| 9 Repair and re-falsify | **Only if Task 7/8 finds a real bug** | same as the campaign that found it | A/B |
| 11 Report sections | No | draft in A with placeholders; final numbers after B | A, then B |

**Phases.** All 4 GPUs were busy on 2026-11-20 (100% load). Phase A is CPU only and runs now. Phase B starts once
`nvidia-smi` shows a card with enough free memory and low load.

Phase A must never start SAM 3: run it with `CUDA_VISIBLE_DEVICES=""` and `CachedDetector(..., allow_miss=False)`
(Task 6), so a cache miss raises an error instead of loading the model.

## Global Constraints

- Frozen system unchanged. Do not edit `configs/eval.yaml`. Any repair in Task 9 sits behind a new config switch whose
  default is the frozen behavior, and `scripts/replay_ablation.py --check` must still reproduce 4830/4830 frames on
  `test_final`.
- No tuning on the test split. Falsification on nuScenes uses the 6 dev scenes of `configs/splits.yaml` only.
  Repairs are designed and checked on dev scenes and on new CARLA seeds.
- Time step Δ = 0.5 s in both sources: nuScenes keyframes are 2 Hz, and CARLA runs perception every 5 ticks of 0.1 s.
- Action ranks: `continue` = 0, `slow_down` = 1, `stop` = 2 (`shield.ACTIONS`).
- Spec constants (review these before Task 2):
  - R1 grace k1 = 1 step.
  - R3 recovery window m = 2 steps.
  - R4: reach speed ≤ 3.06 m/s (10 km/h slow speed + 1 km/h tolerance) within T = 6 steps (3 s) of a front-camera fault onset.
  - R5: TTC threshold 2.0 s, braking reaction within 1 step, gap ≥ 0 m.
  - Vote threshold τ = 0.6 (`recovery.conf_threshold`).
- Assumptions, stated in the report with the specs:
  - A1: at most one sensor fault at a time.
  - A2: calibration and timestamps are correct.
  - A3: a camera can be cross-checked only when at least 2 LiDAR-confirmed objects (≥ 5 points, ≤ 50 m) project into it.
  - A4: LiDAR objects come from ground-truth boxes (oracle detector).
- Every new run and every generated table records `git_state` (`runner.git_state`).
- Shared GPUs: run `nvidia-smi` before each GPU job, use cards 0 and 3 unless others are free, and run vLLM on ports
  8001/8002 (another user holds 8000). Stop every server at the end.
- Jobs longer than about 10 minutes run detached (`setsid nohup ... > outputs/logs/<name>.log 2>&1 &`).
- Only new dependency: `optuna>=5`. STL is implemented in the repo, with no STL library.
- All repo files in English. Do not push. `outputs/`, `REPORT.md` and `report_figures/` stay local.
- Before every shell command: `source scripts/env.sh`.

### The specifications (copied into the report in Task 11)

STL over discrete steps; `G`, `F`, `X` are always, eventually and next; windows are in steps.

| ID | Meaning | Formula |
|---|---|---|
| R1 | No missed fallback | `G( need ≥ 1 → F[0,k1] (a − need ≥ 0) )` |
| R2 | No unnecessary fallback | `G( allow − a ≥ 0 )` |
| R3 | Recovery after a fault ends | `G( fault_end → F[0,m] (a ≤ 0 ∨ fault) )` |
| R4 | Front-fault reaction (CARLA) | `G( front_onset → F[0,6] (speed ≤ 3.06) )` |
| R5 | TTC braking and no collision (CARLA) | `G( (ttc_low → F[0,1] (a ≥ 2)) ∧ gap ≥ 0 )` |
| R6 | No restore while faulty, for each sensor s | `G( (fault_s ∧ iso_s) → X (iso_s ∨ ¬fault_s) )` |

`need` and `allow` are the ranks of the least and most cautious acceptable action (`shield.acceptable_actions`).

## Review Focus

1. **Obligations near the end of a trace.** A `F[0,m]` window that runs past the last step must not count as a
   violation. Expected: such obligations are dropped, not failed (Task 1 test
   `test_check_always_drops_obligations_past_end`).
2. **R1 must agree with the existing missed-fallback metric.** With grace k1 = 0, the frames that violate R1 must be
   exactly the frames `metrics` counts as missed. Expected: equal counts on the same records (Task 3 test
   `test_r1_grace0_matches_missed_fallback`).
3. **Continuous severity creates label boundaries.** A level just under the "medium" value is labelled "low", where
   continuing is acceptable. Expected: `severity_of` uses the existing level tables with fixed rules, and the report
   says that some violations sit at these boundaries (Task 6 test `test_severity_of_boundaries`).
4. **Window runs must not change normal runs.** `run_one` without `plan`/`frames` must behave exactly as before.
   Expected: the full suite passes and the `test_final` replay check stays 4830/4830 (Task 6).
5. **Records with frame errors, missing methods or `ttc: null`.** Expected: trace builders skip error records, report
   how many they skipped, and treat a missing TTC as "not low" (Task 2 tests `test_trace_skips_error_records`,
   `test_carla_trace_ttc_none`).

---

### Task 1: STL robustness module

**Runs:** none.

**Files:**
- Create: `src/recoverdrive/stl.py`
- Test: `tests/test_stl.py`

**Interfaces:**
- Produces:
  - Frozen dataclasses: `Pred(signal: str, op: Literal[">=", "<="], c: float)`, `Not(phi)`, `And(*args)`, `Or(*args)`,
    `Implies(lhs, rhs)`, `Always(a: int, b: int, phi)`, `Eventually(a: int, b: int, phi)`.
  - `Formula` (union type).
  - `rho(phi: Formula, trace: Mapping[str, np.ndarray], t: int) -> float`
  - `horizon(phi: Formula) -> int`
  - `@dataclass SpecResult(robustness: float, satisfied: bool, n_obligations: int, n_violations: int, worst_t: int | None)`
  - `check_always(phi: Formula, trace) -> SpecResult`

- [ ] **Step 1: Write the failing tests**

```python
def test_pred_and_not():          # x=[1,3]: rho(Pred("x",">=",2),t=1)==1; rho(Not(...),t=1)==-1; "<=" gives c-x
def test_always_window_min():     # x=[5,1,4]: rho(Always(0,2,Pred("x",">=",0)),0)==1; empty window -> +inf
def test_eventually_window_max(): # x=[5,1,4]: rho(Eventually(1,2,Pred("x",">=",0)),0)==4; empty window -> -inf
def test_implies():               # Implies(p,q) == max(-rho(p), rho(q))
def test_check_always_drops_obligations_past_end():
    # p=[0,0,0,0,1]; phi=Eventually(0,2,Pred("p",">=",0.5)); obligations only at t in {0,1,2}
    # result.n_obligations==3, n_violations==2 (t=0,1 cannot see t=4), worst_t==0, satisfied is False
def test_empty_trace_satisfied(): # zero-length trace -> SpecResult(inf, True, 0, 0, None)
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `uv run pytest tests/test_stl.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/recoverdrive/stl.py`**

Quantitative semantics:
- `Pred`: `x[t]−c` for `>=`, `c−x[t]` for `<=`.
- `Not`: `−ρ`. `And`: min. `Or`: max. `Implies`: `max(−ρ(lhs), ρ(rhs))`.
- `Always`/`Eventually`: min/max over `[t+a, t+b] ∩ [0, n−1]`, giving +inf/−inf for an empty window.
- `horizon`: largest `b` along any path, summed through nested temporal operators.
- `check_always`: evaluate φ at every `t ≤ n−1−horizon(φ)`.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_stl.py -v && uv run ruff check src tests`
Expected: 6 passed, ruff clean.

- [ ] **Step 5: Commit** `feat: STL robustness on discrete traces`

---

### Task 2: Requirements R1–R6 and trace builders

**Runs:** none.

**Files:**
- Create: `src/recoverdrive/requirements.py`
- Test: `tests/test_requirements.py`

**Interfaces:**
- Consumes: Task 1 (`Pred`, `Always`, …, `check_always`, `SpecResult`); `shield.ACTIONS`, `shield.acceptable_actions`,
  `types.FaultLabel`.
- Produces:
  - Constants: `ACTION_RANK: dict[str, int]`, `NUSCENES_SPECS = ("R1", "R2", "R3", "R6")`,
    `CARLA_SPECS = ("R1", "R2", "R4", "R5")`.
  - `spec(name: str, sensors: tuple[str, ...] = (), grace: int = 1) -> Formula`. R6 is the `And` over `sensors`;
    `grace` is k1 of R1.
  - `nuscenes_trace(records: list[dict], method: str, sensors: tuple[str, ...]) -> tuple[dict[str, np.ndarray], int]`
    returns the trace and the number of skipped records. Records are one run's frames; it sorts them by `frame_idx`.
    Signals: `a`, `need`, `allow`, `slack_lo = a − need`, `slack_hi = allow − a`, `fault`, `fault_end`, and `fault_<s>`
    and `iso_<s>` for each sensor.
  - `carla_trace(steps: list[dict], summary: dict) -> tuple[dict[str, np.ndarray], int]`. Signals: `a`, `need`, `allow`,
    `slack_lo`, `slack_hi`, `fault`, `fault_end`, `front_onset`, `speed`, `ttc_low`, `gap`. `gap` uses per-step
    `gap_m` when present (Task 8 adds it); otherwise it is −1 at the last step if `summary["collision"]`, else +1.
  - `evaluate(trace, names: tuple[str, ...], sensors: tuple[str, ...], grace: int = 1) -> dict[str, SpecResult]`

- [ ] **Step 1: Write the failing tests.** Build records with the `frame(...)` helper of
  `tests/test_recovery_shield.py` (move it to `tests/fakes.py` as `record(...)` and import it in both files).

```python
def test_r1_missed_front_fault():      # front high fault, action continue on 2 frames -> R1 violated, worst_t at onset
def test_r1_grace_allows_one_late_frame():  # continue at onset, slow_down next frame -> R1 holds with grace=1, fails with grace=0
def test_r2_unnecessary_stop_on_clean():    # clean frame with stop -> R2 robustness == -2
def test_r3_recovery_window():         # fault ends, stop for 3 more frames -> R3 fails (m=2); stop for 2 -> holds
def test_r6_restore_while_faulty():    # iso_lidar 1->0 while fault_lidar stays 1 -> R6 fails; restore after fault ends -> holds
def test_trace_skips_error_records():  # one record with "error" -> skipped count 1, trace length n-1
def test_carla_trace_ttc_none():       # ttc None -> ttc_low 0; ttc 1.5 -> 1
def test_r4_front_onset_slowdown():    # speed 8.3 m/s until 4 steps after onset then 2.8 -> holds; never slows -> fails
```

- [ ] **Step 2: Run and see them fail.** Run: `uv run pytest tests/test_requirements.py -v`. Expected: FAIL (import).
- [ ] **Step 3: Implement `src/recoverdrive/requirements.py`.** The formulas are the table in Global Constraints.
  `fault_end[t]` is 1 when `fault[t−1] = 1` and `fault[t] = 0`. `front_onset[t]` is 1 when a `camera:CAM_FRONT` fault
  starts at `t`. R6 never uses `"agent"`.
- [ ] **Step 4: Run and see them pass.** Run: `uv run pytest tests/test_requirements.py tests/test_recovery_shield.py -v`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: formal requirements R1-R6 as STL and trace builders`

---

### Task 3: Check the requirements on the existing logs, with statistical bounds

**Runs:** none. Reads `outputs/eval/test_final`, `outputs/eval/test_checks_only` and `outputs/e3/e3_final`.

**Files:**
- Create: `src/recoverdrive/bounds.py`, `scripts/check_requirements.py`
- Test: `tests/test_bounds.py`; add to `tests/test_requirements.py`

**Interfaces:**
- Consumes: Task 2; `runner.load_records`, `runner.git_state`, `bootstrap.merge_runs`, `metrics` E2 counts.
- Produces:
  - `clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]`
  - `episode_violation_bounds(rows: list[dict], conf: float = 0.95) -> list[dict]`, one row per fault type with keys
    `fault_id, episodes, violated, rate, upper`. An episode is (run, `episode_id`), and it counts as violated when any
    of its frames violates R1.
  - Files `report_figures/requirements.{md,json}`. Per method and spec: runs satisfied / runs, frame-level violation
    rate, min robustness. Also the per-fault R1 bounds for `ours_full` and `checks_only`, and `git_state`.

- [ ] **Step 1: Write the failing tests**

```python
def test_clopper_pearson_known_values():   # k=0,n=10 -> upper≈0.3085; k=5,n=10 -> (≈0.187, ≈0.813)
def test_episode_bounds_counts_episodes(): # 2 episodes, one with a violating frame -> violated=1, episodes=2
def test_r1_grace0_matches_missed_fallback():
    # on a synthetic run mixing faults and actions: frames with rho<0 for spec("R1", grace=0)
    # == frames metrics classifies as "missed"
```

- [ ] **Step 2: Run and see them fail.** Run: `uv run pytest tests/test_bounds.py tests/test_requirements.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `bounds.py`** with `scipy.stats.beta`. Implement `scripts/check_requirements.py`: arguments
  `--eval outputs/eval/test_final outputs/eval/test_checks_only`, `--carla outputs/e3/e3_final` and
  `--out report_figures/requirements`. Merge the two eval runs with `merge_runs`. Use the 6 test cameras plus `lidar`
  as R6 sensors for nuScenes, and `CAM_FRONT`, `CAM_BACK` and `lidar` for CARLA.
- [ ] **Step 4: Run tests, then the script.**
  Run: `uv run pytest -q && uv run python scripts/check_requirements.py`
  Expected: all tests pass. Then check that `report_figures/requirements.md` lists 7 methods × 4 nuScenes specs,
  3 CARLA methods × 4 CARLA specs, and the R1 bound table. Also check that the R1 grace-0 violation rate of
  `ours_full` equals E2 missed fallback 0.053 (± 0.001).
- [ ] **Step 5: Commit** `feat: check R1-R6 on the logged runs, with Clopper-Pearson bounds`

---

### Task 4: Formal verification of the shield

**Runs:** none.

**Files:**
- Create: `src/recoverdrive/formal.py`, `scripts/verify_logic.py`
- Test: `tests/test_formal.py`

**Interfaces:**
- Consumes: `shield.shield`, `shield.ttc_override`, `shield.acceptable_actions`, `shield.FRONT_SENSORS`,
  `faults.SENSOR_FAULTS`, `faults.AGENT_FAULTS`, `faults.FIXED_HIGH`, `faults.SEVERITIES`, `types.RecoveryResult`.
- Produces:
  - `@dataclass PropertyResult(name: str, statement: str, holds: bool, n_cases: int, counterexample: dict | None)`
  - `SENSORS_ALL: tuple[str, ...]` (the 6 nuScenes cameras and `"lidar"`)
  - `verify_shield() -> list[PropertyResult]`

  Exhaustive domain: 3 final states × every subset of `SENSORS_ALL ∪ {"agent"}` (768 results) × TTC values
  (None, 0.5, 1.99, 2.0, 5.0).

| Property | Statement | Expected |
|---|---|---|
| S1 assume-guarantee | Single fault (A1), correctly diagnosed and isolated (clean → healthy and []; sensor fault → degraded and [sensor]; agent fault → degraded and ["agent"]) ⇒ `shield(result)` is acceptable | holds |
| S2 monotone | Same state and `isolated ⊆ isolated'` ⇒ rank(shield(r)) ≤ rank(shield(r')) | holds |
| S3 no front perception | degraded and `FRONT_SENSORS ⊆ isolated` ⇒ `stop` | holds |
| S4 state rules | unresolved ⇒ `stop`; healthy ⇒ `continue` | holds |
| S5 TTC dominance | TTC < 2.0 ⇒ `stop`; TTC None or ≥ 2.0 ⇒ unchanged | holds |
| S6 diagnosis matters | S1 with a wrong isolated sensor | **fails**, with a counterexample (shows that S1 needs a correct diagnosis) |

- [ ] **Step 1: Write the failing tests:** `test_shield_properties_hold` asserts S1–S5 hold with `n_cases > 0`.
  `test_s6_wrong_isolation_has_counterexample` asserts S6 fails and that its counterexample names a label and a result.
- [ ] **Step 2: Run and see them fail.** Run: `uv run pytest tests/test_formal.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `verify_shield`.** It enumerates the domain and calls the real functions (no hand-written
  model), keeping the first counterexample found. `scripts/verify_logic.py` writes `report_figures/formal.{md,json}`
  with `git_state`.
- [ ] **Step 4: Run and see them pass.** Run: `uv run pytest tests/test_formal.py -v && uv run python scripts/verify_logic.py`. Expected: pass; `formal.md` lists S1–S6 with case counts.
- [ ] **Step 5: Commit** `feat: exhaustive verification of the safety shield`

---

### Task 5: Model checking of the recovery controller

**Runs:** none.

**Files:**
- Modify: `src/recoverdrive/formal.py`, `scripts/verify_logic.py`
- Test: `tests/test_formal.py`

**Interfaces:**
- Consumes: `recovery.RecoveryController`, `recovery.RecoveryConfig`, `types.Diagnosis`; Task 4 `PropertyResult`.
- Produces:
  - `@dataclass ControllerGraph(states: list[tuple], edges: list[tuple])`. A state is
    `(tuple(sorted(isolated.items())), tuple(sorted(suspected_prev)))`. An edge is `(src, input, dst, result)`.
  - `explore_controller(sensors: tuple[str, ...], cfg: RecoveryConfig, confs: tuple[float, ...] = (0.5, 1.0), max_states: int = 100_000) -> ControllerGraph`
  - `verify_controller(graph: ControllerGraph, cfg: RecoveryConfig) -> list[PropertyResult]`

  Input alphabet per step: the diagnosis is None (agent failed), ok, uncertain about s, or fault on s; times
  `conf ∈ confs`, `agent_ok ∈ {True, False}`, and `flagged ⊆ sensors`. The default sensors are `("camera:CAM_FRONT",
  "lidar", "camera:CAM_BACK", "camera:CAM_FRONT_LEFT")`: two front and two non-front sensors. The report argues by
  symmetry that this covers every role combination: the controller never looks at sensor names, and the shield only
  checks membership in `FRONT_SENSORS`.

| Property | Statement |
|---|---|
| C1 confident isolation | fault on s with conf ≥ τ ⇒ s is isolated after the step |
| C2 persistence | s suspected at t−1 and at t ⇒ s is isolated after t |
| C3 restoration | s was isolated, and s is neither flagged nor blamed ⇒ s is not isolated after the step |
| C4 agent fallback | `agent_ok` is False ⇒ `"agent"` is isolated and `"isolate:agent"` is in `steps_taken` |
| C5 verify soundness | `degraded` ⇒ no flagged sensor is left outside `isolated`, and every isolation confidence ≥ τ |
| C6 healthy soundness | `healthy` ⇒ `isolated_sensors == []` |

- [ ] **Step 1: Write the failing tests:**
  - `test_explore_reaches_fixpoint`: the number of states is finite, below `max_states`, and the same on two runs.
  - `test_controller_properties_hold`: C1–C6 hold with the default `RecoveryConfig()`.
  - `test_c2_fails_without_reobserve`: with `RecoveryConfig(reobserve=False)`, C2 fails and gives a counterexample.
    This shows that the checker can find violations.
- [ ] **Step 2: Run and see them fail.** Run: `uv run pytest tests/test_formal.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Breadth-first search over states. Rebuild a controller from each state, call the real
  `step`, and record the edges. `verify_controller` checks every edge. `scripts/verify_logic.py` appends C1–C6, the
  state and edge counts, and the symmetry note.
- [ ] **Step 4: Run and see them pass.** Run: `uv run pytest -q && uv run python scripts/verify_logic.py`. Expected: all pass; `formal.md` lists C1–C6.
- [ ] **Step 5: Commit** `feat: explicit-state model checking of the recovery controller`

---

### Task 6: Window runs and fault parameters for falsification

**Runs:** none (fakes only).

**Files:**
- Modify: `src/recoverdrive/runner.py` (`run_one`), `src/recoverdrive/detectors/sam3.py` (`CachedDetector`)
- Create: `src/recoverdrive/falsify.py`
- Test: `tests/test_falsify.py`; move `make_frame` from `tests/test_faults.py` to `tests/fakes.py` and add `FakeAdapter`
  (`scene_sample_tokens`, `load_frame`) and `FakeDetector` (`detect` returns no boxes)

**Interfaces:**
- Consumes: `faults.FaultEvent`, the level tables `C2_LEVELS`, `C3_LEVELS`, `L1_LEVELS`, `L2_LEVELS`, `C4_LAGS`,
  `faults.SEVERITIES`; Task 2 `nuscenes_trace`, `evaluate`.
- Produces:
  - `run_one(spec, env, plan: list[FaultEvent | None] | None = None, frames: range | None = None) -> dict`. With the
    defaults it behaves exactly as now. `plan` replaces `plan_faults`; `frames` limits the frame indices processed.
  - `CachedDetector(..., allow_miss: bool = True)`. With `False`, a cache miss raises `CacheMissError` (new, in
    `detectors/sam3.py`) instead of loading SAM 3.
  - `CPU_FAULTS = ("C1", "C4", "L1", "L2", "L4")`: the faults whose images are all original nuScenes images.
  - `FAULT_SPACES: dict[str, dict]`: the search ranges per fault and subtype.
    - C2: rotate 0–12°, shift 0–0.2, and the sign.
    - C3: blur 0–12, noise 0–80, overexpose 1–5, black.
    - C4: lag 1, 2 or freeze.
    - L1: random 0–1, beam 0–1, sector 0–360°.
    - L2: translation 0–1.5 m and yaw 0–8°.
    - C1 and L4: no level.
  - `severity_of(fault_id: str, subtype: str | None, level: float | None) -> str`. "low" below the medium value of the
    level table, "medium" below the high value, "high" at or above it. C1, L4 and black are always "high".
  - `event_from_params(fault_id: str, p: dict, episode_id: int = 0) -> FaultEvent`
  - `window_plan(n_frames: int, start: int, duration: int, event: FaultEvent) -> list[FaultEvent | None]`
  - `window_frames(n_frames: int, start: int, duration: int, warmup: int = 3, tail: int = 3) -> range`
  - `@dataclass TrialResult(robustness: float, satisfied: bool, n_violations: int, params: dict, run_dir: str)`
  - `evaluate_trial(env, scene: str, fault_id: str, p: dict, method: str, spec_name: str, out_dir: Path) -> TrialResult`

- [ ] **Step 1: Write the failing tests:**
  - `test_run_one_defaults_unchanged`: a fake run gives the same records with and without explicit `plan=None, frames=None`.
  - `test_run_one_window_only_processes_frames`: only the indices in `frames` are written.
  - `test_severity_of_boundaries`: C3 blur 4.99 → "low", 5.0 → "medium", 10.0 → "high"; C1 → "high".
  - `test_event_from_params_ranges`: out-of-range values are clipped into `FAULT_SPACES`; the C4 window keeps ≥ 3
    warm-up frames.
  - `test_evaluate_trial_scores_spec`: with fakes, a C3 black fault on CAM_FRONT and method `checks_only` gives a
    `TrialResult` with finite robustness.
  - `test_cached_detector_cache_only_raises_on_miss`: with `allow_miss=False`, a hit returns the cached boxes and a
    miss raises `CacheMissError` without loading a model.
- [ ] **Step 2: Run and see them fail.** Run: `uv run pytest tests/test_falsify.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `run_one`, use `plan` when it is given and loop over `frames or range(len(tokens))`.
  `evaluate_trial` builds the event and the window, calls `run_one` with `out_dir` per trial, loads its records,
  builds the trace, and scores `spec_name`.
- [ ] **Step 4: Run and see them pass, and check the replay.**
  Run: `uv run pytest -q && uv run python scripts/replay_ablation.py --runs outputs/eval/test_final --check`
  Expected: all pass; the replay reproduces 4830/4830.
- [ ] **Step 5: Commit** `feat: window runs and fault parameters for falsification`

---

### Task 7: Falsification campaigns on nuScenes

**Runs: YES.** SAM 3 on one GPU, for faulted camera images only, because unchanged images hit the cache. One vLLM
server (8B) for the confirmation step.

**Files:**
- Modify: `pyproject.toml` (add `optuna>=5`), `uv.lock`
- Create: `scripts/falsify_nuscenes.py`
- Test: `tests/test_falsify.py` (sampler and budget test with a fake objective)

**Interfaces:**
- Consumes: Task 6 `evaluate_trial`, `FAULT_SPACES`; `optuna.samplers.RandomSampler`, `optuna.samplers.TPESampler`.
- Produces:
  - `campaign(objective: Callable[[dict], TrialResult], space: dict, sampler: str, budget: int, seed: int) -> list[TrialResult]`
    in `falsify.py`. `sampler` is "random" or "tpe". The search space has the dev scene, fault subtype, target camera
    (camera faults), start frame, duration 2–6, and level.
  - Files `outputs/falsify/nuscenes_<tag>/{meta.json, trials.jsonl, summary.md}`. `summary.md` gives, per campaign:
    violations found, the lowest robustness, the trial index of the first violation, and the distinct failure types
    (fault, subtype, front or rear camera).

- [ ] **Step 1: Write the failing test** `test_campaign_budget_and_seed`. With a fake objective (robustness = level − 0.5),
  both samplers run exactly `budget` trials. The same seed gives the same trials. TPE reaches robustness < 0 in no more
  trials than random over 5 seeds.
- [ ] **Step 2: Run and see it fail.** Run: `uv run pytest tests/test_falsify.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `campaign` and the script.** The script takes `--faults C2 C3 C4 L1 L2 C1 --specs R1 R2 R3 R6
  --samplers random tpe --budget 60 --seed 0 --method checks_only --tag ...`. It writes `meta.json` with `git_state`
  and refuses test-split scenes. A `--confirm K --method ours_full` mode reruns the K lowest-robustness trials of a
  finished campaign with the VLM.
- [ ] **Step 4: Run the tests, then the experiments.**
  - [ ] `uv run pytest -q`. Expected: all pass.
  - [ ] Smoke: `--faults C3 --specs R1 --budget 5`. Expected: 10 trials (5 per sampler), no errors.
  - [ ] **7a (Phase A, CPU):** detached run, with `CUDA_VISIBLE_DEVICES=""` and `--cache-only`
    (`allow_miss=False`): `--faults C1 C4 L1 L2 L4`, 4 specs × 2 samplers × 60 trials = 2,400 window runs with
    `checks_only`. Expected: zero cache misses, about 1–2 h.
  - [ ] **7b (Phase B, GPU):** `--faults C2 C3` on a free card (SAM 3 about 3 GB). Expected: 960 window runs, about 1 h.
  - [ ] **7c (Phase B, GPU):** `--confirm 10 --method ours_full` per campaign with a counterexample. vLLM 8B on a free
    card, port 8001. Expected about 2–3 h. Stop the server afterwards.
  - [ ] Expected output: `summary.md` with the random vs. TPE comparison, and for each counterexample whether
    Ours-full also violates it.
- [ ] **Step 5: Commit** `feat: nuScenes falsification with random and TPE search` (code only; outputs stay local)

---

### Task 8: Falsification in CARLA (closed loop)

**Runs: YES.** CARLA server (`scripts/carla_server.sh`) and SAM 3 on one GPU. One vLLM server for the confirmation.

**Files:**
- Modify: `scripts/run_e3.py`
- Create: `scripts/falsify_carla.py`
- Test: `tests/test_carla_adapter.py` (scenario defaults)

**Interfaces:**
- Consumes: Task 2 `carla_trace`, `evaluate`; Task 7 `campaign`; `faults.inject_c1`, `inject_c2`, `inject_c3`,
  `inject_l1`.
- Produces:
  - `@dataclass Scenario` in `run_e3.py` with fields and defaults that reproduce E3 exactly:
    - `hazard: str`
    - `trigger_offset_m: float | None = None` (None means the current random ±3 m)
    - `walker_speed_mps: float = 1.6`
    - `lead_brake_s: float = 3.0`
    - `fault: str = "C1"` (one of "none", "C1", "C3_black", "C2_rotate", "L1_sector")
    - `fault_level: float | None = None`
    - `fault_start_m: float = -25.0`, `fault_end_m: float = 30.0` (relative to the trigger point)
  - `run_route(..., scenario: Scenario | None = None)`. It also writes per step `gap_m`: the distance between the
    centers of the ego and the hazard actor minus 2.6 m (walker) or 4.7 m (lead car), and `collision: bool`.
  - Files `outputs/falsify/carla_<tag>/{meta.json, trials.jsonl, summary.md}`.

- [ ] **Step 1: Write the failing test** `test_scenario_defaults_match_e3`: `Scenario(hazard="pedestrian")` gives
  walker speed 1.6, brake 3.0, a C1 window of −25 to +30 m and `trigger_offset_m=None`.
- [ ] **Step 2: Run and see it fail.** Run: `uv run pytest tests/test_carla_adapter.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the `Scenario` plumbing in `run_route`, the extra step fields, and
  `scripts/falsify_carla.py`. Per hazard, it searches the trigger offset ±5 m, walker speed 1.0–2.5 m/s, lead brake
  1–4 s, the fault type, the fault level, and the fault window. The objective is the robustness of R5 (and R4 for
  front faults). The method is `checks_only`. The script writes `meta.json` with `git_state`.
- [ ] **Step 4: Run the tests, then the experiments.**
  - [ ] `uv run pytest -q`. Expected: all pass.
  - [ ] Regression: rerun one E3 cell (`r0` pedestrian C1 `ours_full` s0) with the default `Scenario`. Expected: same
    collision result and route completion as `outputs/e3/e3_final`.
  - [ ] Full run (detached): 2 hazards × 2 samplers × 40 runs = 160 CARLA runs. Expected about 45 s each, so 2 h.
  - [ ] Confirm: the 5 lowest-robustness scenarios per hazard with `ours_full` (vLLM on port 8002). Expected about 30 min.
  - [ ] Expected output: `summary.md` with collisions and R4/R5 violations found by random vs. TPE, and the
    scenario parameters of each counterexample.
- [ ] **Step 5: Commit** `feat: closed-loop falsification in CARLA`

---

### Task 9: Triage, repair and re-falsify

**Runs: ONLY IF Task 7 or 8 finds a real bug.** Rerun the affected campaign with the same budget and a new seed.

**Files:**
- Create: `docs/superpowers/runs/2026-11-20-falsification.md` (triage log)
- Modify (only for a real bug): the module at fault, behind a new switch whose default is the frozen behavior. The
  switch is turned on only in a new `configs/falsify_repair.yaml`; `configs/eval.yaml` is not edited.
- Test: one regression test per repaired counterexample, in the owning module's test file

**Interfaces:**
- Consumes: Task 7/8 `trials.jsonl`.
- Produces: the triage log. For each counterexample class it records the cause, one of:
  - (a) a violated assumption A1–A4, for example R6 on a camera with fewer than 2 reference objects;
  - (b) a labelling boundary (Review Focus 3);
  - (c) a real bug.
  It also records the action taken and, for (c), a before/after robustness table.

- [ ] **Step 1:** Write the triage log from `summary.md` and the counterexample traces. Expected: every class has a
  cause.
- [ ] **Step 2 (only for class c):** Write the regression test from the counterexample parameters, run it, and see it
  fail.
- [ ] **Step 3 (only for class c):** Fix it behind the switch. The test passes, `uv run pytest -q` passes, and the
  `test_final` replay check is still 4830/4830.
- [ ] **Step 4 (only for class c):** Rerun the campaign with the switch on (same budget, seed + 1). Expected: no
  violation of that class.
- [ ] **Step 5: Commit** `docs: falsification triage` (plus `fix: ...` for any repair)

---

### Task 10: Assurance case figure

**Runs:** none.

**Files:**
- Create: `scripts/make_assurance_figure.py` → `docs/media/assurance_case.png`. Same matplotlib style as
  `scripts/make_architecture_figure.py`.

**Interfaces:**
- Consumes: the numbers in `report_figures/requirements.md`, `formal.md` and the falsification summaries. The script
  takes them as arguments or from those files, so it can redraw the figure after reruns.
- Produces: a Goal Structuring Notation (GSN) figure with these nodes.
  - G0: under faults C1–C4, L1, L2, L4 and A1–A3 and assumptions A1–A4, the vehicle meets R1–R6.
  - S0 (strategy): argue over the shield, the monitor, the closed loop and the VLM failures.
  - Shield branch: G1 shield correct → E1 S1–S5 proved (Task 4).
  - Monitor branch: G2 monitor accurate enough → E2 frozen test results, bootstrap intervals and R1 bounds (Task 3).
  - Closed-loop branch: G3 specs hold on logs and under search → E3 R1–R6 table (Task 3) and falsification results (Tasks 7–9).
  - VLM-failure branch: G4 VLM failures contained → E4 C4 property (Task 5) and A1–A3 results.
  - Context nodes: C-A1…A4. Assumption node: S1 requires a correct diagnosis (S6).

- [ ] **Step 1:** Implement the script and run it: `uv run python scripts/make_assurance_figure.py`. Expected: the PNG
  exists, and all node text is readable at 100% zoom (open the file and check).
- [ ] **Step 2: Commit** `docs: assurance case figure`

---

### Task 11: Report sections and documentation

**Runs:** none.

**Files:**
- Create: `docs/requirements-and-verification.md` (English source text of the new report sections)
- Modify: the report Google Doc (`1TejQjwoDeW_gQMlVCWMl-G1R87ICXqNJk1hCnP4fHA8`, in place with the google-workspace
  skill), `docs/superpowers/runs/results-provenance.md`, `README.md` (one results line), `FutureWork.md`
- Drive: a small zip of `outputs/falsify/` and the new `report_figures/*.md` into `experiment_result` (< 10 MB per file)

- [ ] **Step 1:** Write `docs/requirements-and-verification.md` with three sections:
  - **Requirements:** R1–R6 table, assumptions A1–A4, why the set is more than reach-avoid.
  - **Verification and falsification:** shield and controller properties with case counts; R1–R6 on the logs;
    Clopper–Pearson bounds; random vs. TPE results; triage.
  - **Assurance case:** the figure and one paragraph per branch.
  Every number is copied from the generated files.
- [ ] **Step 2:** Insert the sections into the report: Requirements after "Assumptions and Constraints", Verification
  and falsification in §6, Assurance case before the Conclusion. Update the Abstract, the Conclusion and the
  Limitations. Verify by reading the Doc back.
- [ ] **Step 3:** Add provenance rows for Tasks 3–8 (commit and path), and upload the results zip to Drive.
- [ ] **Step 4:** Check every requirement against `REQUIREMENT.md` and list each point next to the report section that
  meets it.
- [ ] **Step 5: Commit** `docs: requirements, verification and assurance sections`
