# Step 1: Report Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline) or
> superpowers:subagent-driven-development to implement this plan task-by-task. Unattended runs (`/goal`) also follow
> the project skill `.claude/skills/autonomous-run/SKILL.md`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the four Step 1 analyses of `FutureWork.md` to the existing results — a checks-only baseline, easy vs.
hard faults, bootstrap confidence intervals, and the 4B model with the VLM gate — without changing Ours, and make
every run record the git commit that produced it.

**Architecture:** Almost everything is offline. The checks-only baseline is a new method without a VLM that reuses the
recovery controller through a new rules-only entry point; its test run reuses the cached SAM 3 detections and must
reproduce the frozen run's check flags on every frame. Easy/hard tables and bootstrap intervals are computed from the
logged records. The 4B + gate accuracy comes from the existing offline replay of the 4B test run; only its latency
needs a short GPU run.

**Tech Stack:** Python 3.12 (uv), numpy, pytest, ruff. vLLM (Qwen3-VL-8B/4B) and SAM 3 only in Tasks 6–7.

**Spec:** `FutureWork.md` (section "Step 1"); design spec `docs/superpowers/specs/2026-11-03-recoverdrive-design.md`
(§5.2 metrics, §5.4 evaluation rules); current results and their commits in
`docs/superpowers/runs/results-provenance.md`.

## Global Constraints

- Ours (`ours_detect`, `ours_full`) must not change: frozen settings of commit `5d5283c`; `configs/eval.yaml` is not edited.
- On the test split, run only the new baseline (`checks_only`) and offline analyses. A test run needs `frozen: true`
  and a clean tracked tree (`scripts/run_eval.py` enforces both). Use the dev split for smoke tests.
- Every new run and every generated table records `git_state` (Task 1).
- Shared GPUs: run `nvidia-smi` before each GPU job and keep ≥ 9 GB free on every card you use. One vLLM server per
  card at `VLM_GPU_UTIL` ≤ 0.5. Stop every server at the end and show `nvidia-smi`.
- Jobs longer than about 10 minutes run detached (`setsid nohup ... > outputs/logs/<name>.log 2>&1 &`) or in the
  background, and are then polled. A vLLM server is ready when `curl -sf http://127.0.0.1:<port>/health` succeeds.
- Do not edit the design spec. Do not push. `outputs/`, `REPORT.md` and `report_figures/` stay local (untracked).
- All repo files in English.
- Before every shell command: `source scripts/env.sh`.

## Review Focus

1. **The recovery refactor (Task 2) silently changes Ours-full.** Expected: the offline replay still reproduces every
   logged Ours-full frame (`--check` 4830/4830 on `test_final` and `test_abl_4b`), and the full test suite passes.
2. **The checks-only run sees different inputs from the frozen run** (other detector cache, other seeds). Expected:
   `scripts/compare_flags.py` finds identical flags on every frame (Task 3 on dev, Task 6 on test); any difference stops
   the plan (BLOCKED).
3. **Bootstrap strata with repeated or missing blocks** (a block drawn several times must count several times; a
   stratum with a single block must not crash). Tests in Task 5.
4. **Severity labels:** C1 and L4 are always labeled `high`; L1 counts as easy only at `high`; A2 is hard. Test in Task 4.
5. **The 4B timing run overwrites the 8B E4 results or uses a warm cache.** Expected: `latency_report.py --out` writes
   to a new folder and each timing run uses a cache file that did not exist before. Covered in Task 7.

---

### Task 1: Record the git state in every run

**Files:**
- Modify: `src/recoverdrive/runner.py` (add `git_state` next to `check_test_allowed`)
- Modify: `scripts/run_eval.py` (meta.json), `scripts/run_e3.py` (meta.json), `scripts/latency_report.py`
  (latency.json), `scripts/replay_ablation.py` (ablation_replay.json)
- Test: `tests/test_metrics_runner.py`

**Interfaces:**
- Produces: `git_state(repo_root: Path) -> dict` returning `{"commit": str | None, "dirty": bool | None}`. "dirty"
  uses the same rule as `check_test_allowed`: `git status --porcelain --untracked-files=no` is non-empty. It never
  raises: outside a git repository both values are `None`.

- [ ] **Step 1: Write the failing test**

```python
def test_git_state_clean_dirty_and_outside(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    (repo / "f.txt").write_text("a")
    git("add", "f.txt")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    state = git_state(repo)
    assert len(state["commit"]) == 40 and state["dirty"] is False
    (repo / "f.txt").write_text("b")
    assert git_state(repo)["dirty"] is True
    assert git_state(tmp_path) == {"commit": None, "dirty": None}
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_metrics_runner.py::test_git_state_clean_dirty_and_outside -q`
Expected: FAIL (`ImportError: cannot import name 'git_state'`).

- [ ] **Step 3: Implement `git_state` and write it into every output**

- `run_eval.py`: `meta = {"args": ..., "config": cfg, "commit": commit, "git": git_state(REPO)}` for every split (keep
  the existing `commit` key; it is still set only for the test split).
- `run_e3.py`: add `"git": git_state(REPO)` to `meta.json` (define `REPO` like `run_eval.py` does).
- `latency_report.py` and `replay_ablation.py`: add a `"git"` key to the JSON they write.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: all tests pass (106 before this plan, plus the new ones); ruff prints `All checks passed!`.

- [ ] **Step 5: Commit** — `git commit -m "feat: record the git state in every run and report"`

---

### Task 2: Rules-only entry point in the recovery controller

**Files:**
- Modify: `src/recoverdrive/recovery.py` (`RecoveryController`)
- Test: `tests/test_recovery_shield.py`

**Interfaces:**
- Produces: `RecoveryController.step_suspects(suspects: set[str], conf: float, flagged: set[str], steps: list[str]) -> RecoveryResult`.
  Same re-observe / isolate / verify logic as `step`, but the caller supplies the suspects (no VLM). The agent is never
  isolated and `final_diagnosis` is `None`.
- `step()` keeps its signature and behavior. Both methods call a private
  `_resolve(suspects, conf, flagged, steps, agent_failed: bool, diag)`: the code after suspect selection in today's
  `step`, moved without changes. The `isolate=False` and `reobserve=False` switches apply to both entry points.

- [ ] **Step 1: Write the failing tests**

```python
def test_step_suspects_isolates_every_suspect():
    c = RecoveryController()
    r = c.step_suspects({"camera:CAM_FRONT", "lidar"}, 1.0, {"camera:CAM_FRONT", "lidar"}, ["rules"])
    assert r.final_state == "degraded" and r.isolated_sensors == ["camera:CAM_FRONT", "lidar"]
    assert r.final_diagnosis is None


def test_step_suspects_low_confidence_needs_two_frames():
    c = RecoveryController()
    assert c.step_suspects({"lidar"}, 0.5, {"lidar"}, []).final_state == "unresolved"  # pending
    r = c.step_suspects({"lidar"}, 0.5, {"lidar"}, [])
    assert r.final_state == "degraded" and r.isolated_sensors == ["lidar"]


def test_step_suspects_restores_after_clean_frame():
    c = RecoveryController()
    c.step_suspects({"lidar"}, 1.0, {"lidar"}, [])
    r = c.step_suspects(set(), 1.0, set(), [])
    assert r.final_state == "healthy" and r.isolated_sensors == []
```

- [ ] **Step 2: Run them and see them fail** — `uv run pytest tests/test_recovery_shield.py -q` → 3 failures
  (`AttributeError: ... 'step_suspects'`).

- [ ] **Step 3: Implement** the refactor described in Interfaces.

- [ ] **Step 4: Verify that Ours-full did not change**

Run:
```bash
uv run pytest -q && uv run ruff check src tests scripts
uv run python scripts/replay_ablation.py --tag test_final --check
uv run python scripts/replay_ablation.py --tag test_abl_4b --check
```
Expected: tests pass; both replays print `variant 'full' reproduces 4830/4830 logged Ours-full frames`.

- [ ] **Step 5: Commit** — `git commit -m "refactor: rules-only entry point in the recovery controller"`

---

### Task 3: Checks-only baseline and runs without a VLM server

**Files:**
- Modify: `src/recoverdrive/checks.py` (`rank_sensors`)
- Modify: `src/recoverdrive/methods.py` (`ChecksOnly`, `rules_outcome`, `VLM_FREE_METHODS`, `needs_vlm`, `build_methods`)
- Modify: `src/recoverdrive/metrics.py` (`RECOVERY_METHODS`)
- Modify: `scripts/run_eval.py` (skip the VLM server when no selected method needs it)
- Create: `scripts/compare_flags.py`
- Test: `tests/test_methods.py`, `tests/test_checks.py`

**Interfaces:**
- Consumes: `RecoveryController.step_suspects` (Task 2).
- Produces:
  - `rank_sensors(flags: list[str]) -> list[str]` (checks.py): sensors with fault flags, most flags first, ties by
    name; built on `flags_by_sensor`, so `:no_reference` flags are ignored; `agent` is dropped.
  - `rules_outcome(result: RecoveryResult, ranked: list[str]) -> tuple[bool, str]` (methods.py):
    alarm = `final_state != "healthy"`. Blamed sensor = the first ranked sensor that is isolated, else the first ranked
    sensor, else `"none"`.
  - `class ChecksOnly`: `names = ("checks_only",)`; `__init__(self, cfg: dict)` builds its own
    `RecoveryController(RecoveryConfig(**cfg.get("recovery", {})))`. `step(ctx)` does:
    `ranked = rank_sensors(ctx.check.report.flags)`, then
    `result = controller.step_suspects(set(ranked), 1.0, set(ranked), ["rules"])`, then `action = shield(result)`.
    It returns `MethodOutput(None, True, alarm, pred, recovery=result, action=action, n_calls=0, extra={"flagged": ranked})`.
    When `ctx.agent_fault` is set, it also sets `extra["agent_recovered"] = True` (no VLM call, so the agent fault has
    no effect; same convention as the VLM gate).
  - `VLM_FREE_METHODS = ("B0", "checks_only")` and `needs_vlm(methods: list[str], ablation: str) -> bool`
    (True if any method is outside `VLM_FREE_METHODS` or `ablation == "qwen_grounding"`). `build_methods` accepts
    `"checks_only"` with `client=None`.
  - `metrics.RECOVERY_METHODS = ("ours_full", "checks_only")`.
  - `scripts/compare_flags.py --a TAG --b TAG`: joins the two runs' records on `(scene, seed, run_fault, frame_idx)`.
    Every frame of `--a` must exist in `--b` with identical `flags`. It prints `flags identical on N/M frames` and exits
    with 1 on any difference or missing frame.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checks.py
def test_rank_sensors_most_flags_first():
    flags = ["camera:CAM_BACK:low_match", "lidar:low_points", "lidar:point_drop", "camera:CAM_FRONT:no_reference"]
    assert rank_sensors(flags) == ["lidar", "camera:CAM_BACK"]


# tests/test_methods.py
def test_checks_only_healthy_without_flags():
    out = ChecksOnly({"recovery": {}}).step(make_ctx())["checks_only"]
    assert not out.alarm and out.pred_sensor == "none" and out.action.action == "continue" and out.n_calls == 0


def test_checks_only_isolates_flagged_camera():
    out = ChecksOnly({"recovery": {}}).step(make_ctx(flags=["camera:CAM_FRONT:low_match"]))["checks_only"]
    assert out.recovery.final_state == "degraded" and out.recovery.isolated_sensors == ["camera:CAM_FRONT"]
    assert out.alarm and out.pred_sensor == "camera:CAM_FRONT" and out.action.action == "slow_down"


def test_checks_only_blames_the_most_flagged_sensor():
    flags = ["camera:CAM_FRONT:low_match", "lidar:low_points", "lidar:sector_gap"]
    out = ChecksOnly({"recovery": {}}).step(make_ctx(flags=flags))["checks_only"]
    assert out.pred_sensor == "lidar" and out.action.action == "stop"  # front camera and LiDAR both isolated


def test_checks_only_agent_fault_has_no_effect():
    a2 = AgentFaultState("A2", "low", {"n_affected": 1})
    out = ChecksOnly({"recovery": {}}).step(make_ctx(agent_fault=a2))["checks_only"]
    assert out.extra["agent_recovered"] and not out.alarm


def test_vlm_free_methods():
    assert [m.names for m in build_methods(["B0", "checks_only"], None, {"recovery": {}})] == [("B0",), ("checks_only",)]
    assert not needs_vlm(["B0", "checks_only"], "none")
    assert needs_vlm(["checks_only", "ours_full"], "none") and needs_vlm(["checks_only"], "qwen_grounding")
```

- [ ] **Step 2: Run them and see them fail** — `uv run pytest tests/test_checks.py tests/test_methods.py -q`.

- [ ] **Step 3: Implement** the interfaces above. In `run_eval.py`, create the `VLMClient` and check the servers only
  if `needs_vlm(args.methods, args.ablation)`; otherwise use `client = None`.

- [ ] **Step 4: Unit verification** — `uv run pytest -q && uv run ruff check src tests scripts` → all pass.

- [ ] **Step 5: Dev run without a VLM server** (SAM 3 needs about 3 GB on one GPU; no vLLM)

```bash
nvidia-smi   # pick a card with >= 9 GB free
CUDA_VISIBLE_DEVICES=<g> uv run python scripts/run_eval.py --split dev --faults all --methods checks_only \
    --seeds 0 1 2 --tag dev_checks_only
uv run python scripts/compare_flags.py --a p3_dev --b dev_checks_only
```
Expected: `180 runs, ... 0 frame errors`; `flags identical on 2430/2430 frames` (p3_dev is dev, all faults, seed 0).
`dev_checks_only` (seeds 0–2) is also the training data for the classifier in Step 2.

- [ ] **Step 6: Commit** — `git commit -m "feat: checks-only baseline (no VLM) and VLM-free runs"`

---

### Task 4: Easy vs. hard faults

**Files:**
- Modify: `src/recoverdrive/metrics.py`
- Test: `tests/test_metrics_runner.py`

**Interfaces:**
- Produces:
  - `difficulty(label: dict) -> str | None`. Returns `"easy"` when `fault_id` is L4, A1 or A3, or when `fault_id == "L1"`
    and `severity == "high"` (these are caught by a point count or by JSON validation, which every VLM method has).
    Returns `"hard"` for every other fault and `None` for clean frames.
  - `difficulty_table(records: list[dict], methods: list[str], recovery_methods: tuple[str, ...] = RECOVERY_METHODS) -> list[dict]`.
    One row per method × difficulty in `("easy", "hard", "all")`, over fault frames only (seeds pooled), with columns
    `method, difficulty, n_fault, R, sensor_acc` (from `e1_counts`) and `missed_fault, rec_success` (from `e2_counts`).

- [ ] **Step 1: Write the failing tests**

```python
def _label(fault, severity, sensor="x"):
    return {"fault_id": fault, "sensor": sensor, "severity": severity, "episode_id": 0}


def test_difficulty_rules():
    assert difficulty({"fault_id": "none", "sensor": "none", "severity": None, "episode_id": None}) is None
    easy = [("L4", "high"), ("L1", "high"), ("A1", "low"), ("A3", "medium")]
    hard = [("L1", "medium"), ("A2", "high"), ("C1", "high"), ("L2", "high")]
    assert [difficulty(_label(f, s)) for f, s in easy] == ["easy"] * 4
    assert [difficulty(_label(f, s)) for f, s in hard] == ["hard"] * 4


def test_difficulty_table_uses_fault_frames_only():
    def rec(fault, severity, sensor, alarm, pred):
        return {"scene": "s", "seed": 0, "frame_idx": 0, "run_fault": fault, "label": _label(fault, severity, sensor),
                "methods": {"m": {"alarm": alarm, "pred_sensor": pred, "action": {"action": "stop"}}}}

    clean = {"scene": "s", "seed": 0, "frame_idx": 2, "run_fault": "C2",
             "label": {"fault_id": "none", "sensor": "none", "severity": None, "episode_id": None},
             "methods": {"m": {"alarm": True, "pred_sensor": "lidar", "action": {"action": "stop"}}}}
    recs = [rec("L4", "high", "lidar", True, "lidar"), rec("C2", "low", "camera:CAM_FRONT", False, "none"), clean]
    rows = {r["difficulty"]: r for r in difficulty_table(recs, ["m"], recovery_methods=())}
    assert rows["easy"]["R"] == 1.0 and rows["easy"]["sensor_acc"] == 1.0
    assert rows["hard"]["R"] == 0.0 and rows["all"]["n_fault"] == 2
```

- [ ] **Step 2: Run them and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q` → pass.

- [ ] **Step 5: Commit** — `git commit -m "feat: easy vs hard fault tables"`

---

### Task 5: Bootstrap confidence intervals

**Files:**
- Create: `src/recoverdrive/bootstrap.py`, `scripts/bootstrap_ci.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Produces:
  - `block_ids(records: list[dict]) -> list[tuple]`: one id per record. Records of one run must be in frame order.
    A fault frame gets `(scene, seed, run_fault, "fault", episode_id)`. A clean frame gets
    `(scene, seed, run_fault, "clean", k)`, where k is the index of its clean gap within the run.
  - `bootstrap_metrics(records: list[dict], methods: list[str], reference: str, n_boot: int = 2000, seed: int = 0) -> list[dict]`.
    It returns rows `{"method", "metric", "point", "lo", "hi"}` for the metrics `F1, sensor_acc, false_alarm` (E1) and
    `unnecessary, missed` (E2). It also returns paired-difference rows with `"method": f"{reference} - {m}"` for every
    other method.
    - Seeds are pooled.
    - Blocks are resampled with replacement within strata `(run_fault, "fault" | "clean")`, keeping each stratum's
      block count.
    - The same draws are used for every method (paired).
    - The interval is the 2.5 / 97.5 percentile.
    - For speed, precompute per-block count vectors per method (tp, fp, fn, tn, correct_sensor, unnecessary, missed,
      n). Each replicate is then a weighted sum of block vectors, and the metric formulas are those of `e1_counts` /
      `e2_counts`.
  - `merge_runs(runs: list[tuple[list[dict], str | None]]) -> list[dict]`: joins records of several tags on
    `(scene, seed, run_fault, frame_idx)` and merges their `methods` dicts.
    - A non-None prefix renames that tag's methods to `f"{prefix}:{name}"`.
    - It raises `ValueError` if the labels of a joined frame differ, or if two tags bring the same method name.
  - `scripts/bootstrap_ci.py --tags TAG[=PREFIX] ... --methods ... --reference ours_full --n-boot 2000 --out PATH_STEM`
    writes `PATH_STEM.md` and `PATH_STEM.json` (the JSON includes `git_state`).

- [ ] **Step 1: Write the failing tests.** Add a helper `_synthetic(**styles)` that builds two runs (faults C1 and L1,
  10 frames each: clean, clean, episode 0 × 2, clean, episode 1 × 2, clean × 3) and one method per keyword argument:
  - `"good"`: alarms exactly on fault frames, with the right sensor and acceptable actions.
  - `"bad"`: alarms only on clean frames.
  - `"mixed"`: right on episode 0, misses episode 1, one clean false alarm.

```python
def test_block_ids_split_episodes_and_clean_gaps():
    ids = block_ids(_synthetic(a="good")[:10])  # one run: gaps [0,1], [4], [7,8,9]; episodes [2,3], [5,6]
    assert len(set(ids)) == 5
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[7] == ids[9] and ids[4] != ids[0]


def test_identical_methods_have_zero_difference():
    rows = bootstrap_metrics(_synthetic(a="good", b="good"), ["a", "b"], reference="a", n_boot=200)
    d = next(r for r in rows if r["method"] == "a - b" and r["metric"] == "F1")
    assert d["lo"] == d["hi"] == 0.0


def test_clear_gap_excludes_zero():
    rows = bootstrap_metrics(_synthetic(a="good", b="bad"), ["a", "b"], reference="a", n_boot=200)
    assert next(r for r in rows if r["method"] == "a - b" and r["metric"] == "F1")["lo"] > 0


def test_point_estimate_equals_pooled_e1():
    recs = _synthetic(a="mixed")
    f1 = next(r for r in bootstrap_metrics(recs, ["a"], "a", n_boot=50) if r["method"] == "a" and r["metric"] == "F1")
    assert f1["point"] == pytest.approx(e1_counts(method_frames(recs, "a"))["F1"])


def test_same_seed_same_result():
    recs = _synthetic(a="mixed", b="bad")
    assert bootstrap_metrics(recs, ["a", "b"], "a", n_boot=100, seed=3) == \
        bootstrap_metrics(recs, ["a", "b"], "a", n_boot=100, seed=3)


def test_merge_runs_prefix_and_conflicts():
    a, b = _synthetic(a="good"), _synthetic(a="bad")
    merged = merge_runs([(a, None), (b, "v2")])
    assert set(merged[0]["methods"]) == {"a", "v2:a"}
    with pytest.raises(ValueError):
        merge_runs([(a, None), (b, None)])  # same method name twice
```

- [ ] **Step 2: Run them and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q` → pass.

- [ ] **Step 5: Run on the frozen results (CPU, < 1 min)**

```bash
uv run python scripts/bootstrap_ci.py --tags test_final --methods B1 B2 B3 ours_detect ours_full \
    --reference ours_full --out report_figures/bootstrap_test_final
```
Expected: a table; the pooled Ours-full F1 point estimate is about 0.76–0.77. The report's 0.765 is a mean over
seeds, so a small difference is expected and must be stated next to the table.

- [ ] **Step 6: Commit** — `git commit -m "feat: block bootstrap confidence intervals"`

---

### Task 6: Checks-only on the test split

No new code. This is the only new test-split run of this plan.

- [ ] **Step 1: Clean tree** — `git status --porcelain --untracked-files=no` prints nothing.
- [ ] **Step 2: Run** (no vLLM; SAM 3 detections of the frozen run come from the cache)

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<g> uv run python scripts/run_eval.py --split test --faults all --methods checks_only \
    --seeds 0 1 2 --tag test_checks_only
uv run python scripts/compare_flags.py --a test_final --b test_checks_only
```
Expected: `120 runs, 4830 frames, 0 frame errors`; `meta.json` has the commit and `"dirty": false`;
`flags identical on 4830/4830 frames`. **If the flags differ: BLOCKED** (the baseline would not see the same inputs as
the frozen methods).

- [ ] **Step 3: Intervals including the new baseline**

```bash
uv run python scripts/bootstrap_ci.py --tags test_final test_checks_only \
    --methods B1 B2 B3 ours_detect ours_full checks_only --reference ours_full --out report_figures/bootstrap
```
Expected: rows for all six methods and five differences. Write the checks-only F1, false alarm, unnecessary and
missed fallback, and the `ours_full - checks_only` F1 interval into the run log.

---

### Task 7: The 4B model with the VLM gate

**Files:**
- Modify: `src/recoverdrive/metrics.py` (`latency_estimates`)
- Modify: `scripts/make_figures.py` (use `latency_estimates`; the numbers must not change)
- Modify: `scripts/latency_report.py` (`--out` folder, default `outputs/e4`; `--baselines none` skips the baseline section)
- Test: `tests/test_metrics_runner.py`

**Interfaces:**
- Produces: `latency_estimates(pre_vlm_mean: float, flagged_mean: float, unflagged_mean: float, flag_share: float) -> dict`.
  It returns `{"full": pre + share * flagged + (1 - share) * unflagged, "gated": pre + share * flagged}`: the
  re-weighting that `make_figures.py` already does inline.

- [ ] **Step 1: Failing test**

```python
def test_latency_estimates():
    est = latency_estimates(pre_vlm_mean=1.0, flagged_mean=4.0, unflagged_mean=3.0, flag_share=0.25)
    assert est == {"full": pytest.approx(4.25), "gated": pytest.approx(2.0)}
```

- [ ] **Step 2: Implement it and use it in `make_figures.py`.** Verify that the 8B numbers did not move:
  `uv run python scripts/make_figures.py` prints `full_latency_test_estimate` and `gated_latency_test_estimate`
  equal to the values already in `report_figures/summary.json` (about 3.8 s and 2.6 s).

- [ ] **Step 3: Accuracy of 4B + gate (offline, CPU)** — `uv run python scripts/replay_ablation.py --tag test_abl_4b`.
  Expected: `reproduces 4830/4830`. It writes `outputs/eval/test_abl_4b/ablation_replay.{md,json}` with the
  `ours_full[gate_flags]` row. `frames_without_vlm` should be about 0.70, as for 8B: the flags do not depend on the model.

- [ ] **Step 4: Timing runs** (GPU, about 1 h). Run 8B and 4B back to back on the same card, so that their latencies
  are comparable (the 2026-11-03 E4 numbers were taken under a different GPU load).

```bash
nvidia-smi                                     # card A for vLLM (>= 24 GB free), card B for SAM 3 (>= 4 GB)
test ! -e outputs/cache/e4_ours_8b_v2.sqlite && test ! -e outputs/cache/e4_ours_4b.sqlite
setsid nohup env CUDA_VISIBLE_DEVICES=<A> VLM_GPU_UTIL=0.5 ./scripts/serve_vlm.sh > outputs/logs/vllm_8b.log 2>&1 &
# wait until: curl -sf http://127.0.0.1:8000/health   (2-5 min)
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split dev --scenes scene-0061 --faults C1 L1 \
    --methods ours_full --seeds 0 --workers 1 --sequential --no-precompute \
    --cache outputs/cache/e4_ours_8b_v2.sqlite --vlm-url http://127.0.0.1:8000/v1 --tag e4_ours_8b_v2
pkill -INT -f "[v]llm serve"
setsid nohup env CUDA_VISIBLE_DEVICES=<A> VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct VLM_SERVED_NAME=qwen3-vl-4b-instruct \
    VLM_GPU_UTIL=0.4 ./scripts/serve_vlm.sh > outputs/logs/vllm_4b.log 2>&1 &
# wait until: curl -sf http://127.0.0.1:8000/health
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split dev --scenes scene-0061 --faults C1 L1 \
    --methods ours_full --seeds 0 --workers 1 --sequential --no-precompute \
    --cache outputs/cache/e4_ours_4b.sqlite --vlm-url http://127.0.0.1:8000/v1 \
    --vlm-model qwen3-vl-4b-instruct --tag e4_ours_4b
pkill -INT -f "[v]llm serve"
uv run python scripts/latency_report.py --ours e4_ours_8b_v2 --baselines none --out outputs/e4_8b_v2
uv run python scripts/latency_report.py --ours e4_ours_4b --baselines none --out outputs/e4_4b
```
Expected: both runs `0 frame errors`; `outputs/e4/latency.json` (2026-11-03) is unchanged. In the run log, record
the mean seconds per frame for 8B and 4B, and the test-weighted `full` / `gated` estimates from `latency_estimates`
(flag share from Step 3).

- [ ] **Step 5: Verify and commit** — `uv run pytest -q && uv run ruff check src tests scripts`;
  `git commit -m "feat: latency estimates helper; 4B timing support"`

---

### Task 8: Report tables and documentation

**Files:**
- Modify: `scripts/make_figures.py`, `docs/superpowers/runs/results-provenance.md`, `FutureWork.md`
- Local only: `REPORT.md`

- [ ] **Step 1: Extend `make_figures.py`.** Add these sections to `report_figures/tables.md` and keys to
  `summary.json` (key `"step1"`):
  1. "E1/E2 with the checks-only baseline": the main table (mean ± std over seeds) with a `checks_only` row (from
     `test_checks_only`).
  2. "Easy vs. hard faults": `difficulty_table` for B1, B2, B3, ours_detect, ours_full, checks_only.
  3. "Bootstrap 95 % intervals": from `report_figures/bootstrap.json`.
  4. "Model size and the VLM gate": 8B, 8B + gate, 4B, 4B + gate. Columns: F1, false alarm, unnecessary, missed,
     VLM calls per frame (from the replays), and the test-weighted latency (`latency_estimates` with
     `outputs/e4_8b_v2` and `outputs/e4_4b`).
- [ ] **Step 2: Run it and check that nothing old moved**

```bash
cp report_figures/summary.json report_figures/summary.before.json   # before editing make_figures.py
uv run python scripts/make_figures.py
uv run python -c "import json; a=json.load(open('report_figures/summary.before.json')); \
b=json.load(open('report_figures/summary.json')); b.pop('step1'); print('old keys unchanged:', a == b)"
grep -c '^## ' report_figures/tables.md
```
Expected: `old keys unchanged: True`; the section count is 4 higher than before.
- [ ] **Step 3: Provenance.** Add rows for `dev_checks_only`, `test_checks_only`, `report_figures/bootstrap*`, the
  `test_abl_4b` replay, `e4_ours_8b_v2` and `e4_ours_4b`. Their commits are now recorded in each `meta.json` / output
  JSON ("Recorded").
- [ ] **Step 4: `FutureWork.md`.** Mark Step 1 as done, with one line per item pointing to its table.
  **`REPORT.md`** (local): add the checks-only row and the easy/hard and interval tables to §4, the 4B latency to §6,
  and the 4B + gate row to §7. Update §8 if the checks-only result changes a conclusion. In particular, state how much
  F1 and sensor accuracy the VLM adds over the rules.
- [ ] **Step 5: Final verification** — `uv run pytest -q && uv run ruff check src tests scripts`; no vLLM running;
  `nvidia-smi` shows none of our processes.
- [ ] **Step 6: Commit** — `git commit -m "docs: Step 1 results (checks-only, easy/hard, intervals, 4B + gate)"`

---

## Decisions taken as defaults

- **DEFAULT (D1)** Checks-only isolates every flagged sensor with confidence 1.0, i.e. it trusts the rules fully, like
  the controller's existing "agent failed → trust the flags" path. It blames the sensor with the most flags.
- **DEFAULT (D2)** Agent faults have no effect on methods without a VLM: they count as recovered in E2 (the same
  convention as the VLM gate) and as misses in E1.
- **DEFAULT (D3)** The easy set is L4, high-severity L1, A1 and A3. A2 is hard, because it needs the vote.
- **DEFAULT (D4)** Bootstrap units are fault episodes and clean gaps, stratified by (run fault, fault/clean), with 2000
  replicates and a percentile 95 % interval. Seeds are pooled, so point estimates differ slightly from the
  mean-over-seeds numbers. Frames inside a block are dependent; blocks of one run are treated as independent. This is
  a stated limitation.
- **DEFAULT (D5)** 8B is timed again in the same session as 4B. The 2026-11-03 E4 numbers stay in the report as
  measured.
- **DEFAULT (D6)** If `compare_flags` finds any difference, stop (BLOCKED) instead of reporting the baseline.

## GPU budget

| Task | GPU need | Time (estimate) |
|---|---|---|
| 3 (dev checks-only, seeds 0–2) | SAM 3 only (~3 GB) | 30–60 min |
| 6 (test checks-only) | SAM 3 only, mostly cache hits | 10–20 min |
| 7 (timing runs) | one card ≥ 24 GB for vLLM + one card ≥ 4 GB for SAM 3 | about 1 h |
| all other tasks | none | — |

## How to run it unattended

Commit this plan first. Then start a fresh session in the repo and use the project's `/goal` template, for example:

```
/goal 使用 autonomous-run skill 完成 docs/superpowers/plans/2026-11-15-step1-report-completeness.md。事先授權：可以在 test split 上執行 checks_only（Task 6，只跑一次）；可以啟動與停止 Qwen3-VL-8B 和 Qwen3-VL-4B 的 vLLM server；可以在 dev split 上跑 checks-only 和計時。對話中要出現以下證據：所有 plan 任務都有 checkpoint commit；`uv run pytest -q` 全過且 `uv run ruff check src tests scripts` 沒有錯誤；replay check 4830/4830（test_final 與 test_abl_4b）；compare_flags 4830/4830；report_figures/tables.md 有四個新段落；vLLM 已關閉；最後一次列出所有 deferred questions。或者當 Claude 送出以 BLOCKED: 開頭的訊息等我決定時就停止。或在 80 個 turn 後停止並回報進度。
```
