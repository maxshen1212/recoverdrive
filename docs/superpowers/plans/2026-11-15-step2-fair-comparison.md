# Step 2: Fair Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline) or
> superpowers:subagent-driven-development to implement this plan task-by-task. Unattended runs (`/goal`) also follow
> the project skill `.claude/skills/autonomous-run/SKILL.md`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the comparison with the baselines fair. This means a learned rules baseline (a small classifier on the
check numbers), DriveAgent-style chains that get the same evidence and the same prompt-tuning budget as Ours, a
PRAM-R-style evidence variant, and a larger closed-loop test in CARLA.

**Architecture:**
- The classifier is a second VLM-free monitor. It uses the rules-only recovery path from Step 1 and is trained only on
  dev runs.
- Text-chain variants are selected by method name (`B<k>[_tuned][_checks]`). The caption stage stays unchanged, so
  every variant reuses the cached captions.
- The PRAM-R-style variant is an evidence ablation of Ours: the same images, VLM, recovery and shield, with the
  cross-sensor report replaced by single-sensor quality indicators.
- Baseline settings go in a new `configs/baselines.yaml`, so `configs/eval.yaml` (Ours, frozen) stays untouched.
- E3 gets a second front-camera fault and twice as many runs.

**Tech Stack:** Python 3.12 (uv), numpy, scikit-learn (already a dependency), joblib, scipy, pytest, ruff; vLLM
(Qwen3-VL-8B-Instruct), SAM 3, CARLA 0.9.16.

**Spec:** `FutureWork.md` (section "Step 2"); design spec `docs/superpowers/specs/2026-11-03-recoverdrive-design.md`
(§5.1 methods, §5.2 metrics, §5.4 rules); `docs/related-work.md` (PRAM-R).

**Prerequisite:** the Step 1 plan (`2026-11-15-step1-report-completeness.md`) is done. This plan uses its
`git_state`, `RecoveryController.step_suspects`, `rank_sensors`, `rules_outcome`, `ChecksOnly`, `VLM_FREE_METHODS` /
`needs_vlm`, `compare_flags.py`, `bootstrap.py` / `bootstrap_ci.py`, and the `dev_checks_only` runs (dev, all
faults, seeds 0–2).

## Global Constraints

- Ours (`ours_detect`, `ours_full`) does not change, and `configs/eval.yaml` is not edited.
- Every baseline setting (classifier model and threshold, text-chain prompt versions, quality reference ranges) is
  chosen on **dev only** and committed before that baseline's single test run. Each test run needs `frozen: true` and
  a clean tracked tree (`run_eval.py` enforces both).
- Tuned text chains change only the final text-chain call. The caption prompt and caption settings stay as they are,
  so captions come from the cache.
- Shared GPUs: run `nvidia-smi` before each GPU job and keep ≥ 9 GB free on every card used; `VLM_GPU_UTIL` ≤ 0.5; stop
  every vLLM and CARLA server at the end.
- Jobs longer than about 10 minutes (test runs, E3) run detached (`setsid nohup ... > outputs/logs/<name>.log 2>&1 &`)
  or in the background, and are then polled. Servers are ready when `curl -sf http://127.0.0.1:<port>/health`
  succeeds (2–5 min after start).
- Do not edit the design spec. Do not push. `outputs/`, `REPORT.md` and `report_figures/` stay local. All repo files in
  English. Run `source scripts/env.sh` in every shell.

## Review Focus

1. **Caption cache misses.** If any variant changes the caption request, every frame needs six new captions and the
   cost explodes. Expected: caption requests are byte-identical across `B1`, `B1_checks` and `B1_tuned`. Test in Task 4.
2. **Classifier leakage.** The classifier must never see a test scene. Expected: the dataset loader raises on any
   non-dev scene. Test in Task 1.
3. **"Same evidence" means the same text.** `B*_checks` must receive exactly `ctx.summary`, the report Ours receives.
   Test in Task 4.
4. **Quality indicators on degenerate inputs:** an empty point cloud (L4), an all-black image (C3) or a frozen image.
   Expected: no division by zero, and values marked as outside the normal range. Tests in Task 6.
5. **E3 fault window.** C3 must hit only CAM_FRONT and only inside the hazard stretch, exactly like C1. Test in Task 7.

---

### Task 1: Features from the check report

**Files:**
- Create: `src/recoverdrive/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces:
  - `feature_names(cameras: list[str]) -> list[str]`, `report_features(report: dict, cameras: list[str], nominal_points: int, n_sectors: int) -> np.ndarray`
    (float32, same order as `feature_names`). `report` is the logged dict form of `CheckReport` (online callers pass
    `recording.to_jsonable(report)`). The features are:
    - per camera `c`, in config order: `c:match_rate` (−1 if None), `c:has_match_rate`, `c:n_gt`, `c:n_det`,
      `c:n_matched`, `c:match_rate_delta` (0 if None), `c:has_delta`, and the 0/1 flag columns `c:frozen`,
      `c:low_match`, `c:match_drop` (10 per camera).
    - LiDAR: `lidar:points_ratio` (points / nominal), `lidar:points_delta_ratio` (delta / previous count; 0 if None or
      the previous count is 0), `lidar:has_delta`, `lidar:sector_min_ratio` (min / median sector count; 0 without
      points), `lidar:sector_min_nominal` (min sector / (nominal / n_sectors)), `lidar:visible_ratio` (−1 if None),
      `lidar:n_in_range`, `lidar:agreement` (−1 if None), `lidar:agreement_n`, and the 0/1 flag columns
      `lidar:low_points`, `lidar:point_drop`, `lidar:sector_gap`, `lidar:low_visibility`, `lidar:camera_disagree`
      (14 LiDAR features, 74 in total for 6 cameras).
    - A camera that is missing from `report["cameras"]` gets the values of a camera with nothing to check
      (`n_gt = n_det = n_matched = 0`, match rate and delta missing).
  - `frame_target(label: dict) -> str`: the faulty sensor for sensor faults; `"none"` for clean frames and agent faults.
  - `load_dataset(tags: list[str], allowed_scenes: list[str], cameras: list[str], nominal_points: int, n_sectors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]`.
    It returns `X`, `y` (targets), `groups` (scene names) and the source records.
    It raises `ValueError` if any record's scene is not in `allowed_scenes`.

- [ ] **Step 1: Write the failing tests**

```python
def test_feature_vector_shape_and_missing_values():
    cams = ["CAM_FRONT", "CAM_BACK"]
    rep = {"cameras": {"CAM_FRONT": {"camera": "CAM_FRONT", "n_gt": 0, "n_det": 3, "n_matched": 0, "match_rate": None,
                                     "match_rate_delta": None},
                       "CAM_BACK": {"camera": "CAM_BACK", "n_gt": 4, "n_det": 5, "n_matched": 1, "match_rate": 0.25,
                                    "match_rate_delta": -0.5}},
           "lidar_visible_ratio": None, "point_count": 17000, "point_count_delta": -17000, "flags":
           ["camera:CAM_BACK:low_match", "lidar:point_drop"], "lidar_n_in_range": 0, "lidar_sector_counts": [0] * 12,
           "lidar_camera_agreement": None, "lidar_camera_n": 0}
    x = dict(zip(feature_names(cams), report_features(rep, cams, 34000, 12)))
    assert len(x) == 10 * 2 + 14
    assert x["CAM_FRONT:match_rate"] == -1 and x["CAM_FRONT:has_match_rate"] == 0
    assert x["CAM_BACK:low_match"] == 1 and x["lidar:point_drop"] == 1
    assert x["lidar:points_ratio"] == pytest.approx(0.5) and x["lidar:points_delta_ratio"] == pytest.approx(-0.5)
    assert x["lidar:sector_min_ratio"] == 0 and x["lidar:visible_ratio"] == -1


def test_frame_target():
    assert frame_target({"fault_id": "A2", "sensor": "agent", "severity": "low", "episode_id": 0}) == "none"
    assert frame_target({"fault_id": "L1", "sensor": "lidar", "severity": "low", "episode_id": 0}) == "lidar"


def test_load_dataset_refuses_test_scenes(tmp_path):
    runs = tmp_path / "outputs/eval/t/runs"
    runs.mkdir(parents=True)
    rec = {"scene": "scene-0103", "frame_idx": 0, "run_fault": "C1", "seed": 0, "flags": [], "report": {},
           "label": {"fault_id": "none", "sensor": "none", "severity": None, "episode_id": None}}
    (runs / "scene-0103_C1_s0.jsonl").write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError):
        load_dataset(["t"], ["scene-0061"], ["CAM_FRONT"], 34000, 12, root=tmp_path / "outputs/eval")
```

`load_dataset` takes `root: Path = Path("outputs/eval")` as a last keyword argument so tests can point it elsewhere,
and it checks the scene of every record before computing any feature.

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`.

- [ ] **Step 5: Commit** — `git commit -m "feat: features from the check report"`

---

### Task 2: Train the check classifier on dev

**Files:**
- Create: `src/recoverdrive/classifier.py`, `scripts/train_check_classifier.py`, `configs/baselines.yaml`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `load_dataset`, `report_features` (Task 1).
- Produces:
  - `class CheckClassifier` with fields `model`, `classes: list[str]`, `tau: float`, `feature_names: list[str]`:
    - `predict(x: np.ndarray) -> tuple[str, float]`: the most probable non-"none" class and its probability when that
      probability is ≥ `tau`, else `("none", p_none)`.
    - `save(path: Path) -> str`: writes `path` with joblib and returns its sha256.
    - `CheckClassifier.load(path: Path) -> CheckClassifier`.
  - `choose_tau(proba: np.ndarray, y: np.ndarray, classes: list[str], clean: np.ndarray, max_clean_alarm: float = 0.10) -> float`.
    `clean` is a boolean mask of clean frames (label `fault_id == "none"`). It scans τ ∈ {0.30, 0.35, …, 0.95}.
    Among the τ whose clean-frame alarm rate is ≤ `max_clean_alarm`, it returns the one with the highest
    correct-sensor recall on frames with `y != "none"`; ties go to the lowest τ. This is the same budget as
    `scripts/calibrate.py`. If no τ meets the budget, it returns 0.95.
  - `fit_check_classifier(X, y, groups, clean, grid: list[dict]) -> tuple[CheckClassifier, list[dict]]`:
    - Leave-one-scene-out CV for each grid entry, using
      `HistGradientBoostingClassifier(random_state=0, class_weight="balanced", max_iter=200, **entry)`.
    - Picks the entry with the best out-of-fold recall at its own τ (ties go to the lower clean alarm rate).
    - Refits that entry on all data and returns the classifier and one CV row per entry.
  - `GRID = [{"max_depth": 3, "learning_rate": 0.1}, {"max_depth": 3, "learning_rate": 0.05}, {"max_depth": None, "learning_rate": 0.1}, {"max_depth": None, "learning_rate": 0.05}]`.
    Four configurations, the same number of tries Ours had for its prompt.
  - `check_model_pin(path: Path, sha256: str) -> None`: raises `ValueError` if the file's sha256 differs.
  - `scripts/train_check_classifier.py --tags dev_checks_only --out outputs/models/check_classifier`:
    - Writes `.joblib` and a `.json` (CV table, chosen entry, τ, sha256, feature names, `git_state`).
    - Writes `check_classifier: {path: ..., sha256: ...}` into `configs/baselines.yaml`, creating the file with a
      header comment that says these are baseline settings chosen on dev.

- [ ] **Step 1: Write the failing tests**

```python
def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(["none", "lidar", "camera:CAM_FRONT"], n)
    X = np.c_[(y == "lidar") + rng.normal(0, 0.1, n), (y == "camera:CAM_FRONT") + rng.normal(0, 0.1, n)]
    groups = np.array([f"scene-{i % 3}" for i in range(n)])
    return X, y, groups, y == "none"


def test_fit_and_predict_separable():
    X, y, groups, clean = _toy()
    clf, rows = fit_check_classifier(X, y, groups, clean, GRID[:1])
    assert clf.predict(np.array([1.0, 0.0]))[0] == "lidar" and clf.predict(np.array([0.0, 0.0]))[0] == "none"
    assert len(rows) == 1 and rows[0]["recall"] > 0.9


def test_choose_tau_respects_clean_budget():
    classes = ["camera:CAM_FRONT", "lidar", "none"]
    proba = np.array([[0.0, 0.6, 0.4], [0.0, 0.9, 0.1], [0.0, 0.8, 0.2]])
    y = np.array(["none", "lidar", "lidar"])
    tau = choose_tau(proba, y, classes, clean=np.array([True, False, False]), max_clean_alarm=0.0)
    assert 0.6 < tau <= 0.8


def test_save_load_and_pin(tmp_path):
    X, y, groups, clean = _toy()
    clf, _ = fit_check_classifier(X, y, groups, clean, GRID[:1])
    sha = clf.save(tmp_path / "m.joblib")
    assert CheckClassifier.load(tmp_path / "m.joblib").predict(np.array([0.0, 1.0]))[0] == "camera:CAM_FRONT"
    check_model_pin(tmp_path / "m.joblib", sha)
    with pytest.raises(ValueError):
        check_model_pin(tmp_path / "m.joblib", "0" * 64)
```

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`.

- [ ] **Step 5: Train (CPU, minutes)**

```bash
uv run python scripts/train_check_classifier.py --tags dev_checks_only --out outputs/models/check_classifier
```
Expected: a CV table with 4 rows (recall, clean alarm rate, τ); the chosen entry's out-of-fold clean alarm ≤ 0.10;
`configs/baselines.yaml` now holds the path and sha256. Record the CV table in the run log.

- [ ] **Step 6: Commit** (with `configs/baselines.yaml`) — `git commit -m "feat: check classifier trained on dev"`

---

### Task 3: Classifier monitor and its test run

**Files:**
- Modify: `src/recoverdrive/methods.py` (`ClassifierMonitor`, `build_methods`, `VLM_FREE_METHODS`)
- Modify: `src/recoverdrive/metrics.py` (`RECOVERY_METHODS` gets `"check_classifier"`)
- Modify: `scripts/run_eval.py`:
  - Load `configs/baselines.yaml` into `cfg["baselines"]` when it exists, and record it in meta.
  - `--classifier PATH` overrides the model path.
  - Call `check_model_pin` before running `check_classifier`.
  - For the test split, refuse to run if `configs/baselines.yaml` is not tracked: use `require_tracked`.
- Modify: `src/recoverdrive/runner.py` (`require_tracked(path: Path, repo_root: Path) -> None`, which raises
  `RuntimeError` unless `git ls-files --error-unmatch` succeeds)
- Test: `tests/test_methods.py`, `tests/test_metrics_runner.py`

**Interfaces:**
- Consumes: `CheckClassifier` (Task 2), `report_features` (Task 1), `step_suspects` / `rank_sensors` /
  `rules_outcome` (Step 1).
- Produces: `class ClassifierMonitor` with `names = ("check_classifier",)`,
  `__init__(self, cfg: dict, classifier: CheckClassifier)`. Its `step(ctx)` does this:
  - `x = report_features(to_jsonable(ctx.check.report), cams, nominal, n_sectors)`, then
    `pred, p = classifier.predict(x)`.
  - `flagged = set(rank_sensors(flags))`; `suspects = {pred}` if `pred != "none"`, else an empty set.
  - `result = controller.step_suspects(suspects, p, flagged, ["classifier"])`.
  - The blamed sensor comes from `rules_outcome(result, [pred] + ranked)`. `n_calls = 0`.
  - `extra = {"pred": pred, "prob": p}`, plus `agent_recovered = True` under an agent fault.
  - A confidence below the controller's 0.6 behaves like a low-confidence VLM answer: the sensor stays pending for one
    frame.

- [ ] **Step 1: Write the failing tests**

```python
class StubClassifier:
    def __init__(self, pred, p):
        self.out = (pred, p)

    def predict(self, x):
        return self.out


def test_classifier_monitor_isolates_predicted_sensor():
    m = ClassifierMonitor({"recovery": {}, "data": {"nominal_lidar_points": 34000}, "thresholds": {"n_sectors": 12}},
                          StubClassifier("lidar", 0.9))
    out = m.step(make_ctx(flags=["lidar:low_points"]))["check_classifier"]
    assert out.recovery.isolated_sensors == ["lidar"] and out.pred_sensor == "lidar" and out.n_calls == 0


def test_classifier_monitor_none_is_healthy_even_with_flags():
    m = ClassifierMonitor({"recovery": {}, "data": {"nominal_lidar_points": 34000}, "thresholds": {"n_sectors": 12}},
                          StubClassifier("none", 0.8))
    out = m.step(make_ctx(flags=["camera:CAM_FRONT:low_match"]))["check_classifier"]
    assert out.recovery.final_state == "healthy" and not out.alarm


def test_require_tracked(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    (repo / "a.yaml").write_text("x: 1")
    git("add", "a.yaml")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    require_tracked(repo / "a.yaml", repo)
    (repo / "b.yaml").write_text("y: 2")
    with pytest.raises(RuntimeError):
        require_tracked(repo / "b.yaml", repo)
```

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`.

- [ ] **Step 5: Dev smoke, then the test run** (no vLLM; SAM 3 from the cache)

```bash
CUDA_VISIBLE_DEVICES=<g> uv run python scripts/run_eval.py --split dev --scenes scene-0061 --faults all \
    --methods check_classifier checks_only --seeds 0 --tag smoke_classifier
git status --porcelain --untracked-files=no      # must be empty
CUDA_VISIBLE_DEVICES=<g> uv run python scripts/run_eval.py --split test --faults all --methods check_classifier \
    --seeds 0 1 2 --tag test_classifier
uv run python scripts/compare_flags.py --a test_final --b test_classifier
```
Expected: 0 frame errors; `flags identical on 4830/4830 frames`; `meta.json` holds the model sha256. The dev smoke is
in-sample (the classifier was trained on dev), so its numbers are not reported.

- [ ] **Step 6: Commit** — `git commit -m "feat: classifier monitor; test run"`

---

### Task 4: Text chains with the check numbers

**Files:**
- Modify: `src/recoverdrive/vlm.py` (`textchain_messages` gains `report: str | None = None` and `version: str = "v1"`)
- Modify: `src/recoverdrive/methods.py` (`parse_textchain_name`, `TextChain`, `build_methods`)
- Test: `tests/test_methods.py`, `tests/test_render_vlm.py`

**Interfaces:**
- Produces:
  - `parse_textchain_name(name: str) -> tuple[str, bool, bool]`: `(variant, tuned, checks)` for names matching
    `^B([123])(_tuned)?(_checks)?$`, with variant ∈ {zero_shot, cot, self_refine}. It raises `ValueError` otherwise.
  - `TextChain(name: str, client, cfg: dict | None = None)` (`cfg=None` keeps today's tests valid):
    - `_checks` passes `report=ctx.summary`, the exact text Ours gets.
    - `_tuned` uses `cfg["baselines"]["textchain_prompt_version"]["B<k>"]`; untuned names use `"v1"`.
    - `cfg["textchain_version_override"]`, if set, replaces the tuned version (used by the tuning runs in Task 5).
    - The `build_methods` call changes from `TextChain(n, client)` to `TextChain(n, client, cfg)`.
  - With `report` given, `textchain_messages` puts this line before the final question:
    `"Consistency report from deterministic cross-checks of the LiDAR against the cameras:\n{report}\n"`.
    With `report=None` and `version="v1"`, the messages are byte-identical to today's.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_textchain_name():
    assert parse_textchain_name("B2_tuned_checks") == ("cot", True, True)
    assert parse_textchain_name("B1") == ("zero_shot", False, False)
    with pytest.raises(ValueError):
        parse_textchain_name("B4_checks")


def test_v1_messages_unchanged_and_report_added():
    base = textchain_messages("cot", "lidar text", "visual text", SENSORS)
    assert base == textchain_messages("cot", "lidar text", "visual text", SENSORS, report=None, version="v1")
    with_report = textchain_messages("cot", "lidar text", "visual text", SENSORS, report="REPORT-XYZ")
    assert "REPORT-XYZ" in with_report[1]["content"] and "REPORT-XYZ" not in base[1]["content"]


def test_checks_variant_sends_the_ours_summary_and_same_captions():
    calls = {}

    def reply(req):
        name = req["response_format"]["json_schema"]["name"]
        return {"description": "a car ahead"} if name == "caption" else \
            {"verdict": "ok", "fault_type": "none", "confidence": 0.9}

    for name in ("B1", "B1_checks"):
        fake = FakeOpenAI(reply)
        TextChain(name, VLMClient("http://x", "m", client=fake), {"baselines": {}}).step(make_ctx())
        calls[name] = fake.calls
    cap = [c["messages"] for c in calls["B1"] if c["response_format"]["json_schema"]["name"] == "caption"]
    cap_c = [c["messages"] for c in calls["B1_checks"] if c["response_format"]["json_schema"]["name"] == "caption"]
    assert cap == cap_c
    final = [c for c in calls["B1_checks"] if c["response_format"]["json_schema"]["name"] == "diagnosis"][0]
    assert ("Consistency report from deterministic cross-checks of the LiDAR against the cameras:\nsummary"
            in final["messages"][1]["content"])  # "summary" is make_ctx's summary text
```

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`.

- [ ] **Step 5: Dev smoke** (one 8B vLLM server; captions of dev scene-0553 are cached from the Phase 1–4 runs)

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=<A> VLM_GPU_UTIL=0.5 ./scripts/serve_vlm.sh > outputs/logs/vllm.log 2>&1 &
# wait until: curl -sf http://127.0.0.1:8000/health
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split dev --scenes scene-0553 --faults C1 L1 A1 \
    --methods B1_checks B2_checks B3_checks --seeds 0 --tag smoke_textchain_checks
```
Expected: 0 frame errors; in `summary.md` every variant has a JSON-valid share ≥ 0.95 (invalid answers show up as
`pred_sensor == "agent"` on clean frames).

- [ ] **Step 6: Commit, then the test run** (two 8B servers on ports 8000/8001 if two cards are free; about 4–6 h)

```bash
git commit -m "feat: DriveAgent-style chains with the check report (B*_checks)"
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split test --faults all \
    --methods B1_checks B2_checks B3_checks --seeds 0 1 2 --tag test_textchain_checks
```
Expected: 120 runs, 0 frame errors.

---

### Task 5: Tune the B1–B3 prompts on dev with Ours's budget

**Files:**
- Modify: `src/recoverdrive/vlm.py` (prompt versions v1–v4, `textchain_schema(variant, sensors, version="v1")`)
- Modify: `scripts/run_eval.py` (`--textchain-version vN` sets `cfg["textchain_version_override"]`)
- Create: `scripts/select_textchain_prompts.py`
- Modify: `configs/baselines.yaml` (`textchain_prompt_version`)
- Test: `tests/test_render_vlm.py`, `tests/test_methods.py`

**Interfaces and fixed prompt versions** (pre-registered, so tuning cannot drift; each adds one change):
- `v1`: today's DriveAgent prompts.
- `v2`: v1 plus the decision guidance `TEXTCHAIN_GUIDE` in the user message, before the question. It is the same
  domain knowledge that `OURS_GUIDE` gives Ours without check numbers:
  - A camera is faulty when its description does not fit its viewing direction, or when the image is black, washed
    out, blurred beyond recognition, tilted or frozen.
  - The LiDAR is faulty when it has far fewer points than usual, or misses objects that several cameras describe
    nearby.
  - Night, rain, glare, empty roads and small description/LiDAR differences are normal.
  - Answer ok unless the evidence clearly points to one sensor; use uncertain only when strong evidence conflicts.
- `v3`: v2 plus context the chain lacks. `textchain_messages` rewrites each line of `visual` that starts with
  `"<camera>: "` to start with `"<camera> (<CAMERA_VIEWS[camera]>): "`. With the new argument
  `nominal_points: int | None = None` it also ends the LiDAR text with
  `"nominal for this sensor about <nominal_points> points"`; `TextChain` passes `cfg["data"]["nominal_lidar_points"]`.
- `v4`: v3 plus Ours's answer format: `verdict`, `fault_type`, `confidence` first, then the reasoning fields (cot,
  self_refine), then a short `evidence` string.
- `select_version(rows: list[dict]) -> str`: the version with the highest F1; ties go to the lower false alarm, then
  the lower version number.
- `scripts/select_textchain_prompts.py --tags tc_v1 tc_v2 tc_v3 tc_v4 --write configs/baselines.yaml`:
  - Prints the F1 / false alarm / sensor accuracy table per variant × version on the tuning subset.
  - Writes `textchain_prompt_version: {B1: ..., B2: ..., B3: ...}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prompt_versions_add_one_change_each():
    v = {k: textchain_messages("cot", "L", "CAM_FRONT: a road", SENSORS, version=k)[1]["content"]
         for k in ("v1", "v2", "v3", "v4")}
    assert TEXTCHAIN_GUIDE not in v["v1"] and TEXTCHAIN_GUIDE in v["v2"]
    assert "facing forward" in v["v3"] and "facing forward" not in v["v2"]
    assert list(textchain_schema("cot", SENSORS, version="v4")["properties"])[:2] == ["verdict", "fault_type"]
    assert list(textchain_schema("cot", SENSORS, version="v3")["properties"])[0] == "reasoning"


def test_select_version():
    rows = [{"version": "v1", "F1": 0.6, "false_alarm": 0.3}, {"version": "v2", "F1": 0.7, "false_alarm": 0.4},
            {"version": "v3", "F1": 0.7, "false_alarm": 0.2}]
    assert select_version(rows) == "v3"
```

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`; commit.

- [ ] **Step 5: Tuning runs on dev.** Use the protocol of Ours's Phase 1 prompt tuning: scenes scene-0061,
  scene-0553, scene-1077; faults C1, L1; seed 0. That is about 240 frames per run; captions come from the cache.

```bash
for V in v1 v2 v3 v4; do
  CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split dev \
      --scenes scene-0061 scene-0553 scene-1077 --faults C1 L1 --seeds 0 \
      --methods B1_tuned B2_tuned B3_tuned --textchain-version $V --tag tc_$V
done
uv run python scripts/select_textchain_prompts.py --tags tc_v1 tc_v2 tc_v3 tc_v4 --write configs/baselines.yaml
```
Expected: `tc_v1` reproduces B1–B3 on that subset (mostly cache hits); a 4 × 3 table; three chosen versions. Put the
table in the run log. If a variant's best version is v1, its tuned test run equals the plain baseline: record
"tuning did not help" and skip that variant's test run.

- [ ] **Step 6: Commit `configs/baselines.yaml`, then the test run** (about 4–6 h with two servers)

```bash
git commit -m "feat: tuned B1-B3 prompt versions chosen on dev"
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split test --faults all \
    --methods B1_tuned B2_tuned B3_tuned --seeds 0 1 2 --tag test_textchain_tuned
```

- [ ] **Step 7 (optional, only with spare GPU time):** the fourth corner of the 2 × 2 design, "tuned + checks":
  `--methods B1_tuned_checks B2_tuned_checks B3_tuned_checks --tag test_textchain_tuned_checks`.

---

### Task 6: PRAM-R-style evidence (single-sensor quality indicators)

**Files:**
- Create: `src/recoverdrive/quality.py`, `scripts/quality_reference.py`, `configs/quality_reference.yaml`
- Modify: `src/recoverdrive/vlm.py` (`OURS_GUIDE_QUALITY`; `ours_messages(..., evidence: str = "checks")`)
- Modify: `src/recoverdrive/methods.py` (`Ours` with `cfg["ablation"] == "quality"`)
- Modify: `scripts/run_eval.py`:
  - `--ablation` choices gain `quality`.
  - Load `configs/quality_reference.yaml` into `cfg["quality_reference"]`.
  - For the test split, require that file to be tracked.
- Test: `tests/test_quality.py`, `tests/test_methods.py`

**Interfaces:**
- Produces:
  - `camera_quality(image: np.ndarray) -> dict[str, float]` with these keys:
    - `brightness`: mean gray, 0–255.
    - `contrast`: std of gray.
    - `edge_density`: share of pixels whose max absolute neighbour difference in gray is > 20.
  - `lidar_quality(points: np.ndarray, n_sectors: int = 12, max_r: float = 50.0) -> dict[str, float]` with these keys:
    - `points`: point count.
    - `min_sector_share`: smallest sector count / mean sector count, 2–50 m; 0 without points.
    - `mean_intensity`: column 3; 0 without points.
  - `quality_report(frame: Frame, cameras: list[str], ref: dict) -> str`. One line per camera and one for the LiDAR,
    e.g. `"CAM_FRONT: brightness 112 (normal 64–150), contrast 51 (normal 38–70), edge density 0.083 (normal 0.040–0.120)"`
    and `"LiDAR: 34120 points (normal 33500–34800), smallest sector 0.71 of the mean (normal 0.55–0.90), mean intensity 18.2 (normal 12.0–25.0)"`.
    - Number formats: brightness, contrast and points with no decimals; edge density with 3 decimals; sector share
      with 2; intensity with 1.
    - The normal range is the dev p5–p95. Every value outside it gets `" (outside the normal range)"` right after its
      range.
  - `scripts/quality_reference.py --split dev --out configs/quality_reference.yaml`: runs over every keyframe of every
    dev scene, without faults. It writes p5 / median / p95 for each camera name × indicator and for each LiDAR
    indicator, plus `git_state`, and refuses `--split test`.
  - `Ours` with ablation `quality`:
    - Builds the report once per frame with `quality_report(ctx.frame, cams, cfg["quality_reference"])`.
    - Sends it with `evidence="quality"` (header `"Sensor quality report:"` + `OURS_GUIDE_QUALITY`).
    - Uses `flagged = set()`: no cross-sensor flags reach recovery, as in the `no_checks` ablation.
    - Recovery, voting and the shield are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_camera_quality_black_and_noise():
    black = camera_quality(np.zeros((90, 160, 3), np.uint8))
    assert black == {"brightness": 0.0, "contrast": 0.0, "edge_density": 0.0}
    noisy = camera_quality(np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8))
    assert noisy["edge_density"] > 0.5


def test_lidar_quality_empty_and_sector_gap():
    assert lidar_quality(np.zeros((0, 4))) == {"points": 0.0, "min_sector_share": 0.0, "mean_intensity": 0.0}
    ang = np.radians(np.arange(0, 360, 0.5))
    ring = np.c_[10 * np.cos(ang), 10 * np.sin(ang), np.zeros_like(ang), np.full_like(ang, 5.0)]
    assert lidar_quality(ring[(np.degrees(ang) < 300)])["min_sector_share"] == 0.0


REF = {"CAM_FRONT": {k: {"p5": 50.0, "median": 100.0, "p95": 150.0} for k in ("brightness", "contrast")}
       | {"edge_density": {"p5": 0.04, "median": 0.08, "p95": 0.12}},
       "lidar": {"points": {"p5": 33000, "median": 34000, "p95": 35000},
                 "min_sector_share": {"p5": 0.5, "median": 0.7, "p95": 0.9},
                 "mean_intensity": {"p5": 10, "median": 18, "p95": 25}}}


def test_quality_report_marks_out_of_range():
    text = quality_report(make_ctx().frame, ["CAM_FRONT"], REF)  # make_ctx: black 9x16 image, no points
    assert "CAM_FRONT: brightness 0 (normal 50–150) (outside the normal range)" in text
    assert text.count("\n") == 1 and "LiDAR: 0 points" in text


def test_quality_ablation_prompt_and_no_flags():
    fake = FakeOpenAI(GOOD)
    cfg = {"recovery": {}, "ablation": "quality", "quality_reference": REF}
    out = Ours(VLMClient("http://x", "m", client=fake), cfg).step(make_ctx(flags=["lidar:low_points"]))["ours_full"]
    text = fake.calls[0]["messages"][1]["content"][-1]["text"]
    assert "Sensor quality report" in text and "Consistency report" not in text
    assert out.extra["flagged"] == []
```

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement**; **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`.

- [ ] **Step 5: Reference ranges (CPU), dev smoke, commit**

```bash
uv run python scripts/quality_reference.py --split dev --out configs/quality_reference.yaml
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split dev --scenes scene-0061 --faults C1 L1 \
    --methods ours_detect ours_full --ablation quality --seeds 0 --tag smoke_quality
git add configs/quality_reference.yaml && git commit -m "feat: PRAM-R-style quality-indicator evidence for Ours"
```
Expected: the reference file lists the 6 cameras and the LiDAR; the smoke has 0 frame errors and valid JSON.

- [ ] **Step 6: Test run** (8B, about 3 h with two servers)

```bash
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_eval.py --split test --faults all \
    --methods ours_detect ours_full --ablation quality --seeds 0 1 2 --tag test_abl_quality
```

---

### Task 7: E3 with a second front-camera fault and twice the runs

**Files:**
- Modify: `src/recoverdrive/faults.py` (`apply_e3_condition`)
- Modify: `scripts/run_e3.py` (use `apply_e3_condition`; `--conditions` accepts `C3`; collision tests in the summary)
- Modify: `src/recoverdrive/metrics.py` (`fisher_collisions`)
- Test: `tests/test_faults.py`, `tests/test_metrics_runner.py`

**Interfaces:**
- Produces:
  - `apply_e3_condition(frame: Frame, condition: str, progress_m: float, trigger_m: float) -> tuple[Frame, FaultLabel]`.
    The fault is active iff `trigger_m - 25 <= progress_m <= trigger_m + 30`, the window today's inline C1 code uses.
    - `"C1"`: CAM_FRONT shows the CAM_BACK image, with label `FaultLabel("C1", "camera:CAM_FRONT", "high", 0)`.
    - `"C3"`: CAM_FRONT is all black (`inject_c3(..., "black", None, rng)`), with label
      `FaultLabel("C3", "camera:CAM_FRONT", "high", 0)`.
    - `"clean"`, or outside the window: the frame is unchanged, with label `FaultLabel.none()`.
  - `fisher_collisions(results: list[dict], a: str, b: str, condition: str) -> dict`.
    It returns `{"condition", "a", "b", "a_collisions", "a_runs", "b_collisions", "b_runs", "p"}`, with p from the
    two-sided `scipy.stats.fisher_exact`.
  - `run_e3.py --summarize` appends a table of `fisher_collisions(results, "B0", "ours_full", cond)` for each faulty
    condition.

- [ ] **Step 1: Write the failing tests**

The helper `two_camera_frame()` in the test file builds a `Frame` with CAM_FRONT filled with 1s, CAM_BACK filled
with 2s (both 9 × 16 × 3) and an empty point cloud.

```python
def test_apply_e3_condition_window_and_target():
    frame = two_camera_frame()
    f, lab = apply_e3_condition(frame, "C3", progress_m=15.0, trigger_m=40.0)
    assert lab.fault_id == "C3" and not f.cameras["CAM_FRONT"].image.any()
    assert np.array_equal(f.cameras["CAM_BACK"].image, frame.cameras["CAM_BACK"].image)
    assert apply_e3_condition(frame, "C3", 14.9, 40.0)[1].fault_id == "none"
    assert apply_e3_condition(frame, "C1", 70.0, 40.0)[1].fault_id == "C1"
    assert apply_e3_condition(frame, "C1", 70.1, 40.0)[1].fault_id == "none"
    assert apply_e3_condition(frame, "clean", 40.0, 40.0)[1].fault_id == "none"


def test_fisher_collisions_matches_report():
    res = [{"method": "B0", "condition": "C1", "collision": i < 7} for i in range(12)]
    res += [{"method": "ours_full", "condition": "C1", "collision": False} for _ in range(12)]
    row = fisher_collisions(res, "B0", "ours_full", "C1")
    assert row["a_collisions"] == 7 and row["b_collisions"] == 0 and row["p"] == pytest.approx(0.00458, abs=1e-4)
```

- [ ] **Step 2: Run and see them fail**; **Step 3: Implement.** `run_e3.py` must call `apply_e3_condition` exactly
  where the inline C1 code was, so C1 behavior does not change.
- [ ] **Step 4:** `uv run pytest -q && uv run ruff check src tests scripts`; commit.

- [ ] **Step 5: CARLA smoke** (one CARLA server, one vLLM server on port 8002 as in `configs/e3.yaml`)

```bash
CARLA_PORT=2000 setsid nohup ./scripts/carla_server.sh > outputs/logs/carla0.log 2>&1 &
setsid nohup env CUDA_VISIBLE_DEVICES=<A> VLM_PORT=8002 VLM_GPU_UTIL=0.5 ./scripts/serve_vlm.sh \
    > outputs/logs/vllm_e3.log 2>&1 &
# wait until: curl -sf http://127.0.0.1:8002/health   (CARLA needs about 1 min to accept clients)
CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_e3.py --methods B0 ours_full --conditions C3 --seeds 0 \
    --only-route 1 --tag smoke_e3_c3
```
Expected: B0 runs into the hazard and ours_full isolates CAM_FRONT within the fault window (`isolated` in the JSONL).

- [ ] **Step 6: Full run** (about 10 h). The run covers 3 methods × 3 conditions × 4 routes × 6 seeds = 216 runs,
  all under one commit and one tag. Use three CARLA servers, each with its own seeds; runs are resumable and write
  separate files.

```bash
for i in 0 1 2; do
  CARLA_PORT=$((2000 + 3 * i)) setsid nohup ./scripts/carla_server.sh > outputs/logs/carla$i.log 2>&1 &
done
for i in 0 1 2; do
  setsid nohup env CUDA_VISIBLE_DEVICES=<B> uv run python scripts/run_e3.py --methods B0 B1 ours_full \
      --conditions clean C1 C3 --seeds $((2 * i)) $((2 * i + 1)) --port $((2000 + 3 * i)) --tm-port $((8100 + i)) \
      --tag e3_v2 > outputs/logs/e3_v2_$i.log 2>&1 &
done
# poll until all runs are done (about 10 h): ls outputs/e3/e3_v2/r*_s*.json | wc -l  ->  216
uv run python scripts/run_e3.py --summarize --tag e3_v2
```
Expected: 216 result files and a table with 24 runs per cell, plus Fisher p-values for C1 and C3. Stop every CARLA
server with `pkill -INT -f "[C]arlaUE4-Linux-Shipping"`. If a process dies, restart it with the same command: finished
runs are skipped.

---

### Task 8: Report tables and documentation

**Files:**
- Modify: `scripts/make_figures.py`, `docs/superpowers/runs/results-provenance.md`, `FutureWork.md`
- Local only: `REPORT.md`

- [ ] **Step 1: Bootstrap over every method** (CPU)

```bash
uv run python scripts/bootstrap_ci.py \
    --tags test_final test_checks_only test_classifier test_textchain_checks test_textchain_tuned \
           test_abl_no_checks=nochecks test_abl_quality=quality \
    --methods B1 B2 B3 B1_checks B2_checks B3_checks B1_tuned B2_tuned B3_tuned checks_only check_classifier \
              nochecks:ours_full quality:ours_full ours_detect ours_full \
    --reference ours_full --out report_figures/bootstrap_step2
```
(Add `test_textchain_tuned_checks` and its methods if Task 5 Step 7 ran; drop the tuned methods of any variant whose
best version was v1.)

- [ ] **Step 2: Extend `make_figures.py`** with new sections in `tables.md` and a `"step2"` key in `summary.json`:
  1. "Fair comparison": one row per method in Step 1's `--methods` list. Columns: F1, sensor accuracy, false alarm,
     unnecessary and missed fallback (mean ± std over seeds), plus the bootstrap interval of `ours_full - <method>` F1.
  2. "Evidence for the VLM": Ours-full with no report (`test_abl_no_checks`), with quality indicators
     (`test_abl_quality`), and with cross-sensor checks (`test_final`).
  3. "Closed loop v2": the `e3_v2` table, Fisher p-values, and `fig3` redrawn from `e3_v2`.
  Check that the old keys of `summary.json` did not change, as in Step 1 Task 8.
- [ ] **Step 3: Provenance rows** for every new tag; their commits are recorded in `meta.json`. Mark Step 2 as done in
  `FutureWork.md`, with one line per item. Update `REPORT.md` (local) §2, §4, §5, §8 and §9. In particular, state
  whether the gap to the baselines survives equal evidence and equal tuning.
- [ ] **Step 4: Final verification** — `uv run pytest -q && uv run ruff check src tests scripts`; no vLLM or CARLA
  running (`nvidia-smi`); commit `docs: Step 2 results (fair comparison, evidence, E3 v2)`.

---

## Decisions taken as defaults

- **DEFAULT (D1)** Baseline settings live in `configs/baselines.yaml`, so the frozen `configs/eval.yaml` stays
  byte-identical.
- **DEFAULT (D2)** The classifier is gradient boosting with a grid of 4 configurations (the same number of tries as
  Ours's prompt versions). Its threshold is chosen out-of-fold under the threshold calibration's clean-alarm budget
  (≤ 10 %), with leave-one-scene-out CV on the 6 dev scenes.
- **DEFAULT (D3)** Tuned prompts change only the final text-chain call. The captions stay as DriveAgent's caption
  stage, which keeps their cache valid.
- **DEFAULT (D4)** The tuning protocol is the one Ours had: 3 dev scenes, C1 and L1, seed 0, at most 4 versions
  including the original, selected by F1. The four versions are fixed in this plan before any tuning run.
- **DEFAULT (D5)** The "tuned + checks" corner is optional (only with spare GPU time).
- **DEFAULT (D6)** The PRAM-R-style variant is an evidence ablation of Ours: the same images, VLM, recovery and shield,
  and no cross-sensor flags. Its normal ranges come from clean dev frames. PRAM-R itself routes sensors for efficiency
  and has no recovery, so it is not re-implemented one to one.
- **DEFAULT (D7)** E3's second fault is C3 (CAM_FRONT all black during the hazard), not L1 or A2 as first listed in
  `FutureWork.md` (now updated to C3):
  - L1 does not affect the camera-based brake, so collisions cannot differ between methods.
  - A2's defense (the vote) is already measured offline in E5.
  - C3 tests whether the closed-loop safety result holds for a second front-camera failure.
  - If you prefer A2, the cost is a new agent-fault path in `run_e3.py` (agent faults are not wired into E3 today).
- **DEFAULT (D8)** E3 v2 reruns all 216 runs under one commit instead of mixing new runs with `e3_final`.

## Order and GPU budget

| Task | GPU need | Time (estimate) | Priority if time is short |
|---|---|---|---|
| 1–2 | none | — | 1 |
| 3 | SAM 3 only (cache) | < 30 min | 1 |
| 4 | 8B vLLM (1–2 servers) | smoke 30 min; test 4–6 h | 2 |
| 5 | 8B vLLM | tuning 2–3 h; test 4–6 h | 3 |
| 6 | 8B vLLM | smoke 20 min; test about 3 h | 4 |
| 7 | CARLA ×3 + 8B vLLM | smoke 30 min; full about 10 h | 5 |
| 8 | none | — | last |

Roughly 25–30 GPU-hours in total, spread over several days. Each task's test run is independent, so the plan can
stop after any task and still report what is finished.

## How to run it unattended

Commit this plan first. Then, after Step 1 is done, start a fresh session and use the project's `/goal` template, for
example:

```
/goal 使用 autonomous-run skill 完成 docs/superpowers/plans/2026-11-15-step2-fair-comparison.md（前提：Step 1 plan 已完成）。事先授權：可以在 test split 上各跑一次 check_classifier、B1–B3_checks、B1–B3_tuned、Ours 的 quality ablation（每個設定都要先在 dev 決定並 commit）；可以啟動與停止 Qwen3-VL-8B 的 vLLM server 與最多 3 個 CARLA server；可以在 dev split 上跑調參與 smoke test。對話中要出現以下證據：所有 plan 任務都有 checkpoint commit；`uv run pytest -q` 全過且 `uv run ruff check src tests scripts` 沒有錯誤；compare_flags 4830/4830（test_classifier）；configs/baselines.yaml 與 configs/quality_reference.yaml 在 test run 前已 commit；e3_v2 有 216 個 run；report_figures/tables.md 有三個新段落；vLLM 與 CARLA 已關閉；最後一次列出所有 deferred questions。或者當 Claude 送出以 BLOCKED: 開頭的訊息等我決定時就停止。或在 120 個 turn 後停止並回報進度。
```
