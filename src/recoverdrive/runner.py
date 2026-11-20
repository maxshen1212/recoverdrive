"""Evaluation runner: one run = (split, scene, fault, seed). Runs execute in parallel threads; frames within a run
are sequential because methods keep temporal state. Each finished run leaves a .done marker for resume."""

from __future__ import annotations

import json
import subprocess
import time
import traceback
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recoverdrive.cache import digest
from recoverdrive.checks import CheckConfig, CheckThresholds, run_checks
from recoverdrive.faults import AgentFaultState, FaultEvent, SensorFaultInjector, plan_faults
from recoverdrive.methods import FrameContext, build_methods
from recoverdrive.recording import JsonlWriter, to_jsonable
from recoverdrive.render import camera_mosaic, lidar_bev
from recoverdrive.vlm import image_part, summarize_checks


@dataclass(frozen=True)
class RunSpec:
    split: str
    scene: str
    fault_id: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.scene}_{self.fault_id}_s{self.seed}"


@dataclass
class EvalEnv:
    cfg: dict
    adapter: object  # NuScenesAdapter (or any adapter with scene_sample_tokens / load_frame)
    detector: object  # CachedDetector
    client: object | None  # VLMClient
    out_dir: Path
    methods: list[str]

    @property
    def cameras(self) -> list[str]:
        return self.cfg["data"]["cameras"]

    @property
    def check_cfg(self) -> CheckConfig:
        return CheckConfig(**self.cfg["checks"])

    @property
    def thresholds(self) -> CheckThresholds:
        return CheckThresholds(**self.cfg["thresholds"])


def run_rng(spec: RunSpec) -> np.random.Generator:
    return np.random.default_rng([spec.seed, zlib.crc32(spec.scene.encode()), zlib.crc32(spec.fault_id.encode())])


def frame_seed(seed: int, scene: str, frame_idx: int) -> int:
    """Seed for stochastic VLM sampling; independent of the fault so identical prompts share the cache."""
    return int(digest([seed, scene, frame_idx])[:7], 16)


def run_paths(out_dir: Path, spec: RunSpec) -> tuple[Path, Path]:
    jsonl = out_dir / "runs" / f"{spec.name}.jsonl"
    return jsonl, jsonl.with_suffix(".done")


def run_status(out_dir: Path, spec: RunSpec) -> str:
    jsonl, done = run_paths(out_dir, spec)
    if done.exists():
        return "done"
    return "partial" if jsonl.exists() else "missing"


def check_test_allowed(cfg: dict, repo_root: Path, git_status: Callable[[Path], str] | None = None) -> str:
    """The test split runs only from a frozen config and a clean git tree (spec §5.4). Returns the commit."""
    if not cfg.get("frozen", False):
        raise RuntimeError("refusing to run the test split: config is not frozen (set frozen: true after dev tuning)")

    def _status(root: Path) -> str:
        return subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                              capture_output=True, text=True, check=True).stdout

    if (git_status or _status)(repo_root).strip():
        raise RuntimeError("refusing to run the test split: git tree has uncommitted changes")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True,
                          check=False).stdout.strip()


def git_state(repo_root: Path) -> dict:
    """HEAD commit and whether tracked files differ from it (same rule as check_test_allowed). Never raises:
    outside a git repository both values are None."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root,
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def points_in_ego(frame) -> np.ndarray:
    """LiDAR points in the ego frame (x forward, y left), for the top view."""
    if frame.lidar is None or len(frame.lidar.points) == 0:
        return np.zeros((0, 4))
    m = frame.lidar.extrinsic.matrix()
    pts = frame.lidar.points
    return np.concatenate([pts[:, :3] @ m[:3, :3].T + m[:3, 3], pts[:, 3:4]], axis=1)


def sensors_of(cameras: list[str]) -> list[str]:
    return [f"camera:{c}" for c in cameras] + ["lidar"]


def run_one(spec: RunSpec, env: EvalEnv, plan: list[FaultEvent | None] | None = None,
            frames: range | None = None) -> dict:
    """Process every keyframe of one scene with one fault plan; returns a short summary.
    `plan` replaces the seeded fault plan and `frames` limits the keyframes processed (falsification windows);
    with both left as None the run is exactly the evaluation run."""
    jsonl, done = run_paths(env.out_dir, spec)
    if done.exists():
        return json.loads(done.read_text())
    cfg, cams = env.cfg, env.cameras
    tokens = env.adapter.scene_sample_tokens(spec.scene)
    if plan is None:
        plan = plan_faults(spec.fault_id, len(tokens), run_rng(spec), cfg["faults"], cams)
    injector = SensorFaultInjector(plan)
    methods = build_methods(env.methods, env.client, cfg)
    check_cfg, thr = env.check_cfg, env.thresholds
    writer = JsonlWriter(jsonl)
    originals: dict = {}
    prev_frame = prev_report = prev_mosaic = None
    errors = 0
    t_run = time.perf_counter()
    for idx in frames if frames is not None else range(len(tokens)):
        token = tokens[idx]
        try:
            lat = {}
            t0 = time.perf_counter()
            raw = env.adapter.load_frame(spec.scene, idx, token)
            originals[idx] = raw
            originals.pop(idx - 8, None)
            frame, label = injector.apply(raw, originals)
            lat["load"] = time.perf_counter() - t0
            agent_fault = AgentFaultState.from_event(plan[idx])
            t0 = time.perf_counter()
            dets = {c: env.detector.detect(frame.cameras[c].image) for c in cams}
            lat["sam3"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            check = run_checks(frame, prev_frame, prev_report, dets, check_cfg, thr, cams)
            summary = summarize_checks(check.report, check_cfg.max_range_m, check_cfg.iou_threshold,
                                       check_cfg.min_points_in_box, cfg["data"].get("nominal_lidar_points"))
            lat["checks"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            mosaic = image_part(camera_mosaic({c: frame.cameras[c].image for c in cams}))
            bev = image_part(lidar_bev(points_in_ego(frame)))
            lat["render"] = time.perf_counter() - t0
            ctx = FrameContext(frame, label, check, summary, mosaic, prev_mosaic, bev, sensors_of(cams),
                               frame_seed(spec.seed, spec.scene, idx), agent_fault, dets, prev_frame)
            outputs = {}

            def timed(m, ctx=ctx):
                t0 = time.perf_counter()
                return m, m.step(ctx), time.perf_counter() - t0

            if methods and cfg.get("sequential_methods"):  # E4: no contention between methods
                for m in methods:
                    _, out, dt = timed(m)
                    outputs.update(out)
                    lat["method:" + "+".join(m.names)] = dt
            elif methods:  # none = checks-only run (calibration)
                with ThreadPoolExecutor(len(methods)) as pool:  # methods are independent within a frame
                    for m, out, dt in pool.map(timed, methods):
                        outputs.update(out)
                        lat["method:" + "+".join(m.names)] = dt
            event = plan[idx]
            writer.write({
                "scene": spec.scene, "frame_idx": idx, "run_fault": spec.fault_id, "seed": spec.seed,
                "label": label, "event_params": event.params if event else None,
                "flags": check.report.flags, "report": check.report, "latency_s": lat,
                "methods": {name: to_jsonable(o) for name, o in outputs.items()},
            })
            prev_frame, prev_report, prev_mosaic = frame, check.report, mosaic
        except Exception as exc:  # noqa: BLE001 - record and keep going; counted as a frame error
            errors += 1
            writer.write({"scene": spec.scene, "frame_idx": idx, "run_fault": spec.fault_id, "seed": spec.seed,
                          "error": repr(exc), "traceback": traceback.format_exc()})
    writer.close()
    summary = {"run": spec.name, "frames": len(frames) if frames is not None else len(tokens), "errors": errors,
               "wall_s": round(time.perf_counter() - t_run, 1)}
    done.write_text(json.dumps(summary))
    return summary


def run_all(specs: list[RunSpec], env: EvalEnv, workers: int = 8,
            run_fn: Callable[[RunSpec, EvalEnv], dict] = run_one, log: Callable[[str], None] = print) -> list[dict]:
    todo = [s for s in specs if run_status(env.out_dir, s) != "done"]
    results = [json.loads(run_paths(env.out_dir, s)[1].read_text()) for s in specs if s not in todo]
    log(f"{len(specs)} runs: {len(results)} already done, {len(todo)} to run with {workers} workers")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run_fn, s, env): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            log(f"[{i}/{len(todo)}] {res['run']}: {res['frames']} frames, {res['errors']} errors, {res['wall_s']} s")
    return results


def load_records(out_dir: Path, specs: list[RunSpec]) -> list[dict]:
    records = []
    for s in specs:
        jsonl, _ = run_paths(out_dir, s)
        if jsonl.exists():
            records += [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    return records
