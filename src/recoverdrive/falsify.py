"""Falsification on nuScenes: search fault parameters for runs that violate a requirement (R1-R6).

A trial injects one fault episode, with chosen type, target, start, duration and continuous level, into one dev
scene. It then runs the method over a window around the episode with the evaluation runner and scores the window by
the STL robustness of one requirement. Negative means the requirement is violated. The search (Task 7) minimizes
that score.

Continuous levels need a severity label, because the acceptable actions depend on it. `severity_of` uses the existing
level tables: "low" below the table's medium value, "medium" below its high value, "high" from there on. Some
violations sit at these label boundaries; the report says so.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from recoverdrive.faults import (
    C2_LEVELS,
    C3_LEVELS,
    C4_LAGS,
    FIXED_HIGH,
    L1_LEVELS,
    L2_LEVELS,
    SEVERITIES,
    FaultEvent,
)
from recoverdrive.requirements import evaluate, nuscenes_trace
from recoverdrive.runner import EvalEnv, RunSpec, run_one, sensors_of

# Faults whose images are all original nuScenes images (C1 shows another camera's image, C4 an earlier image, L*
# only change points), so every SAM 3 detection is already cached and a trial needs no GPU.
CPU_FAULTS = ("C1", "C4", "L1", "L2", "L4")
GPU_FAULTS = ("C2", "C3")

# Search ranges per fault and subtype: (low, high) for continuous levels, a tuple of choices for C4.
FAULT_SPACES: dict[str, dict] = {
    "C1": {},
    "C2": {"rotate": (0.0, 12.0), "shift": (0.0, 0.2)},
    "C3": {"blur": (0.0, 12.0), "noise": (0.0, 80.0), "overexpose": (1.0, 5.0), "black": None},
    "C4": {"lag": C4_LAGS},
    "L1": {"random": (0.0, 1.0), "beam": (0.0, 1.0), "sector": (0.0, 360.0)},
    "L2": {"translation_m": (0.0, 1.5), "yaw_deg": (0.0, 8.0)},
    "L4": {},
}
MIN_DURATION, MAX_DURATION = 2, 6  # frames, as in the evaluation's episodes
C1_TARGET, C1_SOURCE = "CAM_FRONT", "CAM_BACK"


def _rank(value: float, table: tuple) -> int:
    """0/1/2 = low/medium/high: below the medium value, below the high value, or above."""
    return 0 if value < table[1] else 1 if value < table[2] else 2


def severity_of(fault_id: str, subtype: str | None, level) -> str:
    if fault_id in FIXED_HIGH or (fault_id == "C3" and subtype == "black"):
        return "high"
    if fault_id == "C4":
        return SEVERITIES[C4_LAGS.index(level)]
    if fault_id == "L2":
        t, yaw = level
        return SEVERITIES[max(_rank(t, tuple(x[0] for x in L2_LEVELS)), _rank(abs(yaw), tuple(x[1] for x in L2_LEVELS)))]
    table = {"C2": C2_LEVELS, "C3": C3_LEVELS, "L1": L1_LEVELS}[fault_id][subtype]
    return SEVERITIES[_rank(level, table)]


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return float(min(max(value, bounds[0]), bounds[1]))


def event_from_params(fault_id: str, p: dict, episode_id: int = 0) -> FaultEvent:
    """One fault episode from search parameters; numeric levels are clipped into FAULT_SPACES."""
    seed = int(p.get("seed", 0))
    space = FAULT_SPACES[fault_id]
    if fault_id == "C1":
        return FaultEvent("C1", f"camera:{C1_TARGET}", "high", episode_id, {"target": C1_TARGET, "source": C1_SOURCE})
    if fault_id == "L4":
        return FaultEvent("L4", "lidar", "high", episode_id, {"seed": seed})
    if fault_id == "L2":
        t = _clip(p["translation_m"], space["translation_m"])
        yaw = _clip(abs(p["yaw_deg"]), space["yaw_deg"]) * (1 if p.get("yaw_sign", 1) >= 0 else -1)
        params = {"translation_m": t, "yaw_deg": yaw, "direction_deg": float(p.get("direction_deg", 0.0)) % 360,
                  "seed": seed}
        return FaultEvent("L2", "lidar", severity_of("L2", None, (t, yaw)), episode_id, params)
    if fault_id == "L1":
        sub = p["subtype"]
        level = _clip(p["level"], space[sub])
        params = {"subtype": sub, "level": level, "seed": seed, "azimuth_deg": float(p.get("azimuth_deg", 0.0))}
        return FaultEvent("L1", "lidar", severity_of("L1", sub, level), episode_id, params)
    cam = p["camera"]
    params: dict = {"camera": cam, "seed": seed}
    if fault_id == "C4":
        if p["lag"] not in C4_LAGS:
            raise ValueError(f"C4 lag must be one of {C4_LAGS}")
        params["lag"] = p["lag"]
        return FaultEvent("C4", f"camera:{cam}", severity_of("C4", None, p["lag"]), episode_id, params)
    sub = p["subtype"]
    level = None if space[sub] is None else _clip(p["level"], space[sub])
    params.update(subtype=sub, level=level)
    if fault_id == "C2":
        params["sign"] = 1 if p.get("sign", 1) >= 0 else -1
    return FaultEvent(fault_id, f"camera:{cam}", severity_of(fault_id, sub, level), episode_id, params)


def window_plan(n_frames: int, start: int, duration: int, event: FaultEvent) -> list[FaultEvent | None]:
    """A full-length plan with `event` on frames [start, start+duration), cut at the end of the scene."""
    event.params["start"] = start
    return [event if start <= i < start + duration else None for i in range(n_frames)]


def window_frames(n_frames: int, start: int, duration: int, warmup: int = 3, tail: int = 3) -> range:
    """The frames a trial processes: `warmup` clean frames before the episode (C4 copies up to 2 frames back, and the
    recovery state needs a history) and `tail` frames after it (R3 needs 2)."""
    return range(max(0, start - warmup), min(n_frames, start + duration + tail))


@dataclass
class TrialResult:
    robustness: float
    satisfied: bool
    n_violations: int
    params: dict
    run_dir: str


def evaluate_trial(env: EvalEnv, scene: str, fault_id: str, p: dict, method: str, spec_name: str,
                   out_dir: Path) -> TrialResult:
    """Run one trial window with `method` and score it with requirement `spec_name`. Frame errors stop the trial
    (for example a SAM 3 cache miss in a CPU-only campaign), so a broken trial never counts as a result."""
    n = len(env.adapter.scene_sample_tokens(scene))
    start = int(min(max(int(p.get("start", 0)), 0), n - MIN_DURATION))
    duration = int(min(max(int(p.get("duration", MIN_DURATION)), MIN_DURATION), MAX_DURATION))
    event = event_from_params(fault_id, p)
    trial_env = dataclasses.replace(env, out_dir=Path(out_dir), methods=[method])
    spec = RunSpec("dev", scene, fault_id, 0)
    summary = run_one(spec, trial_env, plan=window_plan(n, start, duration, event),
                      frames=window_frames(n, start, duration))
    jsonl = Path(out_dir) / "runs" / f"{spec.name}.jsonl"
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    if summary["errors"]:
        first = next(r for r in records if "error" in r)
        raise RuntimeError(f"trial {out_dir}: {summary['errors']} frame errors, first: {first['error']}")
    sensors = tuple(sensors_of(env.cameras))
    trace, _ = nuscenes_trace(records, method, sensors)
    res = evaluate(trace, (spec_name,), sensors)[spec_name]
    return TrialResult(res.robustness, res.satisfied, res.n_violations, p, str(out_dir))
