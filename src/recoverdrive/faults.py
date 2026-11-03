"""Fault injection (spec §3.2–3.3). Faults come in seeded episodes of consecutive frames.

Each episode gets one `FaultEvent` (fault id, sensor, severity, parameters) sampled from the run's RNG, so a run
is fully determined by its seed. Sensor faults change the `Frame`; agent faults (A1-A3) are applied to the
diagnosis calls by `AgentFaultState` (see `agent.py`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace

import numpy as np

from recoverdrive.types import FaultLabel, Frame, LidarData

SEVERITIES = ("low", "medium", "high")
SENSOR_FAULTS = ("C1", "C2", "C3", "C4", "L1", "L2", "L4")
AGENT_FAULTS = ("A1", "A2", "A3")
ALL_FAULTS = SENSOR_FAULTS + AGENT_FAULTS
FIXED_HIGH = ("C1", "L4")  # no severity levels; always treated as high (spec §5.2)

# Severity parameters (plan D5). Index 0/1/2 = low/medium/high.
L1_LEVELS = {"random": (0.3, 0.6, 0.9), "beam": (0.25, 0.5, 0.75), "sector": (45.0, 90.0, 180.0)}
AGENT_ATTEMPTS = (1, 2, 3)  # diagnosis attempts affected in a frame (plan D4)
A1_KINDS = ("truncate", "missing_field", "garbage")
# HDL-32E elevation range used to bin points into 32 beams.
BEAM_ELEV_DEG = (-30.67, 10.67)
N_BEAMS = 32


def schedule_episodes(n_frames: int, fault_ratio: float, rng: np.random.Generator,
                      min_len: int = 2, max_len: int = 6) -> list[int | None]:
    """Episode id for each frame (None = clean).

    About `fault_ratio` of frames are faulty, grouped into episodes of `min_len`..`max_len` frames with at
    least one clean frame between episodes.
    """
    target = round(n_frames * fault_ratio)
    if target < min_len:
        return [None] * n_frames
    k_min = math.ceil(target / max_len)
    k_max = min(target // min_len, n_frames - target + 1)  # inner gaps need one clean frame each
    if k_min > k_max:
        raise ValueError(f"cannot fit {target} faulty frames into {n_frames} with episode length {min_len}-{max_len}")
    k = int(rng.integers(k_min, k_max + 1))

    lengths = [min_len] * k
    for _ in range(target - k * min_len):
        open_slots = [i for i in range(k) if lengths[i] < max_len]
        lengths[int(rng.choice(open_slots))] += 1

    gaps = [0] + [1] * (k - 1) + [0]
    for _ in range(n_frames - target - (k - 1)):
        gaps[int(rng.integers(0, k + 1))] += 1

    schedule: list[int | None] = []
    for ep in range(k):
        schedule += [None] * gaps[ep] + [ep] * lengths[ep]
    schedule += [None] * gaps[k]
    return schedule


@dataclass
class FaultEvent:
    fault_id: str
    sensor: str  # "camera:<name>", "lidar" or "agent"
    severity: str
    episode_id: int
    params: dict = field(default_factory=dict)

    def label(self) -> FaultLabel:
        return FaultLabel(fault_id=self.fault_id, sensor=self.sensor, severity=self.severity,
                          episode_id=self.episode_id)


def _pick_camera(rng: np.random.Generator, cameras: list[str], front_prob: float) -> str:
    others = [c for c in cameras if c != "CAM_FRONT"]
    if "CAM_FRONT" in cameras and (not others or rng.random() < front_prob):
        return "CAM_FRONT"
    return str(rng.choice(others))


def sample_event(fault_id: str, episode_id: int, rng: np.random.Generator, cfg: dict,
                 cameras: list[str]) -> FaultEvent:
    """Sample severity, subtype, target and parameters of one episode."""
    severity = "high" if fault_id in FIXED_HIGH else str(rng.choice(SEVERITIES))
    level = SEVERITIES.index(severity)
    seed = int(rng.integers(2**31))
    if fault_id == "C1":
        c1 = cfg.get("c1", {"target": "CAM_FRONT", "source": "CAM_BACK"})
        return FaultEvent("C1", f"camera:{c1['target']}", severity, episode_id, dict(c1))
    if fault_id == "L1":
        subtype = str(rng.choice(list(L1_LEVELS)))
        params = {"subtype": subtype, "level": L1_LEVELS[subtype][level], "seed": seed,
                  "azimuth_deg": float(rng.uniform(-180, 180))}
        return FaultEvent("L1", "lidar", severity, episode_id, params)
    if fault_id in AGENT_FAULTS:
        params = {"n_affected": AGENT_ATTEMPTS[level], "seed": seed}
        if fault_id == "A1":
            params["kind"] = str(rng.choice(A1_KINDS))
        return FaultEvent(fault_id, "agent", severity, episode_id, params)
    raise NotImplementedError(f"fault {fault_id} is not implemented yet")


def plan_faults(fault_id: str, n_frames: int, rng: np.random.Generator, cfg: dict,
                cameras: list[str]) -> list[FaultEvent | None]:
    """Per-frame fault events for one run: about cfg['fault_ratio'] of frames, in episodes (spec §3.3)."""
    schedule = schedule_episodes(n_frames, cfg.get("fault_ratio", 0.5), rng, cfg.get("min_episode_len", 2),
                                 cfg.get("max_episode_len", 6))
    events: dict[int, FaultEvent] = {}
    for idx, ep in enumerate(schedule):
        if ep is not None and ep not in events:
            events[ep] = sample_event(fault_id, ep, rng, cfg, cameras)
            events[ep].params["start"] = idx
    return [None if ep is None else events[ep] for ep in schedule]


# --- sensor injectors (pure functions) -------------------------------------------------------------------

def inject_c1(frame: Frame, target: str, source: str) -> Frame:
    """C1, wrong viewpoint: `target` camera reports the image of `source`, keeping its own calibration."""
    cams = dict(frame.cameras)
    src = frame.cameras[source]
    cams[target] = replace(frame.cameras[target], image=src.image, image_path=src.image_path)
    return replace(frame, cameras=cams)


def _azimuth_deg(points: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(points[:, 1], points[:, 0]))


def beam_index(points: np.ndarray) -> np.ndarray:
    """Approximate beam (0..31) from each point's elevation angle."""
    r = np.linalg.norm(points[:, :2], axis=1)
    elev = np.degrees(np.arctan2(points[:, 2], np.maximum(r, 1e-6)))
    lo, hi = BEAM_ELEV_DEG
    return np.clip(np.floor((elev - lo) / (hi - lo) * N_BEAMS), 0, N_BEAMS - 1).astype(int)


def inject_l1(points: np.ndarray, subtype: str, level: float, rng: np.random.Generator,
              azimuth_deg: float = 0.0, beam_rng: np.random.Generator | None = None) -> np.ndarray:
    """L1 point dropout. random: drop a `level` fraction of points; beam: drop round(32*level) beams chosen by
    `beam_rng` (fixed for an episode); sector: drop a `level`-degree azimuth band centred on `azimuth_deg`."""
    if len(points) == 0:
        return points
    if subtype == "random":
        keep = rng.random(len(points)) >= level
    elif subtype == "beam":
        drop = (beam_rng or rng).choice(N_BEAMS, size=round(N_BEAMS * level), replace=False)
        keep = ~np.isin(beam_index(points), drop)
    elif subtype == "sector":
        diff = (_azimuth_deg(points) - azimuth_deg + 180.0) % 360.0 - 180.0
        keep = np.abs(diff) > level / 2
    else:
        raise ValueError(f"unknown L1 subtype {subtype}")
    return points[keep]


def _with_points(frame: Frame, points: np.ndarray) -> Frame:
    assert frame.lidar is not None
    lidar: LidarData = replace(frame.lidar, points=points)
    return replace(frame, lidar=lidar)


class SensorFaultInjector:
    """Applies the planned sensor fault of each frame. Agent faults leave the frame unchanged."""

    def __init__(self, plan: list[FaultEvent | None]):
        self.plan = plan

    def event(self, frame_idx: int) -> FaultEvent | None:
        return self.plan[frame_idx]

    def apply(self, frame: Frame, originals: dict[int, Frame] | None = None) -> tuple[Frame, FaultLabel]:
        """`originals` maps frame_idx -> unfaulted earlier frames (needed by C4)."""
        ev = self.plan[frame.frame_idx]
        if ev is None:
            return frame, FaultLabel.none()
        p = ev.params
        if ev.fault_id in AGENT_FAULTS:
            return frame, ev.label()
        if ev.fault_id == "C1":
            return inject_c1(frame, p["target"], p["source"]), ev.label()
        if ev.fault_id == "L1":
            rng = np.random.default_rng([p["seed"], frame.frame_idx])
            pts = inject_l1(frame.lidar.points, p["subtype"], p["level"], rng, p["azimuth_deg"],
                            beam_rng=np.random.default_rng(p["seed"]))
            return _with_points(frame, pts), ev.label()
        raise NotImplementedError(f"fault {ev.fault_id} is not implemented yet")


# --- agent faults -----------------------------------------------------------------------------------------

class SimulatedTimeout(Exception):
    """A3: the model did not answer in time (simulated, no real wait)."""


@dataclass
class AgentFaultState:
    """Agent fault active in the current frame. Attempts 0..n_affected-1 of the diagnosis call are affected."""

    fault_id: str
    severity: str
    params: dict

    @staticmethod
    def from_event(ev: FaultEvent | None) -> AgentFaultState | None:
        if ev is None or ev.fault_id not in AGENT_FAULTS:
            return None
        return AgentFaultState(ev.fault_id, ev.severity, dict(ev.params))

    def affects(self, attempt_idx: int) -> bool:
        return attempt_idx < self.params["n_affected"]

    def corrupt_output(self, raw: str) -> str:
        """A1: malformed JSON or a missing field."""
        kind = self.params.get("kind", "truncate")
        if kind == "truncate":
            return raw[: max(1, len(raw) // 2)]
        if kind == "missing_field":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return raw[: max(1, len(raw) // 2)]
            for key in ("verdict", "status", "sensor", "confidence"):
                data.pop(key, None)
            return json.dumps(data)
        return "Sorry, I cannot help with that request."


class FaultInjector:
    """E0 legacy interface: a fixed C1 fault on scheduled frames (scripts/run_e0.py)."""

    def __init__(self, schedule: list[int | None], fault_id: str, params: dict):
        self.schedule, self.fault_id, self.params = schedule, fault_id, params

    def apply(self, frame: Frame) -> tuple[Frame, FaultLabel]:
        episode = self.schedule[frame.frame_idx]
        if episode is None:
            return frame, FaultLabel.none()
        if self.fault_id == "C1":
            target = self.params["target"]
            label = FaultLabel(fault_id="C1", sensor=f"camera:{target}", severity="high", episode_id=episode)
            return inject_c1(frame, target, self.params["source"]), label
        raise NotImplementedError(f"fault {self.fault_id} is not implemented yet")
