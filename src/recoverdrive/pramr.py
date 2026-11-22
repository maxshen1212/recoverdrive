"""PRAM-R-style baseline (plan 2026-11-22): an LLM modality router, adapted to one reliability score per sensor.

Source: Zhang et al., "PRAM-R: A Perception-Reasoning-Action-Memory Framework with LLM-Guided Modality Routing for
Adaptive Autonomous Driving", arXiv 2603.04222 (TUM, 2026). No code was released; this follows the paper:

1. Shallow perception (no LLM): diagnostic indicators per sensor, as in the paper's Table I. Cameras: brightness
   (mean gray), contrast (std of gray), edge density (share of pixels with a strong gradient). LiDAR: point density
   (points per 1 m^2 cell in 2-50 m), noise ratio (share of points with no neighbour within 0.5 m) and reflectivity
   ratio (mean intensity / its median on clean dev frames). External light = mean brightness of all cameras.
   Radar, weather and map indicators are left out: RecoverDrive has no radar input and uses no metadata.
2. Router (Qwen3-VL-8B, one call per keyframe): indicators with their dev normal range and the paper's contextual
   interpretation, memory of the previous step (indicators and reliability scores), and optionally the camera mosaic
   as scene context. It returns strict JSON: a reliability score in [0, 1] and a usage flag per sensor, and a scene
   complexity in [0, 1].
3. Threshold with hysteresis on each sensor's reliability: an active sensor is switched off when r < theta - delta,
   an inactive one back on when r >= theta + delta (the paper switches off at r <= theta - delta; "<" makes delta = 0
   equal to its reliable set r >= theta). Switched-off sensors are the isolated ones; the rule-based shield picks the
   action. The usage flags (an efficiency feature) are logged but never isolate a sensor.

Deviations from the paper, all stated in the report: one score per sensor instead of per modality; the router runs at
every 2 Hz keyframe (the paper's "w/o dual-loop" ablation); the memory holds scores and indicators, not the switched
state, so theta and delta can be tuned offline by replaying the logged scores; no EMA (there is no fusion to weight).
The paper has no re-ask, vote, verification or handling of LLM failures, and none is added: on an invalid or late
answer the previous state is kept.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "configs" / "pramr_reference.yaml"
CAMERA_KEYS = ("brightness", "contrast", "edge_density")
LIDAR_KEYS = ("density", "noise_ratio", "reflectivity_ratio")
LIDAR_KEY = "LIDAR_TOP"
EDGE_THRESHOLD = 40.0  # gray-level gradient magnitude (0-255) that counts as an edge
MIN_RANGE_M, MAX_RANGE_M = 2.0, 50.0
N_CELLS = math.pi * (MAX_RANGE_M**2 - MIN_RANGE_M**2)  # 1 m^2 cells in the 2-50 m annulus
NOISE_RADIUS_M = 0.5

# Frozen after dev tuning (plan step 8-9). Thresholds are in thousandths so that replay and live runs agree exactly.
SCENE_IMAGE = False
THETA_CAMERA = 500
THETA_LIDAR = 500
DELTA = 0
MAX_TOKENS = 384

INTERPRETATION = {  # the paper's Table I, "contextual indicators" and their interpretation
    "brightness": ("illumination level", "low brightness means low illumination and lower segmentation accuracy"),
    "contrast": ("visibility quality", "low contrast means low visibility, as in fog, rain or night"),
    "edge_density": ("texture density", "low edge density means less structural information and degraded clarity"),
    "density": ("LiDAR data completeness", "low point density means sparse, incomplete data"),
    "noise_ratio": ("LiDAR signal quality", "a high noise ratio means low signal quality, as in fog or heavy rain"),
    "reflectivity_ratio": ("LiDAR signal quality", ("a low reflectivity ratio means low signal quality, as in fog or "
                                                   "heavy rain")),
}
NAMES = {"brightness": "brightness", "contrast": "contrast", "edge_density": "edge density", "density": "point density",
         "noise_ratio": "noise ratio", "reflectivity_ratio": "reflectivity ratio"}

SYSTEM = (
    "You are the modality router of an autonomous vehicle with six cameras and one roof LiDAR. At every step you read "
    "diagnostic indicators of each sensor, the scene context and your memory of the previous step, and you estimate "
    "how reliable each sensor is now, how complex the scene is, and which sensors are needed. Respond only with "
    "strictly valid JSON."
)


# --- shallow perception --------------------------------------------------------------------------------------

def camera_indicators(image: np.ndarray) -> dict[str, float]:
    gray = image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    edges = np.hypot(gx, gy) > EDGE_THRESHOLD if gx.size else np.zeros(0, bool)
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()),
            "edge_density": float(edges.mean()) if edges.size else 0.0}


def lidar_raw_indicators(points: np.ndarray) -> dict[str, float]:
    """Density, noise ratio and mean intensity of a sensor-frame point cloud (N x 4: x, y, z, intensity)."""
    if points is None or len(points) == 0:
        return {"density": 0.0, "noise_ratio": 0.0, "mean_intensity": 0.0}
    rng = np.hypot(points[:, 0], points[:, 1])
    inside = points[(rng >= MIN_RANGE_M) & (rng <= MAX_RANGE_M)]
    noise = 0.0
    if len(inside) > 1:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(inside[:, :3]).query(inside[:, :3], k=2, distance_upper_bound=NOISE_RADIUS_M)
        noise = float(np.isinf(dist[:, 1]).mean())
    return {"density": len(inside) / N_CELLS, "noise_ratio": noise, "mean_intensity": float(points[:, 3].mean())}


def frame_indicators(frame, cameras: list[str], reference: dict) -> dict[str, dict[str, float] | None]:
    """Indicators of every camera, the LiDAR (None without points) and the global light level."""
    out: dict[str, dict[str, float] | None] = {c: camera_indicators(frame.cameras[c].image) for c in cameras}
    pts = None if frame.lidar is None else frame.lidar.points
    if pts is None or len(pts) == 0:
        out[LIDAR_KEY] = None
    else:
        raw = lidar_raw_indicators(pts)
        median = reference[LIDAR_KEY]["mean_intensity"]["median"] or 1.0
        out[LIDAR_KEY] = {"density": raw["density"], "noise_ratio": raw["noise_ratio"],
                          "reflectivity_ratio": raw["mean_intensity"] / median}
    out["light"] = {"brightness": float(np.mean([out[c]["brightness"] for c in cameras]))}
    return out


# --- reference ranges and labels ---------------------------------------------------------------------------------

def load_reference(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REFERENCE_PATH).read_text())["ranges"]


def label(value: float, rng: dict) -> str:
    return "low" if value < rng["p5"] else "high" if value > rng["p95"] else "normal"


def _fmt(key: str, value: float) -> str:
    return f"{value:.0f}" if key in ("brightness", "contrast") else f"{value:.3f}" if key in (
        "edge_density", "noise_ratio") else f"{value:.2f}"


def _line(name: str, values: dict[str, float] | None, ranges: dict) -> str:
    if values is None:
        return f"- {name}: no points received"
    parts = []
    for key, value in values.items():
        rng = ranges[key]
        context = INTERPRETATION[key][0]
        parts.append(f"{NAMES[key]} {_fmt(key, value)} (normal {_fmt(key, rng['p5'])}-{_fmt(key, rng['p95'])}; "
                     f"{context} {label(value, rng)})")
    return f"- {name}: " + ", ".join(parts)


def _compact(name: str, values: dict[str, float] | None) -> str:
    if values is None:
        return f"{name} no points"
    return f"{name} " + " ".join(f"{NAMES[k]} {_fmt(k, v)}" for k, v in values.items())


# --- router call ----------------------------------------------------------------------------------------------

def sensor_keys(cameras: list[str]) -> list[str]:
    return [*cameras, LIDAR_KEY]


def pramr_schema(cameras: list[str]) -> dict:
    keys = sensor_keys(cameras)
    props: dict = {f"{k}_reliability": {"type": "number", "minimum": 0, "maximum": 1} for k in keys}
    props["scene_complexity"] = {"type": "number", "minimum": 0, "maximum": 1}
    props.update({f"{k}_use": {"type": "boolean"} for k in keys})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def pramr_messages(cameras: list[str], indicators: dict, reference: dict, prev_indicators: dict | None,
                   prev_scores: dict[str, float] | None, mosaic: dict | None) -> list[dict]:
    """The router prompt. Nothing in it depends on the thresholds or on the switched state."""
    lines = ["Diagnostic indicators of the current step (normal = 5th to 95th percentile over clean driving data):"]
    lines += [_line(c, indicators[c], reference[c]) for c in cameras]
    lines.append(_line(LIDAR_KEY, indicators[LIDAR_KEY], reference[LIDAR_KEY]))
    light = indicators["light"]["brightness"]
    lines.append(f"External context: global illumination {light:.0f} (normal {reference['light']['brightness']['p5']:.0f}-"
                 f"{reference['light']['brightness']['p95']:.0f}; illumination level "
                 f"{label(light, reference['light']['brightness'])}).")
    lines.append("Interpretation: " + "; ".join(INTERPRETATION[k][1] for k in (*CAMERA_KEYS, *LIDAR_KEYS)) + ".")
    if prev_indicators is None:
        lines.append("Memory: none, this is the first step.")
    else:
        scores = ("unavailable" if prev_scores is None else
                  ", ".join(f"{k} {v:.2f}" for k, v in prev_scores.items()))
        lines.append("Memory of the previous step (0.5 s ago): indicators "
                     + "; ".join(_compact(k, prev_indicators[k]) for k in sensor_keys(cameras))
                     + f". Reliability scores you gave: {scores}.")
    lines.append(
        "Return JSON with: <SENSOR>_reliability for every sensor (0 = unreliable, 1 = fully reliable), "
        "scene_complexity (0 = simple, 1 = complex), and <SENSOR>_use for every sensor (true if the sensor is needed "
        "for the current scene). Sensors: " + ", ".join(sensor_keys(cameras)) + ".")
    content: list[dict] = []
    if mosaic is not None:
        content += [{"type": "text", "text": "Scene context: current frame, all cameras."}, mosaic]
    content.append({"type": "text", "text": "\n".join(lines)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def scores_of(data: dict | None, cameras: list[str]) -> dict[str, float] | None:
    if data is None:
        return None
    return {k: round(float(data[f"{k}_reliability"]), 3) for k in sensor_keys(cameras)}


# --- threshold, hysteresis and outcome ----------------------------------------------------------------------

def sensor_id(key: str) -> str:
    return "lidar" if key == LIDAR_KEY else f"camera:{key}"


def latch_step(active: dict[str, bool], scores: dict[str, float] | None, theta_camera: int, theta_lidar: int,
               delta: int) -> dict[str, bool]:
    """Hysteresis on each sensor's reliability (thresholds in thousandths). No scores: the state is kept."""
    if scores is None:
        return dict(active)
    out = {}
    for key, on in active.items():
        theta = theta_lidar if key == LIDAR_KEY else theta_camera
        r = round(scores[key] * 1000)
        if not on and r >= theta + delta:
            out[key] = True
        elif on and r < theta - delta:
            out[key] = False
        else:
            out[key] = on
    return out


def outcome(active: dict[str, bool], scores: dict[str, float] | None, order: list[str]) -> tuple[list[str], str]:
    """Isolated sensor ids (in `order`) and the blamed one: the switched-off sensor with the lowest score."""
    off = [k for k in order if not active[k]]
    isolated = [sensor_id(k) for k in off]
    if not off:
        return isolated, "none"
    if scores is None:
        return isolated, isolated[0]
    blamed = min(off, key=lambda k: (scores[k], order.index(k)))
    return isolated, sensor_id(blamed)
