"""VLM access: one OpenAI-compatible client (Qwen3-VL on vLLM by default) with schema-constrained JSON and a
content-addressed cache, plus the prompts of the diagnosis agent ("Ours")."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from PIL import Image

from recoverdrive.cache import KVCache, digest, image_digest
from recoverdrive.types import CheckReport, Diagnosis

SYSTEM_PROMPT = (
    "You are the sensor-health monitor of an autonomous vehicle. You decide whether the vehicle's sensors are "
    "working correctly, using the camera image you are shown and a consistency report computed by deterministic "
    "code that compares the camera against the LiDAR. Respond only with the requested JSON."
)

CAMERA_VIEWS = {
    "CAM_FRONT": "facing forward",
    "CAM_FRONT_LEFT": "facing front-left",
    "CAM_FRONT_RIGHT": "facing front-right",
    "CAM_BACK": "facing backward",
    "CAM_BACK_LEFT": "facing back-left",
    "CAM_BACK_RIGHT": "facing back-right",
}

FAULT_TYPES = ["none", "wrong_viewpoint", "misalignment", "degraded_image", "frozen_image", "point_dropout",
               "calibration_shift", "lidar_failure", "other"]


# --- images and messages -----------------------------------------------------------------------------------

def encode_image(image: np.ndarray, quality: int = 90) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_part(image: np.ndarray, image_id: str | None = None) -> dict:
    """Placeholder content part; encoded to a data URL only when sent. The cache key uses the digest."""
    return {"type": "image", "image": image, "digest": image_id or image_digest(image)}


def _keyable(messages: list[dict]) -> list:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [{"type": "image", "digest": p["digest"]} if p.get("type") == "image" else p for p in content]
        out.append({"role": m["role"], "content": content})
    return out


def _encode(messages: list[dict]) -> list:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [{"type": "image_url", "image_url": {"url": encode_image(p["image"])}}
                       if p.get("type") == "image" else p for p in content]
        out.append({"role": m["role"], "content": content})
    return out


# --- schema validation -----------------------------------------------------------------------------------

_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list}


def validate_json(raw: str, schema: dict) -> dict | None:
    """Parse `raw` and check it against a flat JSON schema (required, enum, type, min/max); None if invalid."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    props = schema.get("properties", {})
    if any(k not in data for k in schema.get("required", [])):
        return None
    if schema.get("additionalProperties") is False and any(k not in props for k in data):
        return None
    for key, spec in props.items():
        if key not in data:
            continue
        val = data[key]
        typ = spec.get("type")
        if typ in ("number", "integer"):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return None
            if "minimum" in spec and val < spec["minimum"] or "maximum" in spec and val > spec["maximum"]:
                return None
        elif typ in _TYPES and not isinstance(val, _TYPES[typ]):
            return None
        if "enum" in spec and val not in spec["enum"]:
            return None
    return data


def split_verdict(verdict: str) -> tuple[str, str]:
    """'fault:<sensor>' -> ('fault', sensor); 'ok' / 'uncertain' -> (status, 'none')."""
    if verdict.startswith("fault:"):
        return "fault", verdict[len("fault:"):]
    return verdict, "none"


def to_diagnosis(data: dict | None) -> Diagnosis | None:
    if data is None:
        return None
    if "verdict" in data:
        status, sensor = split_verdict(data["verdict"])
    else:
        status, sensor = data["status"], data["sensor"]
    return Diagnosis(status=status, sensor=sensor, confidence=float(data["confidence"]),
                     fault_type=data.get("fault_type"), evidence=str(data.get("evidence", "")))


# --- client --------------------------------------------------------------------------------------------------

@dataclass
class ChatResult:
    raw: str
    latency_s: float
    cached: bool


class VLMClient:
    """All VLM calls go through here, so switching models is a config change only (spec §2.3)."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 256,
                 timeout_s: float = 60, api_key: str = "EMPTY", cache: KVCache | None = None, client=None):
        self.client = client or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.cache = cache
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def chat_json(self, messages: list[dict], schema: dict, temperature: float = 0.0, seed: int | None = None,
                  max_tokens: int | None = None, name: str = "output") -> ChatResult:
        max_tokens = max_tokens or self.max_tokens
        key = "vlm:" + digest({"model": self.model, "messages": _keyable(messages), "schema": schema,
                               "temperature": temperature, "seed": seed if temperature > 0 else None,
                               "max_tokens": max_tokens})
        if self.cache is None:
            return self._send(key, messages, schema, temperature, seed, max_tokens, name)
        with self._locks_guard:
            lock = self._locks[key]
        with lock:  # identical concurrent requests wait for the first one and then hit the cache
            hit = self.cache.get(key)
            if hit is not None:
                return ChatResult(hit["raw"], hit["latency_s"], cached=True)
            return self._send(key, messages, schema, temperature, seed, max_tokens, name)

    def _send(self, key, messages, schema, temperature, seed, max_tokens, name) -> ChatResult:
        kwargs = {"seed": seed} if seed is not None else {}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=_encode(messages), temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
            **kwargs)
        latency = time.perf_counter() - t0
        raw = resp.choices[0].message.content or ""
        if self.cache is not None:
            self.cache.set(key, {"raw": raw, "latency_s": latency})
        return ChatResult(raw, latency, cached=False)

    # E0 interface (scripts/run_e0.py, scripts/smoke_test.py)
    def diagnose(self, image: np.ndarray, camera: str, summary: str, sensors: list[str]) -> VLMResult:
        schema = diagnosis_schema(sensors)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [image_part(image), {"type": "text", "text": build_user_prompt(camera, summary)}]},
        ]
        res = self.chat_json(messages, schema, temperature=self.temperature, name="diagnosis")
        diagnosis = parse_diagnosis(res.raw, schema)
        return VLMResult(diagnosis=diagnosis, raw=res.raw, latency_s=res.latency_s, valid=diagnosis is not None)


# --- check summary (shared by Ours prompts) --------------------------------------------------------------------

def _fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.2f}"


def summarize_checks(report: CheckReport, max_range: float, iou_threshold: float, min_points: int) -> str:
    """Plain-text summary of all §4.1 checks for the VLM."""
    lines = [(f"Camera checks (objects within {max_range:.0f} m that the LiDAR sees, projected into each camera, "
              f"versus the camera's own detections, IoU >= {iou_threshold}):")]
    for name, c in report.cameras.items():
        if c.n_gt == 0:
            lines.append(f"- {name}: no LiDAR objects expected in view, so it cannot be cross-checked "
                         f"({c.n_det} camera detections).")
            continue
        prev = "n/a" if c.match_rate_delta is None else _fmt_rate(c.match_rate - c.match_rate_delta)
        lines.append(f"- {name}: {c.n_matched} of {c.n_gt} expected objects matched (match rate "
                     f"{_fmt_rate(c.match_rate)}, previous frame {prev}); {c.n_det} camera detections.")
    prev_pts = "n/a" if report.point_count_delta is None else f"{report.point_count - report.point_count_delta}"
    lidar = [f"LiDAR: {report.point_count} points (previous frame {prev_pts})."]
    if report.lidar_sector_counts:
        sc = np.asarray(report.lidar_sector_counts)
        width = 360 // len(sc)
        lidar.append(f"Points per {width}-degree sector: min {sc.min()}, median {int(np.median(sc))}.")
    if report.lidar_visible_ratio is not None:
        n_seen = round(report.lidar_visible_ratio * report.lidar_n_in_range)
        lidar.append(f"{n_seen} of {report.lidar_n_in_range} objects within {max_range:.0f} m have >= {min_points} "
                     f"LiDAR points (ratio {_fmt_rate(report.lidar_visible_ratio)}).")
    if report.lidar_camera_agreement is not None:
        lidar.append(f"Of {report.lidar_camera_n} objects confirmed by camera detections, the LiDAR also sees "
                     f"{_fmt_rate(report.lidar_camera_agreement)}.")
    lines.append(" ".join(lidar))
    fault_flags = [f for f in report.flags if not f.endswith(":no_reference")]
    lines.append("Flags raised by the deterministic checks: " + (", ".join(fault_flags) if fault_flags else "none") + ".")
    lines.append("Suspicious sensors according to the checks: " + suspicious_sensors(report, fault_flags) + ".")
    return "\n".join(lines)


FLAG_WORDS = {"low_match": "camera missed most expected objects", "match_drop": "match rate dropped sharply",
              "frozen": "image identical to the previous frame", "low_points": "far fewer LiDAR points than normal",
              "point_drop": "LiDAR point count dropped sharply", "sector_gap": "a LiDAR direction has no points",
              "low_visibility": "few nearby objects have LiDAR points",
              "camera_disagree": "cameras see objects that have no LiDAR points"}


def suspicious_sensors(report: CheckReport, fault_flags: list[str]) -> str:
    """Flags grouped per sensor in plain words, the sensor with most flags first."""
    groups: dict[str, list[str]] = {}
    for f in fault_flags:
        parts = f.split(":")
        sensor = ":".join(parts[:2]) if parts[0] == "camera" else parts[0]
        groups.setdefault(sensor, []).append(FLAG_WORDS.get(parts[-1], parts[-1]))
    if not groups:
        return "none"
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    return "; ".join(f"{s} ({', '.join(w)})" for s, w in ranked)


# --- Ours: diagnosis agent prompt -------------------------------------------------------------------------------

OURS_SYSTEM = (
    "You are the sensor-health monitor of an autonomous vehicle with six cameras and one roof LiDAR. Each frame you "
    "get: (1) a mosaic of the current images of all cameras, each tile labeled with its camera name; (2) the same "
    "mosaic from the previous frame, 0.5 s earlier; (3) a top view of the LiDAR point cloud; and (4) a report from "
    "deterministic code that cross-checks the LiDAR against the cameras. Decide whether one sensor is faulty. "
    "Respond only with the requested JSON."
)

OURS_GUIDE = (
    "Decide in this order:\n"
    "1. Flags. The deterministic checks are reliable. low_match/match_drop on a camera means it missed objects "
    "the LiDAR sees in its view; frozen means the image did not change; lidar flags mean missing points or objects. "
    "One flag with few expected objects can be occlusion or a detector miss; a match rate near 0 with several "
    "expected objects, or several flags on one sensor, is strong evidence of a fault of that sensor.\n"
    "2. Images, only to confirm a flagged sensor or for obvious damage: a tile that is all black, washed out, "
    "unrecognizable from blur or noise, tilted, or the same picture as another tile. Do not report subtle blur, "
    "noise or darkness: night, rain, glare, motion blur and empty roads are normal. Different cameras see different "
    "things; do not judge a camera's direction from content alone.\n"
    "3. LiDAR top view: about 34,000 points in all directions is healthy; an empty wedge or far fewer points is a "
    "LiDAR fault.\n"
    "If no sensor has strong evidence, answer verdict ok with fault_type none. Answer fault:<sensor> only when the "
    "evidence points to that sensor; uncertain only when strong evidence conflicts.\n\n"
    "Return JSON with: verdict (ok, uncertain, or fault:<sensor>), fault_type, confidence in your verdict (0 to 1), "
    "and evidence (one short sentence)."
)


EVIDENCE_FIRST = False  # field order: verdict first (fewer false alarms on dev than reasoning first)


def verdict_enum(sensors: list[str]) -> list[str]:
    return ["ok", "uncertain", *[f"fault:{s}" for s in sensors]]


def ours_schema(sensors: list[str]) -> dict:
    verdict = {
        "verdict": {"type": "string", "enum": verdict_enum(sensors)},
        "fault_type": {"type": "string", "enum": FAULT_TYPES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    evidence = {"evidence": {"type": "string"}}
    props = {**evidence, **verdict} if EVIDENCE_FIRST else {**verdict, **evidence}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def ours_messages(summary: str, mosaic: dict, prev_mosaic: dict | None, bev: dict) -> list[dict]:
    """`mosaic`, `prev_mosaic`, `bev` are image parts from `image_part`."""
    content: list[dict] = [{"type": "text", "text": "Image 1: current frame, all cameras."}, mosaic]
    if prev_mosaic is not None:
        content += [{"type": "text", "text": "Image 2: previous frame, same layout."}, prev_mosaic]
    else:
        content += [{"type": "text", "text": "(No previous frame: this is the first frame.)"}]
    content += [{"type": "text", "text": "Image 3: LiDAR top view."}, bev,
                {"type": "text", "text": f"Consistency report:\n{summary}\n\n{OURS_GUIDE}"}]
    return [{"role": "system", "content": OURS_SYSTEM}, {"role": "user", "content": content}]


# --- E0 legacy helpers ----------------------------------------------------------------------------------------

def diagnosis_schema(sensors: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "fault", "uncertain"]},
            "sensor": {"type": "string", "enum": [*sensors, "none"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["status", "sensor", "confidence"],
        "additionalProperties": False,
    }


def summarize_report(report: CheckReport, max_range: float, iou_threshold: float) -> str:
    """E0 summary of a CheckReport (single camera)."""
    lines = []
    for name, c in report.cameras.items():
        if c.n_gt == 0:
            lines.append(f"- {name}: the LiDAR sees no objects that should appear in this camera, so the camera "
                         f"cannot be cross-checked this frame. The camera detector found {c.n_det} objects.")
            continue
        prev = "n/a" if c.match_rate_delta is None else _fmt_rate(c.match_rate - c.match_rate_delta)
        lines.append(
            f"- {name}: the LiDAR sees {c.n_gt} objects within {max_range:.0f} m that should appear in this "
            f"camera. The camera detector found {c.n_det} objects. {c.n_matched} of the {c.n_gt} LiDAR objects "
            f"overlap a camera detection at the expected place (IoU >= {iou_threshold}): "
            f"match rate {_fmt_rate(c.match_rate)} (previous frame: {prev}).")
    lidar_ratio = _fmt_rate(report.lidar_visible_ratio)
    lines.append(f"- LiDAR: {report.point_count} points; share of nearby annotated objects with enough LiDAR "
                 f"points: {lidar_ratio}.")
    if report.flags:
        lines.append(f"- Flags: {', '.join(report.flags)}")
    return "\n".join(lines)


def build_user_prompt(camera: str, summary: str) -> str:
    view = f" ({CAMERA_VIEWS[camera]})" if camera in CAMERA_VIEWS else ""
    return (
        f"The image is the current frame reported by the {camera} camera{view}.\n\n"
        f"Consistency report:\n{summary}\n\n"
        "A healthy camera usually agrees with most LiDAR objects. A low match rate can mean the camera is faulty "
        "(for example wrong viewpoint, misalignment, or a degraded image), but it can also come from occlusion "
        "or detector misses, so also check whether the image looks like a plausible view from this camera.\n\n"
        "Return JSON with: status (ok, fault, or uncertain); sensor (the faulty sensor, or none if ok); "
        "confidence in your status (0 to 1)."
    )


@dataclass
class VLMResult:
    diagnosis: Diagnosis | None  # None when the output is not valid JSON for the schema
    raw: str
    latency_s: float
    valid: bool


def parse_diagnosis(raw: str, schema: dict) -> Diagnosis | None:
    """Parse and validate the VLM output against the schema; None if anything is off."""
    return to_diagnosis(validate_json(raw, schema))


# --- B1-B3: DriveAgent-style text chain (prompts from DriveAgent/vehicle_prompts.json) --------------------------

CAPTION_SYSTEM = (
    "You are a visual perception assistant for an autonomous vehicle. Describe what this camera image shows "
    "that could impact driving.")
CAPTION_USER = (
    "This image comes from the {camera} camera. In two or three sentences, list the road users and obstacles you "
    "see (type, rough position: left/centre/right, near/far) and anything unusual about the image itself.")
CAPTION_SCHEMA = {"type": "object", "properties": {"description": {"type": "string"}},
                  "required": ["description"], "additionalProperties": False}

TEXTCHAIN_SYSTEM = {
    "zero_shot": ("You are a helpful assistant. Your task is to determine whether a vehicle is experiencing an "
                  "abnormal situation based on the readings from two sensors."),
    "cot": ("You are a helpful assistant. Your task is to determine whether a vehicle is experiencing an abnormal "
            "situation based on readings from two sensors. Before providing your final answer, please outline your "
            "reasoning step-by-step. Your chain-of-thought should include the following steps:\n1. Analyze the "
            "LiDAR data to identify any unusual patterns or discrepancies.\n2. Analyze the visual description for "
            "abnormal features or inconsistencies.\n3. Compare the sensor data to determine if there are any "
            "mismatches that indicate an abnormal situation.\n4. Conclude whether the vehicle is in an abnormal "
            "situation, and provide a final answer along with your reasoning."),
    "self_refine": ("You are a helpful assistant. Your task is to determine whether a vehicle is experiencing an "
                    "abnormal situation based on the readings from two sensors. You will produce an initial answer, "
                    "critique it, and finally produce a refined answer."),
}
TEXTCHAIN_STEPS = {
    "zero_shot": "",
    "cot": "",
    "self_refine": ("\n\nStep 1: Provide your initial analysis regarding whether there's an abnormal situation.\n"
                    "Step 2: Critique and reflect on your initial analysis. Identify any missing details or possible "
                    "errors.\nStep 3: Provide a refined final analysis, correcting mistakes and clarifying "
                    "uncertainties."),
}
TEXTCHAIN_REASONING = {"zero_shot": [], "cot": ["reasoning"], "self_refine": ["initial_analysis", "critique",
                                                                            "refined_analysis"]}


def textchain_schema(variant: str, sensors: list[str]) -> dict:
    props = {k: {"type": "string"} for k in TEXTCHAIN_REASONING[variant]}
    props.update({"verdict": {"type": "string", "enum": verdict_enum(sensors)},
                  "fault_type": {"type": "string", "enum": FAULT_TYPES},
                  "confidence": {"type": "number", "minimum": 0, "maximum": 1}})
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def textchain_messages(variant: str, lidar: str, visual: str, sensors: list[str]) -> list[dict]:
    user = (f"Object information (from lidar): {lidar}\nVisual Description: {visual}{TEXTCHAIN_STEPS[variant]}\n\n"
            f"The vehicle's sensors are: {', '.join(sensors)}. Report which sensor, if any, is abnormal. Return "
            "JSON with verdict (ok, uncertain, or fault:<sensor>), fault_type, and confidence (0 to 1)"
            + (" after your reasoning fields. Keep each reasoning field to at most five sentences."
               if TEXTCHAIN_REASONING[variant] else "."))
    return [{"role": "system", "content": TEXTCHAIN_SYSTEM[variant]}, {"role": "user", "content": user}]
