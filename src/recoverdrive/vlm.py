"""Diagnosis agent: an OpenAI-compatible VLM (Qwen3-VL on vLLM by default) that returns schema-checked JSON."""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from PIL import Image

from recoverdrive.types import CheckReport, Diagnosis

SYSTEM_PROMPT = (
    "You are the sensor-health monitor of an autonomous vehicle. You decide whether the vehicle's sensors are "
    "working correctly, using the camera image you are shown and a consistency report computed by deterministic "
    "code that compares the camera against the LiDAR. Respond only with the requested JSON."
)


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


def _fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.2f}"


def summarize_report(report: CheckReport, max_range: float, iou_threshold: float) -> str:
    """Plain-text summary of a CheckReport for the VLM."""
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


CAMERA_VIEWS = {
    "CAM_FRONT": "facing forward",
    "CAM_FRONT_LEFT": "facing front-left",
    "CAM_FRONT_RIGHT": "facing front-right",
    "CAM_BACK": "facing backward",
    "CAM_BACK_LEFT": "facing back-left",
    "CAM_BACK_RIGHT": "facing back-right",
}


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


def encode_image(image: np.ndarray, quality: int = 90) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class VLMResult:
    diagnosis: Diagnosis | None  # None when the output is not valid JSON for the schema
    raw: str
    latency_s: float
    valid: bool


class VLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 256,
                 timeout_s: float = 60, api_key: str = "EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def diagnose(self, image: np.ndarray, camera: str, summary: str, sensors: list[str]) -> VLMResult:
        schema = diagnosis_schema(sensors)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image)}},
                {"type": "text", "text": build_user_prompt(camera, summary)},
            ]},
        ]
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "diagnosis", "schema": schema, "strict": True}},
        )
        latency = time.perf_counter() - t0
        raw = resp.choices[0].message.content or ""
        diagnosis = parse_diagnosis(raw, schema)
        return VLMResult(diagnosis=diagnosis, raw=raw, latency_s=latency, valid=diagnosis is not None)


def parse_diagnosis(raw: str, schema: dict) -> Diagnosis | None:
    """Parse and validate the VLM output against the schema; None if anything is off."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    props = schema["properties"]
    if not isinstance(data, dict) or set(data) != set(schema["required"]):
        return None
    if data["status"] not in props["status"]["enum"] or data["sensor"] not in props["sensor"]["enum"]:
        return None
    conf = data["confidence"]
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None
    return Diagnosis(status=data["status"], sensor=data["sensor"], confidence=float(conf))
