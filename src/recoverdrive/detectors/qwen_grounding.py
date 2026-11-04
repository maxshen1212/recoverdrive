"""Qwen3-VL grounding as the camera detector (E5 ablation: SAM 3 vs Qwen grounding, spec §1.4 / §5.3).

Qwen3-VL returns `bbox_2d` in coordinates normalized to 0-1000; they are rescaled to pixels here. Results are cached by
the VLM client (temperature 0), so repeated images cost nothing.
"""

from __future__ import annotations

import json
import threading

import numpy as np

from recoverdrive.types import Det2D
from recoverdrive.vlm import VLMClient, image_part


def grounding_schema(prompts: list[str]) -> dict:
    box = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}
    obj = {"type": "object", "properties": {"label": {"type": "string", "enum": list(prompts)}, "bbox_2d": box},
           "required": ["label", "bbox_2d"], "additionalProperties": False}
    return {"type": "object", "properties": {"objects": {"type": "array", "items": obj}},
            "required": ["objects"], "additionalProperties": False}


def parse_grounding(raw: str, width: int, height: int, prompts: list[str]) -> list[Det2D] | None:
    """Pixel-space detections from a grounding answer; None if the answer is not valid JSON."""
    try:
        data = json.loads(raw)
        objs = data["objects"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    dets = []
    for o in objs:
        try:
            x1, y1, x2, y2 = (float(v) for v in o["bbox_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        if o.get("label") not in prompts or x2 <= x1 or y2 <= y1:
            continue
        box = np.array([x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000])
        dets.append(Det2D(box=box, label=o["label"], score=1.0))
    return dets


class QwenGroundingDetector:
    """Drop-in replacement for CachedDetector (same `detect`, `hits`, `misses`)."""

    def __init__(self, client: VLMClient, prompts: list[str], max_tokens: int = 1024):
        self.client, self.prompts, self.max_tokens = client, list(prompts), max_tokens
        self.schema = grounding_schema(self.prompts)
        self.hits = self.misses = self.invalid = 0
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray, image_id: str | None = None) -> list[Det2D]:
        h, w = image.shape[:2]
        text = (f"Detect every {', '.join(self.prompts)} in the image. Return JSON with `objects`: a list of "
                "{label, bbox_2d}, where bbox_2d is [x1, y1, x2, y2] in coordinates normalized to 0-1000. "
                "Return an empty list if there is none.")
        msgs = [{"role": "user", "content": [image_part(image, image_id), {"type": "text", "text": text}]}]
        res = self.client.chat_json(msgs, self.schema, temperature=0.0, max_tokens=self.max_tokens, name="grounding")
        dets = parse_grounding(res.raw, w, h, self.prompts)
        with self._lock:
            self.hits += res.cached
            self.misses += not res.cached
            self.invalid += dets is None
        return dets or []
