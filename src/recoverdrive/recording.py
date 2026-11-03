"""Run records: JSONL per frame, Rerun recordings (.rrd, for a headless server), and overlay images."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image, ImageDraw

GT_COLOR = (40, 200, 60)  # projected LiDAR-confirmed GT boxes
DET_COLOR = (40, 120, 255)  # camera detections


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fp.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class RerunRecorder:
    """Streams a recording straight to an .rrd file; open it later with the Rerun viewer."""

    def __init__(self, path: str | Path, app_id: str = "recoverdrive"):
        self.rec = rr.RecordingStream(app_id)
        self.rec.save(str(path))

    def log_camera(self, frame_idx: int, camera: str, image: np.ndarray, gt_boxes: np.ndarray,
                   gt_labels: list[str], det_boxes: np.ndarray, det_labels: list[str]) -> None:
        self.rec.set_time("frame", sequence=frame_idx)
        base = f"cameras/{camera}"
        self.rec.log(f"{base}/image", rr.Image(image))
        self.rec.log(f"{base}/gt_projected", rr.Boxes2D(array=gt_boxes.reshape(-1, 4),
                                                       array_format=rr.Box2DFormat.XYXY, colors=GT_COLOR,
                                                       labels=gt_labels))
        self.rec.log(f"{base}/sam3", rr.Boxes2D(array=det_boxes.reshape(-1, 4), array_format=rr.Box2DFormat.XYXY,
                                               colors=DET_COLOR, labels=det_labels))

    def log_scalar(self, frame_idx: int, path: str, value: float | None) -> None:
        if value is None:
            return
        self.rec.set_time("frame", sequence=frame_idx)
        self.rec.log(path, rr.Scalars(value))

    def log_text(self, frame_idx: int, path: str, text: str) -> None:
        self.rec.set_time("frame", sequence=frame_idx)
        self.rec.log(path, rr.TextDocument(text))

    def close(self) -> None:
        self.rec.flush()


def draw_overlay(image: np.ndarray, gt_boxes: np.ndarray, det_boxes: np.ndarray, title: str) -> Image.Image:
    """Image with projected GT boxes (green) and detections (blue), plus a title bar."""
    img = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(img)
    for b in det_boxes.reshape(-1, 4):
        draw.rectangle(b.tolist(), outline=DET_COLOR, width=3)
    for b in gt_boxes.reshape(-1, 4):
        draw.rectangle(b.tolist(), outline=GT_COLOR, width=3)
    draw.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255), font_size=20)
    return img
