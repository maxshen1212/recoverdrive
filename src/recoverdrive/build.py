"""Build pipeline components from a YAML config (see configs/e0.yaml)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from recoverdrive.adapters.nuscenes import NuScenesAdapter
from recoverdrive.detectors.sam3 import Sam3Detector
from recoverdrive.pipeline import CheckConfig, DetectDiagnosePipeline
from recoverdrive.vlm import VLMClient


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_adapter(cfg: dict) -> NuScenesAdapter:
    data = cfg["data"]
    root = os.environ.get("NUSCENES_ROOT", "data/nuscenes")
    cameras = list(dict.fromkeys(data["monitored_cameras"] + data.get("source_cameras", [])))
    return NuScenesAdapter(root, data["version"], cameras, cfg["categories"])


def build_pipeline(cfg: dict, device: str = "cuda", with_vlm: bool = True) -> DetectDiagnosePipeline:
    s = cfg["sam3"]
    detector = Sam3Detector(s["prompts"], s["score_threshold"], s["model_id"], device=device)
    vlm = None
    if with_vlm:
        v = cfg["vlm"]
        vlm = VLMClient(v["base_url"], v["model"], v["temperature"], v["max_tokens"], v["timeout_s"])
    return DetectDiagnosePipeline(cfg["data"]["monitored_cameras"], CheckConfig(**cfg["checks"]), detector, vlm)
