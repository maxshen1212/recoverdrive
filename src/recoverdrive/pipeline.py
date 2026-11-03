"""Detect -> diagnose for one frame (the "Ours-detect" path, without recovery yet)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from recoverdrive.checks import (
    CameraConsistency,
    CheckConfig,
    build_report,
    camera_consistency,
    frozen_camera_flags,
    lidar_objects,
)
from recoverdrive.detectors.sam3 import Sam3Detector
from recoverdrive.types import CheckReport, Det2D, Diagnosis, Frame
from recoverdrive.vlm import VLMClient, VLMResult, summarize_report

__all__ = ["CheckConfig", "DetectDiagnosePipeline", "FrameResult"]


@dataclass
class FrameResult:
    frame_idx: int
    report: CheckReport
    consistency: dict[str, CameraConsistency]
    detections: dict[str, list[Det2D]]
    n_lidar_objects: int
    summary: str
    vlm: VLMResult | None
    latency_s: dict[str, float] = field(default_factory=dict)

    @property
    def diagnosis(self) -> Diagnosis | None:
        return self.vlm.diagnosis if self.vlm else None


class DetectDiagnosePipeline:
    def __init__(self, cameras: list[str], checks: CheckConfig, detector: Sam3Detector, vlm: VLMClient | None):
        self.cameras = cameras
        self.checks = checks
        self.detector = detector
        self.vlm = vlm

    def step(self, frame: Frame, prev_frame: Frame | None, prev_report: CheckReport | None) -> FrameResult:
        lat: dict[str, float] = {}
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        objects, visible_ratio = lidar_objects(frame, self.checks.max_range_m, self.checks.min_points_in_box)
        lat["lidar"] = time.perf_counter() - t0

        detections, consistency = {}, {}
        lat["sam3"] = lat["checks"] = 0.0
        for cam in self.cameras:
            t0 = time.perf_counter()
            detections[cam] = self.detector.detect(frame.cameras[cam].image)
            lat["sam3"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            consistency[cam] = camera_consistency(cam, frame.cameras[cam], objects, detections[cam],
                                                  self.checks.iou_threshold, self.checks.min_box_px)
            lat["checks"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        flags = frozen_camera_flags(frame, prev_frame, self.cameras)
        point_count = 0 if frame.lidar is None else len(frame.lidar.points)
        report = build_report(consistency, visible_ratio, point_count, prev_report, flags)
        summary = summarize_report(report, self.checks.max_range_m, self.checks.iou_threshold)
        lat["checks"] += time.perf_counter() - t0

        vlm_result = None
        if self.vlm is not None:
            cam = self.cameras[0]
            vlm_result = self.vlm.diagnose(frame.cameras[cam].image, cam, summary,
                                           sensors=[f"camera:{c}" for c in self.cameras] + ["lidar"])
            lat["vlm"] = vlm_result.latency_s
            report.agent_output_valid = vlm_result.valid

        lat["total"] = time.perf_counter() - t_start
        return FrameResult(frame_idx=frame.frame_idx, report=report, consistency=consistency, detections=detections,
                           n_lidar_objects=len(objects), summary=summary, vlm=vlm_result, latency_s=lat)
