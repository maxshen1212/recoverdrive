"""Deterministic consistency checks between LiDAR and cameras (spec §4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from recoverdrive.geometry import count_points_in_boxes, distance_xy, project_box, to_sensor_frame
from recoverdrive.types import Box3D, CameraCheck, CameraData, CheckReport, Det2D, Frame


@dataclass
class CheckConfig:
    max_range_m: float = 50.0
    min_points_in_box: int = 5
    iou_threshold: float = 0.3
    min_box_px: float = 10.0


@dataclass
class CheckThresholds:
    """Flag thresholds. Initial values; calibrated on dev scenes and then frozen (spec §4.1)."""

    match_rate_min: float = 0.5  # camera:<C>:low_match below this
    min_gt: int = 2  # a rate needs at least this many reference boxes to raise a flag
    match_drop: float = 0.5  # camera:<C>:match_drop when the rate falls by at least this much
    min_points: int = 30000  # lidar:low_points below this many points (clean dev keyframes: 34.4k-34.8k)
    point_drop: float = 0.3  # lidar:point_drop when the count falls by at least this fraction
    n_sectors: int = 12
    sector_min_points: int = 200  # lidar:sector_gap when any sector has fewer points (2-50 m; clean dev min ~900)
    visible_ratio_min: float = 0.3  # lidar:low_visibility below this
    agreement_min: float = 0.5  # lidar:camera_disagree below this


# Flags that point at a faulty sensor. Others (e.g. no_reference) are informational.
FAULT_FLAGS = ("low_match", "match_drop", "frozen", "low_points", "point_drop", "sector_gap", "low_visibility",
               "camera_disagree", "invalid", "timeout")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) boxes in x1, y1, x2, y2 format."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_boxes(gt: np.ndarray, det: np.ndarray, iou_threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one, class-agnostic matching that maximizes total IoU. Returns (gt_idx, det_idx, iou)."""
    iou = iou_matrix(gt, det)
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]


def match_rate(n_matched: int, n_gt: int) -> float | None:
    return n_matched / n_gt if n_gt > 0 else None


@dataclass
class LidarObject:
    """A GT box within range, with the number of LiDAR points inside it."""

    box: Box3D
    n_points: int
    distance: float


def lidar_box_counts(frame: Frame, max_range: float) -> list[LidarObject]:
    """All GT boxes within `max_range` of the LiDAR, with their in-box point counts."""
    lidar = frame.lidar
    if lidar is None:
        return []
    in_range = []
    for box in frame.gt_boxes:
        box_lidar = to_sensor_frame(box, lidar.ego_pose, lidar.extrinsic)
        dist = distance_xy(box_lidar)
        if dist <= max_range:
            in_range.append((box, box_lidar, dist))
    counts = count_points_in_boxes(lidar.points, [b for _, b, _ in in_range])
    return [LidarObject(box, int(n), dist) for (box, _, dist), n in zip(in_range, counts)]


def lidar_objects(frame: Frame, max_range: float, min_points: int) -> tuple[list[LidarObject], float | None]:
    """GT boxes the LiDAR sees (>= `min_points` points within range), and their share of all in-range boxes."""
    all_boxes = lidar_box_counts(frame, max_range)
    confirmed = [o for o in all_boxes if o.n_points >= min_points]
    ratio = len(confirmed) / len(all_boxes) if all_boxes else None
    return confirmed, ratio


def lidar_sector_counts(points: np.ndarray, n_sectors: int = 12, min_r: float = 2.0, max_r: float = 50.0) -> np.ndarray:
    """Points per azimuth sector (LiDAR frame), counting only points between min_r and max_r horizontally."""
    counts = np.zeros(n_sectors, dtype=int)
    if len(points) == 0:
        return counts
    xy = np.asarray(points)[:, :2]
    r = np.linalg.norm(xy, axis=1)
    keep = (r >= min_r) & (r <= max_r)
    az = np.arctan2(xy[keep, 1], xy[keep, 0])  # [-pi, pi]
    idx = np.floor((az + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    np.add.at(counts, idx, 1)
    return counts


@dataclass
class CameraConsistency:
    check: CameraCheck
    gt_boxes_2d: np.ndarray  # (n_gt, 4) projected LiDAR-confirmed boxes
    gt_labels: list[str]
    det_boxes_2d: np.ndarray  # (n_det, 4)
    matches: list[tuple[int, int, float]]


def project_objects(cam: CameraData, objects: list[LidarObject], min_box_px: float) -> tuple[np.ndarray, list[str]]:
    """Project boxes into a camera; keep those inside the image and at least min_box_px wide/tall."""
    boxes, labels, _ = _project(cam, objects, min_box_px)
    return boxes, labels


def _project(cam: CameraData, objects: list[LidarObject], min_box_px: float):
    h, w = cam.image.shape[:2]
    boxes, labels, kept = [], [], []
    for i, obj in enumerate(objects):
        box_cam = to_sensor_frame(obj.box, cam.ego_pose, cam.extrinsic)
        b2d = project_box(box_cam, cam.intrinsic, (w, h))
        if b2d is None or (b2d[2] - b2d[0]) < min_box_px or (b2d[3] - b2d[1]) < min_box_px:
            continue
        boxes.append(b2d)
        labels.append(obj.box.category)
        kept.append(i)
    return np.array(boxes, float).reshape(-1, 4), labels, kept


def camera_consistency(name: str, cam: CameraData, objects: list[LidarObject], detections: list[Det2D],
                       iou_threshold: float, min_box_px: float) -> CameraConsistency:
    gt_2d, gt_labels = project_objects(cam, objects, min_box_px)
    det_2d = np.array([d.box for d in detections], float).reshape(-1, 4)
    matches = match_boxes(gt_2d, det_2d, iou_threshold)
    check = CameraCheck(camera=name, n_gt=len(gt_2d), n_det=len(det_2d), n_matched=len(matches),
                        match_rate=match_rate(len(matches), len(gt_2d)))
    return CameraConsistency(check, gt_2d, gt_labels, det_2d, matches)


def lidar_camera_agreement(frame: Frame, cameras: list[str], detections: dict[str, list[Det2D]],
                           all_boxes: list[LidarObject], cfg: CheckConfig) -> tuple[float | None, int]:
    """Cross-check in the other direction: of in-range GT boxes that some camera detection confirms
    (IoU >= threshold, any point count), the share the LiDAR also sees (>= k points)."""
    confirmed: set[int] = set()
    for cam in cameras:
        if cam not in frame.cameras:
            continue
        boxes, _, kept = _project(frame.cameras[cam], all_boxes, cfg.min_box_px)
        det = np.array([d.box for d in detections.get(cam, [])], float).reshape(-1, 4)
        for gi, _, _ in match_boxes(boxes, det, cfg.iou_threshold):
            confirmed.add(kept[gi])
    if not confirmed:
        return None, 0
    seen = sum(all_boxes[i].n_points >= cfg.min_points_in_box for i in confirmed)
    return seen / len(confirmed), len(confirmed)


def frozen_camera_flags(frame: Frame, prev: Frame | None, cameras: list[str]) -> list[str]:
    """Flag cameras whose image is byte-identical to the previous frame (frozen feed)."""
    if prev is None:
        return []
    return [f"camera:{c}:frozen" for c in cameras
            if c in prev.cameras and np.array_equal(frame.cameras[c].image, prev.cameras[c].image)]


def build_report(cam_results: dict[str, CameraConsistency], lidar_visible_ratio: float | None, point_count: int,
                 prev: CheckReport | None, flags: list[str] | None = None) -> CheckReport:
    """Assemble a CheckReport, filling frame-to-frame changes from the previous report."""
    flags = list(flags or [])
    cameras = {}
    for name, res in cam_results.items():
        check = res.check
        prev_check = prev.cameras.get(name) if prev else None
        if check.match_rate is not None and prev_check is not None and prev_check.match_rate is not None:
            check.match_rate_delta = check.match_rate - prev_check.match_rate
        if check.match_rate is None:
            flags.append(f"camera:{name}:no_reference")
        cameras[name] = check
    point_delta = point_count - prev.point_count if prev else None
    return CheckReport(cameras=cameras, lidar_visible_ratio=lidar_visible_ratio, point_count=point_count,
                       point_count_delta=point_delta, flags=flags)


def compute_flags(report: CheckReport, thr: CheckThresholds, frozen: list[str] | None = None) -> list[str]:
    """Fault flags from a report and thresholds (deterministic)."""
    flags = list(frozen or [])
    for name, c in report.cameras.items():
        if c.match_rate is None or c.n_gt < thr.min_gt:
            continue
        if c.match_rate < thr.match_rate_min:
            flags.append(f"camera:{name}:low_match")
        if c.match_rate_delta is not None and c.match_rate_delta <= -thr.match_drop:
            flags.append(f"camera:{name}:match_drop")
    if report.point_count < thr.min_points:
        flags.append("lidar:low_points")
    if report.point_count_delta is not None:
        prev_count = report.point_count - report.point_count_delta
        if prev_count > 0 and report.point_count_delta / prev_count <= -thr.point_drop:
            flags.append("lidar:point_drop")
    if report.lidar_sector_counts and min(report.lidar_sector_counts) < thr.sector_min_points:
        flags.append("lidar:sector_gap")
    if (report.lidar_visible_ratio is not None and report.lidar_n_in_range >= thr.min_gt
            and report.lidar_visible_ratio < thr.visible_ratio_min):
        flags.append("lidar:low_visibility")
    if (report.lidar_camera_agreement is not None and report.lidar_camera_n >= thr.min_gt
            and report.lidar_camera_agreement < thr.agreement_min):
        flags.append("lidar:camera_disagree")
    return flags


def flagged_sensors(flags: list[str]) -> set[str]:
    """Sensors named by fault flags: 'camera:<C>', 'lidar' or 'agent'."""
    out = set()
    for f in flags:
        parts = f.split(":")
        if parts[-1] not in FAULT_FLAGS:
            continue
        out.add(":".join(parts[:2]) if parts[0] == "camera" else parts[0])
    return out


def flags_by_sensor(flags: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in flags:
        parts = f.split(":")
        if parts[-1] not in FAULT_FLAGS:
            continue
        key = ":".join(parts[:2]) if parts[0] == "camera" else parts[0]
        out.setdefault(key, []).append(parts[-1])
    return out


def rank_sensors(flags: list[str]) -> list[str]:
    """Sensors with fault flags, most flags first, ties by name ('agent' and no_reference flags are ignored)."""
    groups = flags_by_sensor(flags)
    groups.pop("agent", None)
    return sorted(groups, key=lambda s: (-len(groups[s]), s))


@dataclass
class CheckOutput:
    report: CheckReport
    consistency: dict[str, CameraConsistency]
    objects: list[LidarObject]  # LiDAR-confirmed GT boxes (>= k points)
    all_boxes: list[LidarObject]  # every in-range GT box with its point count


def run_checks(frame: Frame, prev_frame: Frame | None, prev_report: CheckReport | None,
               detections: dict[str, list[Det2D]], cfg: CheckConfig, thr: CheckThresholds,
               cameras: list[str]) -> CheckOutput:
    """All §4.1 checks for one frame: cross-sensor, LiDAR health, temporal, and the resulting flags."""
    all_boxes = lidar_box_counts(frame, cfg.max_range_m)
    objects = [o for o in all_boxes if o.n_points >= cfg.min_points_in_box]
    visible = len(objects) / len(all_boxes) if all_boxes else None
    consistency = {cam: camera_consistency(cam, frame.cameras[cam], objects, detections.get(cam, []),
                                           cfg.iou_threshold, cfg.min_box_px) for cam in cameras}
    points = frame.lidar.points if frame.lidar is not None else np.zeros((0, 4))
    report = build_report(consistency, visible, len(points), prev_report)
    report.lidar_n_in_range = len(all_boxes)
    report.lidar_sector_counts = lidar_sector_counts(points, thr.n_sectors, max_r=cfg.max_range_m).tolist()
    report.lidar_camera_agreement, report.lidar_camera_n = lidar_camera_agreement(frame, cameras, detections,
                                                                                  all_boxes, cfg)
    report.flags += compute_flags(report, thr, frozen_camera_flags(frame, prev_frame, cameras))
    return CheckOutput(report, consistency, objects, all_boxes)
