"""Deterministic consistency checks between LiDAR and cameras (spec §4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from recoverdrive.geometry import count_points_in_boxes, distance_xy, project_box, to_sensor_frame
from recoverdrive.types import Box3D, CameraCheck, CameraData, CheckReport, Det2D, Frame


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
    """A GT box that the LiDAR actually sees (enough points inside)."""

    box: Box3D
    n_points: int
    distance: float


def lidar_objects(frame: Frame, max_range: float, min_points: int) -> tuple[list[LidarObject], float | None]:
    """GT boxes within range and their LiDAR point counts.

    Returns the boxes with at least `min_points` points, and the ratio of such boxes to all in-range boxes
    (None when no box is in range).
    """
    lidar = frame.lidar
    if lidar is None:
        return [], None
    in_range = []
    for box in frame.gt_boxes:
        box_lidar = to_sensor_frame(box, lidar.ego_pose, lidar.extrinsic)
        dist = distance_xy(box_lidar)
        if dist <= max_range:
            in_range.append((box, box_lidar, dist))
    counts = count_points_in_boxes(lidar.points, [b for _, b, _ in in_range])
    confirmed = [LidarObject(box, int(n), dist) for (box, _, dist), n in zip(in_range, counts) if n >= min_points]
    ratio = len(confirmed) / len(in_range) if in_range else None
    return confirmed, ratio


@dataclass
class CameraConsistency:
    check: CameraCheck
    gt_boxes_2d: np.ndarray  # (n_gt, 4) projected LiDAR-confirmed boxes
    gt_labels: list[str]
    det_boxes_2d: np.ndarray  # (n_det, 4)
    matches: list[tuple[int, int, float]]


def project_objects(cam: CameraData, objects: list[LidarObject], min_box_px: float) -> tuple[np.ndarray, list[str]]:
    """Project LiDAR-confirmed boxes into a camera; keep those inside the image and at least min_box_px wide/tall."""
    h, w = cam.image.shape[:2]
    boxes, labels = [], []
    for obj in objects:
        box_cam = to_sensor_frame(obj.box, cam.ego_pose, cam.extrinsic)
        b2d = project_box(box_cam, cam.intrinsic, (w, h))
        if b2d is None or (b2d[2] - b2d[0]) < min_box_px or (b2d[3] - b2d[1]) < min_box_px:
            continue
        boxes.append(b2d)
        labels.append(obj.box.category)
    return np.array(boxes, float).reshape(-1, 4), labels


def camera_consistency(name: str, cam: CameraData, objects: list[LidarObject], detections: list[Det2D],
                       iou_threshold: float, min_box_px: float) -> CameraConsistency:
    gt_2d, gt_labels = project_objects(cam, objects, min_box_px)
    det_2d = np.array([d.box for d in detections], float).reshape(-1, 4)
    matches = match_boxes(gt_2d, det_2d, iou_threshold)
    check = CameraCheck(camera=name, n_gt=len(gt_2d), n_det=len(det_2d), n_matched=len(matches),
                        match_rate=match_rate(len(matches), len(gt_2d)))
    return CameraConsistency(check, gt_2d, gt_labels, det_2d, matches)


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
