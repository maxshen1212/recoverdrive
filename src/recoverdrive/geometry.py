"""Deterministic geometry: frame transforms, points-in-box counts, and 3D -> 2D box projection."""

from __future__ import annotations

import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box, view_points
from pyquaternion import Quaternion

from recoverdrive.types import Box3D, Pose

# Corner index pairs of the 12 box edges, following nuScenes Box.corners() ordering.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def to_sensor_frame(box: Box3D, ego_pose: Pose, extrinsic: Pose) -> Box:
    """Move a global-frame box into a sensor frame, given ego -> global and sensor -> ego poses."""
    b = Box(np.asarray(box.center, float), np.asarray(box.size, float), Quaternion(box.rotation),
            name=box.category, token=box.token)
    b.translate(-np.asarray(ego_pose.translation, float))
    b.rotate(ego_pose.quaternion.inverse)
    b.translate(-np.asarray(extrinsic.translation, float))
    b.rotate(extrinsic.quaternion.inverse)
    return b


def count_points_in_boxes(points: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Number of points (N, >=3) inside each box. Points and boxes must share a frame."""
    if not boxes:
        return np.zeros(0, dtype=int)
    if len(points) == 0:
        return np.zeros(len(boxes), dtype=int)
    xyz = points[:, :3].T
    return np.array([int(points_in_box(b, xyz).sum()) for b in boxes], dtype=int)


def project_box(box_cam: Box, intrinsic: np.ndarray, image_wh: tuple[int, int], near: float = 0.1):
    """2D bounding box (x1, y1, x2, y2) of a camera-frame 3D box, clipped to the image.

    Edges that cross the near plane are cut there, so boxes partly behind the camera still project
    correctly. Returns None when nothing of the box is in front of the camera or inside the image.
    """
    corners = box_cam.corners()  # (3, 8), z points forward
    pts = []
    for i, j in _EDGES:
        a, b = corners[:, i], corners[:, j]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            pts.append(a)
        if b_in:
            pts.append(b)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            pts.append(a + t * (b - a))
    if not pts:
        return None
    uv = view_points(np.stack(pts, axis=1), intrinsic, normalize=True)[:2]
    w, h = image_wh
    x1, y1 = np.clip(uv.min(axis=1), 0, [w, h])
    x2, y2 = np.clip(uv.max(axis=1), 0, [w, h])
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=float)


def distance_xy(box: Box) -> float:
    """Horizontal distance from the sensor origin to the box center."""
    return float(np.linalg.norm(box.center[:2]))
