"""CARLA adapter: sensor data from a CARLA 0.9.16 ego vehicle -> `Frame` (same conventions as nuScenes).

Conventions after conversion (all right-handed, metres):
- global / ego: x forward, y left, z up. CARLA (Unreal) is left-handed with y to the right, so every position and
  transform is mirrored with S = diag(1, -1, 1).
- camera frame: OpenCV (x right, y down, z forward), like nuScenes; `extrinsic` maps camera -> ego.
- LiDAR points: in the LiDAR frame, right-handed; `extrinsic` maps LiDAR -> ego.
The pure-math helpers below need no CARLA install; `CarlaRig` imports `carla` lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion

from recoverdrive.types import Box3D, CameraData, Frame, LidarData, Pose

S = np.diag([1.0, -1.0, 1.0])
# Columns: the OpenCV camera axes (x right, y down, z forward) expressed in a body frame (x fwd, y left, z up).
CV_TO_BODY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def ue_matrix_to_rh(m: np.ndarray) -> np.ndarray:
    """4x4 Unreal (left-handed) transform -> the same transform in the mirrored right-handed frame."""
    out = np.eye(4)
    out[:3, :3] = S @ m[:3, :3] @ S
    out[:3, 3] = S @ m[:3, 3]
    return out


def ue_points_to_rh(points: np.ndarray) -> np.ndarray:
    out = np.array(points, dtype=np.float32, copy=True)
    out[:, 1] *= -1
    return out


def pose_from_matrix(m: np.ndarray) -> Pose:
    q = Quaternion(matrix=m[:3, :3], atol=1e-6, rtol=1e-6)
    return Pose(np.array(q.elements), np.array(m[:3, 3], float))


def ue_transform_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Same matrix as carla.Transform.get_matrix() (degrees; UE rotation order yaw-pitch-roll)."""
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0, 0, 0, 1.0]])


def camera_extrinsic(mount_ue: np.ndarray) -> Pose:
    """Camera -> ego pose for a camera mounted with UE transform `mount_ue` (relative to the vehicle)."""
    m = ue_matrix_to_rh(mount_ue)
    out = np.eye(4)
    out[:3, :3] = m[:3, :3] @ CV_TO_BODY
    out[:3, 3] = m[:3, 3]
    return pose_from_matrix(out)


def intrinsic_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def bgra_to_rgb(raw: bytes | np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(arr[:, :, [2, 1, 0]])


VEHICLE_CLASSES = {"car": "car", "van": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
                   "bicycle": "bicycle"}


def actor_box(actor_matrix_ue: np.ndarray, bb_location: tuple[float, float, float],
              bb_extent: tuple[float, float, float], category: str, token: str = "") -> Box3D:
    """Global right-handed Box3D from an actor's world transform and its local bounding box."""
    m = ue_matrix_to_rh(actor_matrix_ue)
    center = m[:3, :3] @ (S @ np.array(bb_location, float)) + m[:3, 3]
    q = Quaternion(matrix=m[:3, :3], atol=1e-6, rtol=1e-6)
    ex, ey, ez = bb_extent  # half sizes along the actor's x (length), y (width), z
    return Box3D(center=center, size=np.array([2 * ey, 2 * ex, 2 * ez]), rotation=np.array(q.elements),
                 category=category, token=token)


@dataclass
class CameraSpec:
    name: str
    x: float
    y: float
    z: float
    yaw: float  # UE degrees: 0 forward, 180 backward
    width: int = 800
    height: int = 450
    fov: float = 70.0

    def mount(self) -> np.ndarray:
        return ue_transform_matrix(self.x, self.y, self.z, 0.0, 0.0, self.yaw)


DEFAULT_CAMERAS = [CameraSpec("CAM_FRONT", 1.5, 0.0, 1.6, 0.0), CameraSpec("CAM_BACK", -1.5, 0.0, 1.6, 180.0)]
LIDAR_MOUNT = (0.0, 0.0, 1.9)


def build_frame(scene: str, frame_idx: int, timestamp_us: int, ego_matrix_ue: np.ndarray,
                images: dict[str, np.ndarray], cameras: list[CameraSpec], lidar_points_ue: np.ndarray,
                boxes: list[Box3D]) -> Frame:
    """Assemble a Frame. All sensors are read on the same synchronous tick, so they share the ego pose."""
    ego = pose_from_matrix(ue_matrix_to_rh(ego_matrix_ue))
    cams = {}
    for spec in cameras:
        cams[spec.name] = CameraData(image=images[spec.name], intrinsic=intrinsic_from_fov(spec.width, spec.height,
                                                                                          spec.fov),
                                     extrinsic=camera_extrinsic(spec.mount()), ego_pose=ego, timestamp=timestamp_us)
    lidar_ext = pose_from_matrix(ue_matrix_to_rh(ue_transform_matrix(*LIDAR_MOUNT, 0.0, 0.0, 0.0)))
    pts = ue_points_to_rh(lidar_points_ue) if len(lidar_points_ue) else np.zeros((0, 4), np.float32)
    lidar = LidarData(points=pts, extrinsic=lidar_ext, ego_pose=ego, timestamp=timestamp_us)
    return Frame("carla", scene, frame_idx, timestamp_us, cams, lidar, ego, boxes)
