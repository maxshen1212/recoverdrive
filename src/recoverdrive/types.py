"""Data classes passed between pipeline modules (spec §2.2).

Every module is a function from these classes to these classes, so each one can be tested with fake data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pyquaternion import Quaternion

Source = Literal["nuscenes", "carla"]
Severity = Literal["low", "medium", "high"]
Status = Literal["ok", "fault", "uncertain"]
FinalState = Literal["healthy", "degraded", "unresolved"]
ActionName = Literal["continue", "slow_down", "stop"]


@dataclass
class Pose:
    """Rigid transform from a child frame to its parent (e.g. sensor -> ego, ego -> global)."""

    rotation: np.ndarray  # quaternion (w, x, y, z)
    translation: np.ndarray  # (3,)

    @property
    def quaternion(self) -> Quaternion:
        return Quaternion(self.rotation)

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.quaternion.rotation_matrix
        m[:3, 3] = self.translation
        return m


@dataclass
class CameraData:
    image: np.ndarray  # (H, W, 3) uint8 RGB
    intrinsic: np.ndarray  # (3, 3)
    extrinsic: Pose  # camera -> ego
    ego_pose: Pose  # ego -> global, at this camera's timestamp
    timestamp: int  # microseconds
    image_path: str = ""


@dataclass
class LidarData:
    points: np.ndarray  # (N, 4) x, y, z, intensity in the LiDAR frame
    extrinsic: Pose  # lidar -> ego
    ego_pose: Pose  # ego -> global, at the LiDAR timestamp
    timestamp: int


@dataclass
class Box3D:
    center: np.ndarray  # (3,) global frame
    size: np.ndarray  # (3,) width, length, height
    rotation: np.ndarray  # quaternion (w, x, y, z), global frame
    category: str  # detector class, e.g. "car"
    token: str = ""


@dataclass
class Frame:
    source: Source
    scene_id: str
    frame_idx: int
    timestamp: int
    cameras: dict[str, CameraData]
    lidar: LidarData | None
    ego_pose: Pose  # ego -> global at the keyframe (LiDAR) timestamp
    gt_boxes: list[Box3D]


@dataclass
class FaultLabel:
    fault_id: str  # "C1" ... "A3", or "none"
    sensor: str  # "camera:<name>", "lidar", "agent", or "none"
    severity: Severity | None = None
    episode_id: int | None = None

    @staticmethod
    def none() -> FaultLabel:
        return FaultLabel(fault_id="none", sensor="none")


@dataclass
class Det2D:
    box: np.ndarray  # (4,) x1, y1, x2, y2 in pixels
    label: str
    score: float


@dataclass
class CameraCheck:
    camera: str
    n_gt: int  # LiDAR-confirmed GT boxes that project into this camera
    n_det: int  # camera detections
    n_matched: int
    match_rate: float | None  # None when n_gt == 0: nothing to check against
    match_rate_delta: float | None = None  # change from the previous frame


@dataclass
class CheckReport:
    cameras: dict[str, CameraCheck]
    lidar_visible_ratio: float | None  # GT boxes in range seen by LiDAR (>= k points) / all in range
    point_count: int
    point_count_delta: int | None = None
    agent_output_valid: bool | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    status: Status
    sensor: str
    confidence: float
    fault_type: str | None = None
    evidence: str = ""


@dataclass
class RecoveryResult:
    steps_taken: list[str]
    final_state: FinalState
    isolated_sensors: list[str]
    final_diagnosis: Diagnosis | None


@dataclass
class Action:
    action: ActionName
    reason: str
