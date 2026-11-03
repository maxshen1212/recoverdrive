"""Shared test doubles."""

import json
from types import SimpleNamespace

import numpy as np

from recoverdrive.checks import CheckOutput, LidarObject
from recoverdrive.methods import FrameContext
from recoverdrive.types import Box3D, CameraData, CheckReport, FaultLabel, Frame, LidarData, Pose

GOOD = {"verdict": "ok", "fault_type": "none", "confidence": 0.9, "evidence": "all consistent"}
IDENTITY = Pose(np.array([1.0, 0, 0, 0]), np.zeros(3))


class FakeOpenAI:
    """Minimal stand-in for openai.OpenAI: returns `reply` (or a function of the request) and counts calls."""

    def __init__(self, reply=GOOD, exc=None):
        self.calls = []
        self.reply, self.exc = reply, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        reply = self.reply(kwargs) if callable(self.reply) else self.reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_ctx(flags=(), objects=(), agent_fault=None, label=None) -> FrameContext:
    cam = CameraData(image=np.zeros((9, 16, 3), np.uint8), intrinsic=np.eye(3), extrinsic=IDENTITY,
                     ego_pose=IDENTITY, timestamp=0)
    lidar = LidarData(points=np.zeros((0, 4)), extrinsic=IDENTITY, ego_pose=IDENTITY, timestamp=0)
    frame = Frame("nuscenes", "s", 0, 0, {"CAM_FRONT": cam}, lidar, IDENTITY, [])
    report = CheckReport(cameras={}, lidar_visible_ratio=None, point_count=0, flags=list(flags))
    check = CheckOutput(report, {}, list(objects), list(objects))
    part = {"type": "image", "image": np.zeros((4, 4, 3), np.uint8), "digest": "x"}
    return FrameContext(frame, label or FaultLabel.none(), check, "summary", part, None, part,
                        ["camera:CAM_FRONT", "lidar"], 123, agent_fault)


def lidar_object(x, y, n_points, category="car"):
    box = Box3D(center=np.array([x, y, 0.0]), size=np.array([2.0, 4.0, 1.5]), rotation=np.array([1.0, 0, 0, 0]),
                category=category)
    return LidarObject(box, n_points, float(np.hypot(x, y)))
