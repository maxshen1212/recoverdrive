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


def make_frame(idx=0, scene="s"):
    """A tiny two-camera frame without LiDAR and without objects."""
    front = CameraData(image=np.zeros((4, 4, 3), np.uint8), intrinsic=np.eye(3) * 2, extrinsic=IDENTITY,
                       ego_pose=IDENTITY, timestamp=0, image_path="front.jpg")
    back = CameraData(image=np.full((4, 4, 3), 255, np.uint8), intrinsic=np.eye(3) * 3, extrinsic=IDENTITY,
                      ego_pose=IDENTITY, timestamp=0, image_path="back.jpg")
    return Frame(source="nuscenes", scene_id=scene, frame_idx=idx, timestamp=0,
                 cameras={"CAM_FRONT": front, "CAM_BACK": back}, lidar=None, ego_pose=IDENTITY, gt_boxes=[])


class FakeAdapter:
    """Scenes of `n` make_frame() keyframes."""

    def __init__(self, n=6):
        self.n = n

    def scene_sample_tokens(self, scene):
        return [f"{scene}-{i}" for i in range(self.n)]

    def load_frame(self, scene, idx, token):
        return make_frame(idx, scene)


class FakeDetector:
    def detect(self, image, image_id=None):
        return []


class DictCache:
    """In-memory stand-in for cache.KVCache."""

    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def set(self, key, value):
        self.d[key] = value


def record(idx, fault="none", sensor="none", sev=None, ep=None, action="continue", state="healthy", iso=(),
           method="m", scene="s", seed=0, run_fault="L1"):
    """One logged frame as run_one writes it, with a single method output."""
    label = {"fault_id": fault, "sensor": sensor, "severity": sev, "episode_id": ep}
    out = {"action": {"action": action}, "recovery": {"final_state": state, "isolated_sensors": list(iso)}}
    return {"scene": scene, "seed": seed, "run_fault": run_fault, "frame_idx": idx, "label": label,
            "methods": {method: out}}


def carla_step(step, fault="none", sensor="none", sev=None, action="continue", acceptable=("continue",), speed=8.3,
               ttc=None, isolated=(), state="healthy", monitor_action=None):
    """One logged CARLA step as scripts/run_e3.py writes it. `action` is after the TTC brake override,
    `monitor_action` before it (defaults to `action`)."""
    label = {"fault_id": fault, "sensor": sensor, "severity": sev, "episode_id": None if fault == "none" else 0}
    return {"step": step, "label": label, "monitor_action": monitor_action or action, "action": action,
            "acceptable": list(acceptable), "speed_mps": speed, "ttc": ttc, "isolated": list(isolated),
            "state": state}
