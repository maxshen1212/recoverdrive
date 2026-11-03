"""nuScenes adapter: reads keyframes of a scene as Frame objects."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image

from recoverdrive.types import Box3D, CameraData, Frame, LidarData, Pose


def map_category(name: str, category_map: dict[str, str]) -> str | None:
    """Map a nuScenes category (e.g. 'vehicle.bus.rigid') to a detector class by longest matching prefix."""
    best = None
    for prefix, cls in category_map.items():
        if (name == prefix or name.startswith(prefix + ".")) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, cls)
    return best[1] if best else None


class NuScenesAdapter:
    def __init__(self, dataroot: str | Path, version: str, cameras: list[str], category_map: dict[str, str]):
        self.dataroot = Path(dataroot)
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)
        self.cameras = cameras
        self.category_map = category_map

    def scene_sample_tokens(self, scene_name: str) -> list[str]:
        scene = next((s for s in self.nusc.scene if s["name"] == scene_name), None)
        if scene is None:
            raise KeyError(f"scene {scene_name} not in {self.nusc.version}")
        tokens, tok = [], scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = self.nusc.get("sample", tok)["next"]
        return tokens

    def scene_frames(self, scene_name: str) -> Iterator[Frame]:
        for idx, tok in enumerate(self.scene_sample_tokens(scene_name)):
            yield self.load_frame(scene_name, idx, tok)

    def _poses(self, sd: dict) -> tuple[Pose, Pose]:
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        extrinsic = Pose(np.array(cs["rotation"]), np.array(cs["translation"]))
        ego = Pose(np.array(ep["rotation"]), np.array(ep["translation"]))
        return extrinsic, ego

    def load_frame(self, scene_name: str, frame_idx: int, sample_token: str) -> Frame:
        sample = self.nusc.get("sample", sample_token)

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(str(self.dataroot / lidar_sd["filename"]))
        lidar_ext, lidar_ego = self._poses(lidar_sd)
        lidar = LidarData(points=pc.points.T.astype(np.float32), extrinsic=lidar_ext, ego_pose=lidar_ego,
                          timestamp=lidar_sd["timestamp"])

        cameras = {}
        for cam in self.cameras:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            ext, ego = self._poses(sd)
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            path = self.dataroot / sd["filename"]
            image = np.asarray(Image.open(path).convert("RGB"))
            cameras[cam] = CameraData(image=image, intrinsic=np.array(cs["camera_intrinsic"]), extrinsic=ext,
                                      ego_pose=ego, timestamp=sd["timestamp"], image_path=str(path))

        boxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cls = map_category(ann["category_name"], self.category_map)
            if cls is None:
                continue
            boxes.append(Box3D(center=np.array(ann["translation"]), size=np.array(ann["size"]),
                               rotation=np.array(ann["rotation"]), category=cls, token=ann_token))

        return Frame(source="nuscenes", scene_id=scene_name, frame_idx=frame_idx, timestamp=sample["timestamp"],
                     cameras=cameras, lidar=lidar, ego_pose=lidar_ego, gt_boxes=boxes)
