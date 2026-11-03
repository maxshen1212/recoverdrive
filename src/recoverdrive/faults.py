"""Fault injection (spec §3.2–3.3). Faults come in seeded episodes of consecutive frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from recoverdrive.types import FaultLabel, Frame


def schedule_episodes(n_frames: int, fault_ratio: float, rng: np.random.Generator,
                      min_len: int = 2, max_len: int = 6) -> list[int | None]:
    """Episode id for each frame (None = clean).

    About `fault_ratio` of frames are faulty, grouped into episodes of `min_len`..`max_len` frames with at
    least one clean frame between episodes.
    """
    target = round(n_frames * fault_ratio)
    if target < min_len:
        return [None] * n_frames
    k_min = math.ceil(target / max_len)
    k_max = min(target // min_len, n_frames - target + 1)  # inner gaps need one clean frame each
    if k_min > k_max:
        raise ValueError(f"cannot fit {target} faulty frames into {n_frames} with episode length {min_len}-{max_len}")
    k = int(rng.integers(k_min, k_max + 1))

    lengths = [min_len] * k
    for _ in range(target - k * min_len):
        open_slots = [i for i in range(k) if lengths[i] < max_len]
        lengths[int(rng.choice(open_slots))] += 1

    gaps = [0] + [1] * (k - 1) + [0]
    for _ in range(n_frames - target - (k - 1)):
        gaps[int(rng.integers(0, k + 1))] += 1

    schedule: list[int | None] = []
    for ep in range(k):
        schedule += [None] * gaps[ep] + [ep] * lengths[ep]
    schedule += [None] * gaps[k]
    return schedule


def inject_c1(frame: Frame, target: str, source: str) -> Frame:
    """C1, wrong viewpoint: `target` camera reports the image of `source`, keeping its own calibration."""
    cams = dict(frame.cameras)
    src = frame.cameras[source]
    cams[target] = replace(frame.cameras[target], image=src.image, image_path=src.image_path)
    return replace(frame, cameras=cams)


@dataclass
class FaultInjector:
    """Applies the scheduled fault to each frame and returns the ground-truth label."""

    schedule: list[int | None]
    fault_id: str
    params: dict

    def apply(self, frame: Frame) -> tuple[Frame, FaultLabel]:
        episode = self.schedule[frame.frame_idx]
        if episode is None:
            return frame, FaultLabel.none()
        if self.fault_id == "C1":
            target = self.params["target"]
            # C1 has no severity levels and is always treated as high (spec §5.2).
            label = FaultLabel(fault_id="C1", sensor=f"camera:{target}", severity="high", episode_id=episode)
            return inject_c1(frame, target, self.params["source"]), label
        raise NotImplementedError(f"fault {self.fault_id} is not implemented yet")
