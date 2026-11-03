import itertools

import numpy as np
import pytest

from recoverdrive.faults import FaultInjector, inject_c1, schedule_episodes
from recoverdrive.types import CameraData, Frame, Pose

IDENTITY = Pose(np.array([1.0, 0, 0, 0]), np.zeros(3))


def episodes(schedule):
    """List of (episode_id, length) in order of appearance."""
    out = []
    for i, ep in enumerate(schedule):
        if ep is not None and (i == 0 or schedule[i - 1] != ep):
            out.append([ep, 0])
        if ep is not None:
            out[-1][1] += 1
    return out


@pytest.mark.parametrize("n_frames", [10, 39, 40, 41])
@pytest.mark.parametrize("seed", range(5))
def test_schedule_properties(n_frames, seed):
    sched = schedule_episodes(n_frames, 0.5, np.random.default_rng(seed), min_len=2, max_len=6)
    assert len(sched) == n_frames
    assert sum(ep is not None for ep in sched) == round(n_frames * 0.5)
    eps = episodes(sched)
    assert all(2 <= length <= 6 for _, length in eps)
    assert [ep for ep, _ in eps] == list(range(len(eps)))  # each episode id is one contiguous run
    for a, b in itertools.pairwise(sched):
        assert a is None or b is None or a == b  # clean frame between episodes


def test_schedule_is_reproducible():
    a = schedule_episodes(39, 0.5, np.random.default_rng(0))
    b = schedule_episodes(39, 0.5, np.random.default_rng(0))
    assert a == b


def make_frame(idx=0):
    front = CameraData(image=np.zeros((4, 4, 3), np.uint8), intrinsic=np.eye(3) * 2, extrinsic=IDENTITY,
                       ego_pose=IDENTITY, timestamp=0, image_path="front.jpg")
    back = CameraData(image=np.full((4, 4, 3), 255, np.uint8), intrinsic=np.eye(3) * 3, extrinsic=IDENTITY,
                      ego_pose=IDENTITY, timestamp=0, image_path="back.jpg")
    return Frame(source="nuscenes", scene_id="s", frame_idx=idx, timestamp=0,
                 cameras={"CAM_FRONT": front, "CAM_BACK": back}, lidar=None, ego_pose=IDENTITY, gt_boxes=[])


def test_inject_c1_swaps_image_keeps_calibration():
    frame = make_frame()
    faulty = inject_c1(frame, "CAM_FRONT", "CAM_BACK")
    assert (faulty.cameras["CAM_FRONT"].image == 255).all()
    assert faulty.cameras["CAM_FRONT"].image_path == "back.jpg"
    np.testing.assert_array_equal(faulty.cameras["CAM_FRONT"].intrinsic, np.eye(3) * 2)
    assert (frame.cameras["CAM_FRONT"].image == 0).all()  # original frame untouched


def test_injector_labels():
    inj = FaultInjector(schedule=[None, 0], fault_id="C1", params={"target": "CAM_FRONT", "source": "CAM_BACK"})
    _, clean = inj.apply(make_frame(0))
    assert clean.fault_id == "none" and clean.sensor == "none"
    faulty, label = inj.apply(make_frame(1))
    assert label.fault_id == "C1" and label.sensor == "camera:CAM_FRONT" and label.episode_id == 0
    assert (faulty.cameras["CAM_FRONT"].image == 255).all()
