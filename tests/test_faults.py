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


# --- T2: fault plans, L1, A1 -------------------------------------------------------------------------------

import json

from recoverdrive.faults import (
    AgentFaultState,
    SensorFaultInjector,
    beam_index,
    inject_l1,
    plan_faults,
)
from recoverdrive.types import LidarData

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CFG = {"fault_ratio": 0.5, "min_episode_len": 2, "max_episode_len": 6}


def test_plan_faults_seeded_and_labeled():
    a = plan_faults("L1", 40, np.random.default_rng(3), CFG, CAMS)
    b = plan_faults("L1", 40, np.random.default_rng(3), CFG, CAMS)
    assert a == b
    assert sum(e is not None for e in a) == 20
    for ev in a:
        if ev is not None:
            assert ev.sensor == "lidar" and ev.severity in ("low", "medium", "high")
            same = [e for e in a if e is not None and e.episode_id == ev.episode_id]
            assert all(e is ev for e in same)  # one event object per episode
            assert a[ev.params["start"]] is ev and (ev.params["start"] == 0 or a[ev.params["start"] - 1] is not ev)


def test_c1_is_high_severity():
    plan = plan_faults("C1", 20, np.random.default_rng(0), {**CFG, "c1": {"target": "CAM_FRONT", "source": "CAM_BACK"}},
                       CAMS)
    evs = [e for e in plan if e is not None]
    assert evs and all(e.severity == "high" and e.sensor == "camera:CAM_FRONT" for e in evs)


def _ring_points(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    az = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(3, 40, n)
    return np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-1.5, 1, n), np.ones(n)], axis=1)


def test_l1_random_drop_ratio():
    pts = _ring_points()
    out = inject_l1(pts, "random", 0.6, np.random.default_rng(0))
    assert abs(len(out) / len(pts) - 0.4) < 0.03


def test_l1_sector_removes_azimuth_band():
    pts = _ring_points()
    out = inject_l1(pts, "sector", 90.0, np.random.default_rng(0), azimuth_deg=0.0)
    az = np.degrees(np.arctan2(out[:, 1], out[:, 0]))
    assert np.all(np.abs(az) > 45)
    assert abs(len(out) / len(pts) - 0.75) < 0.03


def test_l1_beam_drop():
    elevs = np.linspace(-30, 10, 32)  # one point per beam, at 20 m
    pts = np.stack([20 * np.ones(32), np.zeros(32), 20 * np.tan(np.radians(elevs)), np.ones(32)], axis=1)
    assert len(set(beam_index(pts))) == 32
    out = inject_l1(pts, "beam", 0.5, np.random.default_rng(0), beam_rng=np.random.default_rng(7))
    assert len(out) == 16


def test_l1_injector_keeps_beams_fixed_within_episode():
    plan = plan_faults("L1", 12, np.random.default_rng(1), CFG, CAMS)
    ev = next(e for e in plan if e is not None)
    ev.params.update(subtype="beam", level=0.5)
    inj = SensorFaultInjector(plan)
    idxs = [i for i, e in enumerate(plan) if e is ev][:2]
    pts = _ring_points(5000)
    kept = []
    for i in idxs:
        f = make_frame(i)
        f.lidar = LidarData(points=pts, extrinsic=IDENTITY, ego_pose=IDENTITY, timestamp=0)
        out, label = inj.apply(f)
        assert label.fault_id == "L1" and label.sensor == "lidar"
        kept.append(set(beam_index(out.lidar.points)))
    assert kept[0] == kept[1]


def test_a1_corrupts_first_n_attempts():
    raw = json.dumps({"evidence": "x", "status": "ok", "sensor": "none", "confidence": 0.9})
    st = AgentFaultState("A1", "medium", {"n_affected": 2, "kind": "missing_field"})
    assert st.affects(0) and st.affects(1) and not st.affects(2)
    assert "status" not in json.loads(st.corrupt_output(raw))
    trunc = AgentFaultState("A1", "low", {"n_affected": 1, "kind": "truncate"})
    with pytest.raises(json.JSONDecodeError):
        json.loads(trunc.corrupt_output(raw))
