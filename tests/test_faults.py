import itertools

import numpy as np
import pytest
from fakes import IDENTITY, make_frame

from recoverdrive.faults import FaultInjector, inject_c1, schedule_episodes


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
    raw = json.dumps({"verdict": "ok", "fault_type": "none", "confidence": 0.9, "evidence": "x"})
    st = AgentFaultState("A1", "medium", {"n_affected": 2, "kind": "missing_field"})
    assert st.affects(0) and st.affects(1) and not st.affects(2)
    assert "verdict" not in json.loads(st.corrupt_output(raw))
    trunc = AgentFaultState("A1", "low", {"n_affected": 1, "kind": "truncate"})
    with pytest.raises(json.JSONDecodeError):
        json.loads(trunc.corrupt_output(raw))


# --- T8: C2, C3, C4, L2, L4, A3 ----------------------------------------------------------------------------

from recoverdrive.faults import FaultEvent, inject_c2, inject_c3, inject_l2


def _img(h=40, w=60):
    rng = np.random.default_rng(0)
    return rng.integers(30, 220, (h, w, 3), dtype=np.uint8)


def test_c2_rotate_and_shift():
    img = _img()
    rot = inject_c2(img, "rotate", 10.0, 1)
    assert rot.shape == img.shape and not np.array_equal(rot, img) and (rot[0, 0] == 0).all()  # black corner
    sh = inject_c2(img, "shift", 0.1, 1)  # 6 px right, 4 px down
    np.testing.assert_array_equal(sh[4:, 6:], img[:-4, :-6])
    assert (sh[:4] == 0).all() and (sh[:, :6] == 0).all()
    assert (inject_c2(img, "shift", 0.1, -1)[-4:] == 0).all()


def test_c3_variants():
    img = _img()
    rng = np.random.default_rng(0)
    assert (inject_c3(img, "black", None, rng) == 0).all()
    blur = inject_c3(img, "blur", 5.0, rng)
    assert blur.std() < img.std()
    noisy = inject_c3(img, "noise", 30.0, rng)
    assert noisy.dtype == np.uint8 and np.abs(noisy.astype(int) - img).mean() > 10
    assert inject_c3(img, "overexpose", 4.0, rng).mean() > img.mean() * 1.5
    plan = plan_faults("C3", 200, np.random.default_rng(5), CFG, CAMS)
    blacks = [e for e in plan if e is not None and e.params["subtype"] == "black"]
    assert blacks and all(e.severity == "high" for e in blacks)
    assert {e.params["camera"] for e in plan if e is not None} <= set(CAMS)


def _cam_frame(idx, value):
    f = make_frame(idx)
    f.cameras["CAM_FRONT"].image = np.full((4, 4, 3), value, np.uint8)
    return f


def test_c4_lag_and_freeze():
    originals = {i: _cam_frame(i, 10 * i) for i in range(6)}
    for lag, expected in ((1, 40), (2, 30), ("freeze", 10)):
        ev = FaultEvent("C4", "camera:CAM_FRONT", "low", 0, {"camera": "CAM_FRONT", "lag": lag, "start": 2})
        plan = [None, None, ev, ev, ev, ev]
        out, label = SensorFaultInjector(plan).apply(originals[5], originals)
        assert label.fault_id == "C4" and (out.cameras["CAM_FRONT"].image == expected).all(), lag
        assert (originals[5].cameras["CAM_FRONT"].image == 50).all()  # original untouched


def test_l2_shift_moves_points():
    pts = np.array([[10.0, 0, 0, 1], [0, 5, 0, 1]])
    out = inject_l2(pts, 1.0, 90.0, 0.0)
    np.testing.assert_allclose(out[:, :2], [[1.0, 10.0], [-4.0, 0.0]], atol=1e-9)
    assert pts[0, 0] == 10.0 and out[0, 3] == 1


def test_l4_empty():
    plan = plan_faults("L4", 10, np.random.default_rng(0), CFG, CAMS)
    idx = next(i for i, e in enumerate(plan) if e is not None)
    f = make_frame(idx)
    f.lidar = LidarData(points=_ring_points(100), extrinsic=IDENTITY, ego_pose=IDENTITY, timestamp=0)
    out, label = SensorFaultInjector(plan).apply(f)
    assert out.lidar.points.shape == (0, 4) and label.severity == "high" and label.sensor == "lidar"


def test_all_faults_plan_and_apply():
    from recoverdrive.faults import ALL_FAULTS

    for fid in ALL_FAULTS:
        plan = plan_faults(fid, 12, np.random.default_rng(1), CFG, CAMS)
        for i, ev in enumerate(plan):
            f = make_frame(i)
            for c in CAMS:
                f.cameras.setdefault(c, f.cameras["CAM_FRONT"])
            f.lidar = LidarData(points=_ring_points(50), extrinsic=IDENTITY, ego_pose=IDENTITY, timestamp=0)
            _, label = SensorFaultInjector(plan).apply(f, {i: f})
            assert label.fault_id == (ev.fault_id if ev else "none")


def test_a3_timeout():
    from fakes import FakeOpenAI

    from recoverdrive.agent import AgentCaller
    from recoverdrive.vlm import VLMClient, ours_schema

    caller = AgentCaller(VLMClient("http://x", "m", timeout_s=30, client=FakeOpenAI()))
    caller.new_frame(AgentFaultState("A3", "low", {"n_affected": 1}))
    att = caller.call([{"role": "user", "content": "x"}], ours_schema(["lidar"]))
    assert att.timeout and not att.valid and att.latency_s == 30 and att.clean_data is not None
    assert caller.call([{"role": "user", "content": "x"}], ours_schema(["lidar"])).valid
