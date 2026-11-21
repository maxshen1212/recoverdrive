import numpy as np
from fakes import IDENTITY, make_frame

from recoverdrive.scenario import Scenario
from recoverdrive.types import LidarData


def test_scenario_defaults_match_e3():
    sc = Scenario.from_condition("pedestrian", "C1")
    assert (sc.walker_speed_mps, sc.lead_brake_s, sc.fault, sc.fault_start_m, sc.fault_end_m) == (1.6, 3.0, "C1",
                                                                                                    -25.0, 30.0)
    assert sc.trigger_offset_m is None and sc.trigger_m(1.5) == 41.5
    assert Scenario.from_condition("lead_brake", "clean").fault == "none"
    assert Scenario("pedestrian", trigger_offset_m=-4.0).trigger_m(1.5) == 36.0


def test_apply_fault_only_inside_window():
    sc = Scenario("pedestrian")  # C1, window -25..+30 m around the trigger
    rng = np.random.default_rng(0)
    frame, label = sc.apply_fault(make_frame(), progress_m=40.0, trigger_m=40.0, rng=rng)
    assert label.fault_id == "C1" and (frame.cameras["CAM_FRONT"].image == 255).all()
    frame, label = sc.apply_fault(make_frame(), progress_m=10.0, trigger_m=40.0, rng=rng)
    assert label.fault_id == "none" and (frame.cameras["CAM_FRONT"].image == 0).all()


def test_c3_black_and_c2_rotate_labels():
    rng = np.random.default_rng(0)
    base = make_frame()
    base.cameras["CAM_FRONT"].image[:] = 128
    frame, label = Scenario("pedestrian", fault="C3_black").apply_fault(base, 40.0, 40.0, rng)
    assert (label.fault_id, label.severity) == ("C3", "high") and frame.cameras["CAM_FRONT"].image.max() == 0
    _, label = Scenario("pedestrian", fault="C2_rotate", fault_level=6.0).apply_fault(base, 40.0, 40.0, rng)
    assert (label.fault_id, label.severity) == ("C2", "medium")
    _, label = Scenario("pedestrian", fault="C2_rotate", fault_level=1.0).apply_fault(base, 40.0, 40.0, rng)
    assert label.severity == "low"


def test_scenario_from_params():
    from recoverdrive.scenario import CARLA_SEARCH, scenario_from_params
    p = {"trigger_offset_m": 2.0, "walker_speed_mps": 2.2, "lead_brake_s": 1.5, "fault": "C2_rotate",
         "level_u": 0.5, "fault_start_m": -10.0, "fault_len_m": 20.0}
    sc = scenario_from_params("pedestrian", p)
    assert (sc.trigger_offset_m, sc.walker_speed_mps, sc.lead_brake_s) == (2.0, 2.2, 3.0)  # brake only for lead_brake
    assert (sc.fault, sc.fault_level, sc.fault_start_m, sc.fault_end_m) == ("C2_rotate", 6.0, -10.0, 10.0)
    lead = scenario_from_params("lead_brake", {**p, "fault": "C1"})
    assert lead.lead_brake_s == 1.5 and lead.walker_speed_mps == 1.6 and lead.fault_level is None
    assert set(CARLA_SEARCH["pedestrian"]) >= {"route", "fault", "walker_speed_mps"}
    assert "walker_speed_mps" not in CARLA_SEARCH["lead_brake"]


def test_l1_sector_drops_points_ahead():
    pts = np.array([[10.0, 0.0, 0.0, 1.0], [-10.0, 0.0, 0.0, 1.0], [0.0, 10.0, 0.0, 1.0]])
    frame = make_frame()
    frame.lidar = LidarData(points=pts, extrinsic=IDENTITY, ego_pose=IDENTITY, timestamp=0)
    out, label = Scenario("pedestrian", fault="L1_sector", fault_level=90.0).apply_fault(
        frame, 40.0, 40.0, np.random.default_rng(0))
    assert (label.fault_id, label.sensor, label.severity) == ("L1", "lidar", "medium")
    assert len(out.lidar.points) == 2 and not (out.lidar.points[:, 0] > 5).any()
