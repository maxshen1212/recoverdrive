import numpy as np

from recoverdrive.checks import (
    CameraConsistency,
    build_report,
    iou_matrix,
    match_boxes,
    match_rate,
    rank_sensors,
)
from recoverdrive.types import CameraCheck


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_iou_matrix_empty():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_match_boxes_is_one_to_one():
    gt = np.array([[0, 0, 10, 10], [1, 0, 11, 10]])
    det = np.array([[0, 0, 10, 10]])
    matches = match_boxes(gt, det, 0.3)
    assert len(matches) == 1 and matches[0][1] == 0


def test_match_boxes_threshold():
    gt = np.array([[0, 0, 10, 10]])
    det = np.array([[8, 0, 18, 10]])  # IoU = 20 / 180
    assert match_boxes(gt, det, 0.3) == []
    assert len(match_boxes(gt, det, 0.1)) == 1


def test_match_rate_without_gt_is_none():
    assert match_rate(0, 0) is None
    assert match_rate(3, 4) == 0.75


def _consistency(name, n_gt, n_matched):
    check = CameraCheck(camera=name, n_gt=n_gt, n_det=n_matched, n_matched=n_matched,
                        match_rate=match_rate(n_matched, n_gt))
    return CameraConsistency(check, np.zeros((n_gt, 4)), [], np.zeros((n_matched, 4)), [])


def test_build_report_deltas_and_flags():
    first = build_report({"CAM_FRONT": _consistency("CAM_FRONT", 4, 4)}, 0.8, 1000, prev=None)
    assert first.cameras["CAM_FRONT"].match_rate_delta is None and first.point_count_delta is None

    second = build_report({"CAM_FRONT": _consistency("CAM_FRONT", 4, 1)}, 0.8, 900, prev=first)
    assert second.cameras["CAM_FRONT"].match_rate_delta == -0.75
    assert second.point_count_delta == -100

    third = build_report({"CAM_FRONT": _consistency("CAM_FRONT", 0, 0)}, None, 900, prev=second)
    assert third.cameras["CAM_FRONT"].match_rate is None
    assert "camera:CAM_FRONT:no_reference" in third.flags


# --- T1: LiDAR health, agreement, flags ---------------------------------------------------------------

from recoverdrive.checks import (
    CheckThresholds,
    compute_flags,
    flagged_sensors,
    flags_by_sensor,
    lidar_sector_counts,
)
from recoverdrive.types import CheckReport


def test_sector_counts_quadrants():
    # Bins start at -pi: [-pi, -pi/2), [-pi/2, 0), [0, pi/2), [pi/2, pi). Near (1 m) and far (80 m) points dropped.
    pts = np.array([[-10, -10, 0, 0], [10, -10, 0, 0], [10, 10, 0, 0], [12, 9, 0, 0], [-10, 10, 0, 0],
                    [1, 0, 0, 0], [80, 0, 0, 0]], float)
    assert lidar_sector_counts(pts, n_sectors=4).tolist() == [1, 1, 2, 1]


def _report(points=34000, sectors=None, **cams):
    cameras = {name: CameraCheck(name, n_gt=n_gt, n_det=n_gt, n_matched=m, match_rate=match_rate(m, n_gt))
               for name, (n_gt, m) in cams.items()}
    return CheckReport(cameras=cameras, lidar_visible_ratio=0.8, point_count=points,
                       lidar_n_in_range=10, lidar_sector_counts=sectors if sectors is not None else [500] * 12,
                       lidar_camera_agreement=0.9, lidar_camera_n=5)


def test_lidar_health_empty_cloud():
    counts = lidar_sector_counts(np.zeros((0, 4)), 12)
    assert counts.tolist() == [0] * 12
    flags = compute_flags(_report(points=0, sectors=counts.tolist()), CheckThresholds())
    assert "lidar:low_points" in flags and "lidar:sector_gap" in flags


def test_flags_skip_camera_without_reference():
    flags = compute_flags(_report(CAM_FRONT=(0, 0)), CheckThresholds())
    assert not any(f.startswith("camera:") for f in flags)


def test_flags_low_match_requires_min_gt():
    thr = CheckThresholds(min_gt=2)
    assert "camera:CAM_FRONT:low_match" not in compute_flags(_report(CAM_FRONT=(1, 0)), thr)
    assert "camera:CAM_FRONT:low_match" in compute_flags(_report(CAM_FRONT=(4, 1)), thr)
    assert compute_flags(_report(CAM_FRONT=(4, 4)), thr) == []


def test_flags_point_drop_and_disagree():
    r = _report(points=15000)
    r.point_count_delta = -19000
    r.lidar_camera_agreement, r.lidar_camera_n = 0.2, 5
    flags = compute_flags(r, CheckThresholds())
    assert {"lidar:low_points", "lidar:point_drop", "lidar:camera_disagree"} <= set(flags)


def test_flagged_sensors_maps_flags():
    flags = ["camera:CAM_FRONT:low_match", "camera:CAM_BACK:no_reference", "lidar:sector_gap", "agent:invalid",
             "camera:CAM_FRONT:frozen"]
    assert flagged_sensors(flags) == {"camera:CAM_FRONT", "lidar", "agent"}
    assert flags_by_sensor(flags)["camera:CAM_FRONT"] == ["low_match", "frozen"]


def test_rank_sensors_most_flags_first():
    flags = ["camera:CAM_BACK:low_match", "lidar:low_points", "lidar:point_drop", "camera:CAM_FRONT:no_reference"]
    assert rank_sensors(flags) == ["lidar", "camera:CAM_BACK"]
