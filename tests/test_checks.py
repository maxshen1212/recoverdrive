import numpy as np

from recoverdrive.checks import CameraConsistency, build_report, iou_matrix, match_boxes, match_rate
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
