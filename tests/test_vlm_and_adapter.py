import json

from recoverdrive.adapters.nuscenes import map_category
from recoverdrive.types import CameraCheck, CheckReport
from recoverdrive.vlm import diagnosis_schema, parse_diagnosis, summarize_report

CATEGORIES = {"vehicle.car": "car", "vehicle.bus": "bus", "human.pedestrian.adult": "pedestrian"}
SCHEMA = diagnosis_schema(["camera:CAM_FRONT"])


def test_map_category():
    assert map_category("vehicle.car", CATEGORIES) == "car"
    assert map_category("vehicle.bus.rigid", CATEGORIES) == "bus"
    assert map_category("vehicle.carriage", CATEGORIES) is None
    assert map_category("human.pedestrian.stroller", CATEGORIES) is None


def test_parse_diagnosis_valid():
    d = parse_diagnosis(json.dumps({"status": "fault", "sensor": "camera:CAM_FRONT", "confidence": 0.9}), SCHEMA)
    assert d is not None and d.status == "fault" and d.confidence == 0.9


def test_parse_diagnosis_invalid():
    assert parse_diagnosis("not json", SCHEMA) is None
    assert parse_diagnosis(json.dumps({"status": "fault", "sensor": "camera:CAM_FRONT"}), SCHEMA) is None
    assert parse_diagnosis(json.dumps({"status": "bad", "sensor": "none", "confidence": 0.5}), SCHEMA) is None
    assert parse_diagnosis(json.dumps({"status": "ok", "sensor": "none", "confidence": 1.5}), SCHEMA) is None


def test_summarize_report():
    report = CheckReport(
        cameras={"CAM_FRONT": CameraCheck("CAM_FRONT", n_gt=5, n_det=4, n_matched=1, match_rate=0.2,
                                          match_rate_delta=-0.6)},
        lidar_visible_ratio=0.7, point_count=34000, flags=[])
    text = summarize_report(report, 50, 0.3)
    assert "match rate 0.20 (previous frame: 0.80)" in text
    empty = CheckReport(cameras={"CAM_FRONT": CameraCheck("CAM_FRONT", 0, 2, 0, None)}, lidar_visible_ratio=None,
                        point_count=0, flags=["camera:CAM_FRONT:no_reference"])
    assert "cannot be cross-checked" in summarize_report(empty, 50, 0.3)
