import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from fakes import DictCache, FakeAdapter, FakeDetector

from recoverdrive.detectors.sam3 import CachedDetector, CacheMissError
from recoverdrive.falsify import (
    CPU_FAULTS,
    FAULT_SPACES,
    evaluate_trial,
    event_from_params,
    severity_of,
    window_frames,
    window_plan,
)
from recoverdrive.runner import EvalEnv, RunSpec, run_one

REPO = Path(__file__).resolve().parents[1]


def make_env(out_dir, methods=("checks_only",)):
    cfg = yaml.safe_load((REPO / "configs/eval.yaml").read_text())
    cfg["data"]["cameras"] = ["CAM_FRONT", "CAM_BACK"]
    return EvalEnv(cfg, FakeAdapter(6), FakeDetector(), None, Path(out_dir), list(methods))


def rows(out_dir, spec):
    path = Path(out_dir) / "runs" / f"{spec.name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_one_defaults_unchanged(tmp_path):
    spec = RunSpec("dev", "s", "C1", 0)
    run_one(spec, make_env(tmp_path / "a"))
    run_one(spec, make_env(tmp_path / "b"), plan=None, frames=None)
    keep = ("frame_idx", "label", "flags", "methods")
    a = [{k: r[k] for k in keep} for r in rows(tmp_path / "a", spec)]
    b = [{k: r[k] for k in keep} for r in rows(tmp_path / "b", spec)]
    assert a == b and len(a) == 6


def test_run_one_window_only_processes_frames(tmp_path):
    spec = RunSpec("dev", "s", "C1", 0)
    event = event_from_params("C1", {})
    run_one(spec, make_env(tmp_path), plan=window_plan(6, 2, 2, event), frames=range(1, 5))
    recs = rows(tmp_path, spec)
    assert [r["frame_idx"] for r in recs] == [1, 2, 3, 4]
    assert [r["label"]["fault_id"] for r in recs] == ["none", "C1", "C1", "none"]


def test_severity_of_boundaries():
    assert severity_of("C3", "blur", 4.99) == "low"
    assert severity_of("C3", "blur", 5.0) == "medium"
    assert severity_of("C3", "blur", 10.0) == "high"
    assert severity_of("C3", "black", None) == "high"
    assert severity_of("C1", None, None) == "high"
    assert severity_of("L2", None, (0.1, 2.0)) == "medium"  # the stronger of translation and yaw counts
    assert severity_of("C4", None, "freeze") == "high"


def test_event_from_params_ranges():
    ev = event_from_params("C3", {"camera": "CAM_FRONT", "subtype": "blur", "level": 50.0, "seed": 1})
    assert ev.params["level"] == FAULT_SPACES["C3"]["blur"][1] and ev.severity == "high"
    ev = event_from_params("L1", {"subtype": "sector", "level": 400.0, "azimuth_deg": 0.0, "seed": 1})
    assert ev.params["level"] == 360.0 and ev.sensor == "lidar"
    ev = event_from_params("C2", {"camera": "CAM_BACK", "subtype": "rotate", "level": -3.0, "sign": 1})
    assert ev.params["level"] == 0.0 and ev.sensor == "camera:CAM_BACK" and ev.severity == "low"
    assert window_frames(40, 10, 3) == range(7, 16)
    assert window_frames(40, 1, 3) == range(7)
    assert set(CPU_FAULTS) == {"C1", "C4", "L1", "L2", "L4"}


def test_evaluate_trial_scores_spec(tmp_path):
    env = make_env(tmp_path / "base")
    p = {"start": 2, "duration": 2, "camera": "CAM_FRONT", "subtype": "black", "seed": 0}
    res = evaluate_trial(env, "s", "C3", p, "checks_only", "R1", tmp_path / "trial")
    # the fake frames have no LiDAR points, so the LiDAR is flagged on every frame and the method never says healthy
    assert math.isfinite(res.robustness) and res.satisfied == (res.n_violations == 0)
    assert res.params == p and Path(res.run_dir).exists()
    recs = rows(tmp_path / "trial", RunSpec("dev", "s", "C3", 0))
    assert [r["frame_idx"] for r in recs] == [0, 1, 2, 3, 4, 5]  # window: 3 warm-up (cut at 0), 2 fault, 3 tail
    assert [r["label"]["fault_id"] for r in recs] == ["none", "none", "C3", "C3", "none", "none"]


def test_cached_detector_cache_only_raises_on_miss():
    cache = DictCache()
    det = CachedDetector(cache, ["car"], allow_miss=False)
    seen = np.zeros((4, 4, 3), np.uint8)
    cache.set(det.key(seen), [{"box": [0, 0, 1, 1], "label": "car", "score": 0.9}])
    assert len(det.detect(seen)) == 1
    with pytest.raises(CacheMissError):
        det.detect(np.ones((4, 4, 3), np.uint8))
    assert det._detector is None
