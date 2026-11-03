import json

import pytest

from recoverdrive.metrics import aggregate_seeds, e1_counts, e1_table, markdown_table
from recoverdrive.runner import RunSpec, check_test_allowed, run_all, run_paths, run_status


def rec(fault, sensor, alarm, pred, run_fault="C1", seed=0, severity="high"):
    return {"run_fault": run_fault, "seed": seed,
            "label": {"fault_id": fault, "sensor": sensor, "severity": severity if fault != "none" else None},
            "methods": {"m": {"alarm": alarm, "pred_sensor": pred}}}


def test_e1_metrics_counts():
    records = [rec("C1", "camera:CAM_FRONT", True, "camera:CAM_FRONT"),  # TP, right sensor
               rec("C1", "camera:CAM_FRONT", True, "lidar"),  # TP, wrong sensor
               rec("C1", "camera:CAM_FRONT", False, "none"),  # FN
               rec("none", "none", True, "lidar"),  # FP
               rec("none", "none", False, "none"),  # TN
               rec("none", "none", False, "none")]  # TN
    c = e1_counts((r, r["methods"]["m"]) for r in records)
    assert c["n"] == 6 and c["n_fault"] == 3
    assert c["P"] == pytest.approx(2 / 3) and c["R"] == pytest.approx(2 / 3) and c["F1"] == pytest.approx(2 / 3)
    assert c["sensor_acc"] == 0.5 and c["false_alarm"] == pytest.approx(1 / 3)
    rows = e1_table(records + [rec("L1", "lidar", True, "lidar", run_fault="L1")], ["m"], ["C1", "L1"])
    assert [r["fault"] for r in rows] == ["C1", "L1", "all"] and rows[1]["R"] == 1.0 and rows[2]["n"] == 7


def test_aggregate_seeds_mean_std_and_table():
    t0 = [{"method": "m", "fault": "C1", "R": 0.5, "n": 10}]
    t1 = [{"method": "m", "fault": "C1", "R": 0.7, "n": 10}]
    agg = aggregate_seeds([t0, t1])[0]
    assert agg["R"] == pytest.approx(0.6) and agg["R_std"] == pytest.approx(0.1)
    md = markdown_table([agg], ["method", "fault", "R"])
    assert "0.600 ± 0.100" in md


def test_runner_resumes_completed_runs(tmp_path):
    class Env:
        out_dir = tmp_path

    done_spec, partial_spec = RunSpec("dev", "a", "C1", 0), RunSpec("dev", "b", "C1", 0)
    jsonl, done = run_paths(tmp_path, done_spec)
    jsonl.parent.mkdir(parents=True)
    jsonl.write_text("{}\n")
    done.write_text(json.dumps({"run": done_spec.name, "frames": 1, "errors": 0, "wall_s": 0}))
    run_paths(tmp_path, partial_spec)[0].write_text("{}\n")
    assert run_status(tmp_path, done_spec) == "done" and run_status(tmp_path, partial_spec) == "partial"
    ran = []

    def fake_run(spec, env):
        ran.append(spec)
        return {"run": spec.name, "frames": 2, "errors": 0, "wall_s": 0}

    results = run_all([done_spec, partial_spec], Env(), workers=2, run_fn=fake_run, log=lambda s: None)
    assert ran == [partial_spec] and len(results) == 2


def test_runner_refuses_unfrozen_test(tmp_path):
    with pytest.raises(RuntimeError, match="not frozen"):
        check_test_allowed({"frozen": False}, tmp_path, git_status=lambda p: "")
    with pytest.raises(RuntimeError, match="uncommitted"):
        check_test_allowed({"frozen": True}, tmp_path, git_status=lambda p: " M configs/eval.yaml\n")
