import pytest

from recoverdrive.bootstrap import block_ids, bootstrap_metrics, merge_runs
from recoverdrive.metrics import e1_counts, method_frames

LAYOUT = [None, None, 0, 0, None, 1, 1, None, None, None]  # episode id per frame, None = clean
CLEAN = {"fault_id": "none", "sensor": "none", "severity": None, "episode_id": None}


def _out(style: str, ep, sensor: str, idx: int) -> dict:
    right = {"alarm": True, "pred_sensor": sensor, "action": {"action": "slow_down"}}
    quiet = {"alarm": False, "pred_sensor": "none", "action": {"action": "continue"}}
    false_alarm = {"alarm": True, "pred_sensor": sensor, "action": {"action": "stop"}}
    if style == "good":
        return right if ep is not None else quiet
    if style == "bad":
        return quiet if ep is not None else false_alarm
    # mixed: right on episode 0, misses episode 1, one false alarm on the first clean frame
    if ep is not None:
        return right if ep == 0 else quiet
    return false_alarm if idx == 0 else quiet


def _synthetic(**styles) -> list[dict]:
    """Two runs (C1 on CAM_FRONT, L1 on the LiDAR), 10 frames each in the LAYOUT order."""
    recs = []
    for fault, sensor in (("C1", "camera:CAM_FRONT"), ("L1", "lidar")):
        for idx, ep in enumerate(LAYOUT):
            label = CLEAN if ep is None else {"fault_id": fault, "sensor": sensor, "severity": "high", "episode_id": ep}
            recs.append({"scene": "s", "seed": 0, "run_fault": fault, "frame_idx": idx, "label": label,
                         "methods": {m: _out(style, ep, sensor, idx) for m, style in styles.items()}})
    return recs


def test_block_ids_split_episodes_and_clean_gaps():
    ids = block_ids(_synthetic(a="good")[:10])  # one run: gaps [0,1], [4], [7,8,9]; episodes [2,3], [5,6]
    assert len(set(ids)) == 5
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[7] == ids[9] and ids[4] != ids[0]


def test_identical_methods_have_zero_difference():
    rows = bootstrap_metrics(_synthetic(a="good", b="good"), ["a", "b"], reference="a", n_boot=200)
    d = next(r for r in rows if r["method"] == "a - b" and r["metric"] == "F1")
    assert d["lo"] == d["hi"] == 0.0


def test_clear_gap_excludes_zero():
    rows = bootstrap_metrics(_synthetic(a="good", b="bad"), ["a", "b"], reference="a", n_boot=200)
    assert next(r for r in rows if r["method"] == "a - b" and r["metric"] == "F1")["lo"] > 0


def test_point_estimate_equals_pooled_e1():
    recs = _synthetic(a="mixed")
    f1 = next(r for r in bootstrap_metrics(recs, ["a"], "a", n_boot=50) if r["method"] == "a" and r["metric"] == "F1")
    assert f1["point"] == pytest.approx(e1_counts(method_frames(recs, "a"))["F1"])


def test_same_seed_same_result():
    recs = _synthetic(a="mixed", b="bad")
    assert bootstrap_metrics(recs, ["a", "b"], "a", n_boot=100, seed=3) == \
        bootstrap_metrics(recs, ["a", "b"], "a", n_boot=100, seed=3)


def test_merge_runs_prefix_and_conflicts():
    a, b = _synthetic(a="good"), _synthetic(a="bad")
    merged = merge_runs([(a, None), (b, "v2")])
    assert set(merged[0]["methods"]) == {"a", "v2:a"}
    with pytest.raises(ValueError):
        merge_runs([(a, None), (b, None)])  # same method name twice
