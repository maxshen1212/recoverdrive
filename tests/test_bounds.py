import pytest

from recoverdrive.bounds import clopper_pearson, episode_violation_bounds


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)
    assert clopper_pearson(10, 10)[1] == 1.0


def test_episode_bounds_counts_episodes():
    rows = [{"run": "r1", "fault_id": "C1", "episode_id": 0, "violated": False},
            {"run": "r1", "fault_id": "C1", "episode_id": 0, "violated": True},
            {"run": "r1", "fault_id": "C1", "episode_id": 1, "violated": False},
            {"run": "r2", "fault_id": "L1", "episode_id": 0, "violated": False}]
    out = {r["fault_id"]: r for r in episode_violation_bounds(rows)}
    assert out["C1"]["episodes"] == 2 and out["C1"]["violated"] == 1 and out["C1"]["rate"] == 0.5
    assert out["L1"]["violated"] == 0 and out["L1"]["upper"] == pytest.approx(0.975)
