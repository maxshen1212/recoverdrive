import pytest

from recoverdrive.formal import (
    DEFAULT_CONTROLLER_SENSORS,
    explore_controller,
    verify_controller,
    verify_shield,
)
from recoverdrive.recovery import RecoveryConfig


def by_name(results):
    return {r.name: r for r in results}


@pytest.fixture(scope="module")
def default_graph():
    return explore_controller(DEFAULT_CONTROLLER_SENSORS, RecoveryConfig())


def test_explore_reaches_fixpoint():
    sensors = ("camera:CAM_FRONT", "camera:CAM_BACK")
    g1 = explore_controller(sensors, RecoveryConfig())
    g2 = explore_controller(sensors, RecoveryConfig())
    assert 1 < len(g1.states) < 100_000 and set(g1.states) == set(g2.states) and len(g1.edges) == len(g2.edges)


def test_controller_properties_hold(default_graph):
    res = by_name(verify_controller(default_graph, RecoveryConfig()))
    assert set(res) == {"RC1", "RC2", "RC3", "RC4", "RC5", "RC6"}
    for name, r in res.items():
        assert r.holds and r.n_cases > 0, (name, r.counterexample)


def test_c2_fails_without_reobserve():
    cfg = RecoveryConfig(reobserve=False)
    g = explore_controller(("camera:CAM_FRONT", "lidar", "camera:CAM_BACK"), cfg)
    c2 = by_name(verify_controller(g, cfg))["RC2"]
    assert not c2.holds and c2.counterexample is not None


def test_shield_properties_hold():
    res = by_name(verify_shield())
    for name in ("SH1", "SH2", "SH3", "SH4", "SH5", "SH7"):
        assert res[name].holds, (name, res[name].counterexample)
        assert res[name].n_cases > 0 and res[name].counterexample is None


def test_s6_without_correct_detection_has_counterexamples():
    res = by_name(verify_shield())
    false_alarm, missed = res["SH6a"], res["SH6b"]
    assert not false_alarm.holds and false_alarm.counterexample["label"]["fault_id"] == "none"
    assert not missed.holds and missed.counterexample["result"]["final_state"] == "healthy"
    assert missed.counterexample["action"] == "continue"
    assert {"label", "result", "action", "acceptable"} <= set(missed.counterexample)
