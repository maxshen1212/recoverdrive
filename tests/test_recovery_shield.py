import pytest

from recoverdrive.metrics import e2_counts, recovery_success
from recoverdrive.recovery import RecoveryConfig, RecoveryController, no_recovery, vote
from recoverdrive.shield import acceptable_actions, classify_action, shield, ttc_override
from recoverdrive.types import Action, Diagnosis, FaultLabel, RecoveryResult


def res(state, isolated=()):
    return RecoveryResult([], state, list(isolated), None)


def test_shield_table():
    assert shield(res("healthy")).action == "continue"
    assert shield(res("degraded", ["camera:CAM_FRONT"])).action == "slow_down"  # LiDAR still sees ahead
    assert shield(res("degraded", ["camera:CAM_BACK"])).action == "slow_down"
    assert shield(res("degraded", ["camera:CAM_FRONT", "lidar"])).action == "stop"
    assert shield(res("degraded", ["agent"])).action == "slow_down"
    assert shield(res("unresolved")).action == "stop"
    assert ttc_override(Action("continue", ""), 1.5).action == "stop"
    assert ttc_override(Action("continue", ""), 3.0).action == "continue"


def test_acceptable_actions_rules():
    assert acceptable_actions(FaultLabel.none()) == ("continue",)
    assert acceptable_actions(FaultLabel("A1", "agent", "high")) == ("continue", "slow_down")
    assert acceptable_actions(FaultLabel("L1", "lidar", "low")) == ("continue", "slow_down")
    assert acceptable_actions(FaultLabel("C3", "camera:CAM_BACK", "high")) == ("continue", "slow_down")
    assert acceptable_actions(FaultLabel("L1", "lidar", "medium")) == ("slow_down", "stop")
    assert acceptable_actions(FaultLabel("C2", "camera:CAM_FRONT", "medium")) == ("slow_down", "stop")
    assert acceptable_actions(FaultLabel("C1", "camera:CAM_FRONT", None)) == ("slow_down", "stop")  # fixed high
    assert acceptable_actions(FaultLabel("L4", "lidar", None)) == ("slow_down", "stop")


def test_classify_action():
    assert classify_action("stop", ("continue",)) == "unnecessary"
    assert classify_action("slow_down", ("continue", "slow_down")) == "ok"
    assert classify_action("continue", ("slow_down", "stop")) == "missed"
    assert classify_action("stop", ("slow_down", "stop")) == "ok"


def d(status, sensor="none", conf=0.9):
    return Diagnosis(status=status, sensor=sensor, confidence=conf)


def test_vote():
    diag, conf = vote([d("fault", "lidar"), d("ok"), d("fault", "lidar")])
    assert diag.status == "fault" and diag.sensor == "lidar" and conf == pytest.approx(2 / 3)
    diag, conf = vote([d("fault", "lidar"), d("ok"), d("uncertain", "lidar")])
    assert diag.status == "fault" and conf == pytest.approx(1 / 3)  # three-way tie -> earliest


def test_recovery_isolates_confident_fault():
    rc = RecoveryController(RecoveryConfig(conf_threshold=0.6))
    r = rc.step(d("fault", "camera:CAM_FRONT", 2 / 3), True, {"camera:CAM_FRONT"}, ["resample"])
    assert r.final_state == "degraded" and r.isolated_sensors == ["camera:CAM_FRONT"]
    assert "isolate:camera:CAM_FRONT" in r.steps_taken


def test_recovery_low_conf_pending_then_persistent():
    rc = RecoveryController(RecoveryConfig(conf_threshold=0.6))
    first = rc.step(d("fault", "lidar", 1 / 3), True, {"lidar"}, [])
    assert first.final_state == "unresolved" and first.isolated_sensors == []
    second = rc.step(d("fault", "lidar", 1 / 3), True, {"lidar"}, [])  # re-observed in the next keyframe
    assert second.final_state == "degraded" and second.isolated_sensors == ["lidar"]


def test_recovery_restores_after_clean_frame():
    rc = RecoveryController()
    rc.step(d("fault", "lidar", 1.0), True, {"lidar"}, [])
    kept = rc.step(d("ok", conf=0.9), True, {"lidar"}, [])  # flag still raised -> stays isolated
    assert kept.isolated_sensors == ["lidar"] and kept.final_state == "degraded"
    back = rc.step(d("ok", conf=0.9), True, set(), [])
    assert back.final_state == "healthy" and back.isolated_sensors == []
    assert "reobserve:restored:lidar" in back.steps_taken


def test_recovery_agent_failure_falls_back_to_rules():
    rc = RecoveryController()
    r = rc.step(None, False, set(), ["reask", "reask"])
    assert r.final_state == "degraded" and r.isolated_sensors == ["agent"] and shield(r).action == "slow_down"
    r2 = RecoveryController().step(None, False, {"camera:CAM_FRONT"}, [])
    assert r2.isolated_sensors == ["agent", "camera:CAM_FRONT"] and r2.final_state == "degraded"


def test_recovery_unresolved_when_remaining_flagged():
    rc = RecoveryController()
    r = rc.step(d("fault", "camera:CAM_FRONT", 1.0), True, {"camera:CAM_FRONT", "lidar"}, [])
    assert r.final_state == "unresolved" and shield(r).action == "stop"


def test_recovery_uncertain_without_flags_is_healthy():
    assert RecoveryController().step(d("uncertain", "lidar", 0.5), True, set(), []).final_state == "healthy"
    assert no_recovery(d("uncertain", "lidar"), True).final_state == "unresolved"
    assert no_recovery(d("ok"), True).final_state == "healthy"
    assert no_recovery(None, False).final_state == "unresolved"


def frame(idx, fault="L1", sensor="lidar", sev="medium", ep=0, state="degraded", iso=("lidar",), action="slow_down"):
    label = {"fault_id": fault, "sensor": sensor, "severity": sev, "episode_id": ep}
    if fault == "none":
        label = {"fault_id": "none", "sensor": "none", "severity": None, "episode_id": None}
    return ({"scene": "s", "seed": 0, "run_fault": "L1", "frame_idx": idx, "label": label},
            {"action": {"action": action}, "recovery": {"final_state": state, "isolated_sensors": list(iso)}})


def test_e2_frames_to_recover():
    pairs = [frame(0, "none", state="healthy", iso=(), action="continue"),
             frame(1, state="unresolved", iso=(), action="stop"),  # episode 0: recovered on its 2nd frame
             frame(2), frame(3),
             frame(4, "none", state="healthy", iso=(), action="stop"),  # unnecessary stop on a clean frame
             frame(5, ep=1, state="unresolved", iso=(), action="continue"),  # episode 1: never recovered, missed
             frame(6, ep=1, state="unresolved", iso=(), action="stop")]
    row = e2_counts(pairs, with_recovery=True)
    assert row["rec_success"] == pytest.approx(2 / 5)
    assert row["ep_recovered"] == 0.5 and row["frames_to_recover"] == 2
    assert row["unnecessary"] == pytest.approx(1 / 7) and row["missed"] == pytest.approx(1 / 7)
    assert row["unnecessary_clean"] == 0.5 and row["missed_fault"] == pytest.approx(1 / 5)
    assert e2_counts(pairs, with_recovery=False)["rec_success"] is None
    assert recovery_success(*frame(2)) and not recovery_success(*frame(2, iso=("lidar", "camera:CAM_FRONT")))


# --- Phase 5 ablation switches ------------------------------------------------------------------------------

def test_controller_no_isolation():
    rc = RecoveryController(RecoveryConfig(isolate=False))
    r = rc.step(d("fault", "lidar", 1.0), True, {"lidar"}, [])
    assert r.final_state == "unresolved" and r.isolated_sensors == [] and shield(r).action == "stop"
    assert rc.step(None, False, set(), ["reask", "reask"]).final_state == "unresolved"
    assert rc.step(d("ok"), True, set(), []).final_state == "healthy"


def test_controller_no_reobserve():
    full, ablated = RecoveryController(), RecoveryController(RecoveryConfig(reobserve=False))
    for rc in (full, ablated):
        rc.step(d("fault", "lidar", 1.0), True, {"lidar"}, [])
    assert full.step(d("ok"), True, {"lidar"}, []).isolated_sensors == ["lidar"]  # flag still up: kept isolated
    assert ablated.step(d("ok"), True, {"lidar"}, []).final_state == "healthy"  # no memory of the isolation
    weak = RecoveryController(RecoveryConfig(reobserve=False))
    weak.step(d("fault", "lidar", 1 / 3), True, {"lidar"}, [])
    assert weak.step(d("fault", "lidar", 1 / 3), True, {"lidar"}, []).final_state == "unresolved"  # never promoted
